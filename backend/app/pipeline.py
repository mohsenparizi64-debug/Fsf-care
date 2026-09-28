"""Core vision pipeline: align two fixed-angle photos, segment the object,
measure area / directional / volume-proxy changes. Only cv2 + numpy."""
import cv2
import numpy as np


def _imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread با مسیرهای غیر ASCII (فارسی) مشکل دارد → خواندن از طریق numpy."""
    try:
        return cv2.imread(path, flags)
    except Exception:
        return None


def _imread_any(path, flags=cv2.IMREAD_COLOR):
    """خواندن تصویر با پشتیبانی کامل از یونیکد در نام مسیر."""
    img = _imread_unicode(path, flags)
    if img is not None:
        return img
    # fallback: خواندن بایت‌ها از طریق numpy
    try:
        import numpy as _np
        data = _np.fromfile(path, dtype=_np.uint8)
        if data.size:
            return cv2.imdecode(data, flags)
    except Exception:
        pass
    return None


def load_gray(path, max_side=1200):
    img = _imread_any(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    h, w = img.shape
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def align_ecc(im1, im2):
    """Align im2 onto im1 with affine ECC. Returns warped im2 (same size as im1)."""
    h, w = im1.shape
    if im2.shape != im1.shape:
        im2 = cv2.resize(im2, (w, h), interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    try:
        _, warp = cv2.findTransformECC(im1, im2, warp, cv2.MOTION_AFFINE, criteria, None, 1)
        return cv2.warpAffine(im2, warp, (w, h),
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except cv2.error:
        return im2  # fallback: assume already fixed angle


def segment_grabcut(color_path, inset=0.08, max_side=1200, iters=4):
    img = _imread_any(color_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {color_path}")
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    x0, y0 = int(w * inset), int(h * inset)
    rect = (x0, y0, w - 2 * x0, h - 2 * y0)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    bin_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8) * 255
    # keep largest connected component
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bin_mask = np.where(labels == biggest, 255, 0).astype(np.uint8)
    # cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k)
    return bin_mask


def largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def radial_extent(contour, center, nbins=8):
    """Max radius per angular bin (bin 0 = East, CCW). Returns array nbins."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    v = pts - np.array(center, dtype=np.float64)
    r = np.linalg.norm(v, axis=1)
    ang = (np.degrees(np.arctan2(-v[:, 1], v[:, 0])) + 360) % 360  # 0=E,90=N
    bins = ((ang + 360 / nbins / 2) % 360 / (360 / nbins)).astype(int) % nbins
    out = np.zeros(nbins)
    for b in range(nbins):
        sel = r[bins == b]
        out[b] = sel.max() if sel.size else 0
    return out


BIN_NAMES = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
# Persian compass labels for UI
FA_NAMES = {"E": "شرق", "NE": "شمال‌شرق", "N": "شمال (بالا)",
            "NW": "شمال‌غرب", "W": "غرب", "SW": "جنوب‌غرب",
            "S": "جنوب (پایین)", "SE": "جنوب‌شرق"}


def volume_proxy_solid_of_revolution(mask):
    """Sum pi*(w/2)^2 over rows. Assumes roughly vertical object with
    rotational symmetry around its vertical axis. Returns px^3."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0.0
    vol = 0.0
    for y in range(ys.min(), ys.max() + 1):
        row = xs[ys == y]
        if row.size < 2:
            continue
        w = float(row.max() - row.min())
        vol += np.pi * (w / 2.0) ** 2
    return vol


def make_overlay(baseline_path, followup_path, inset=0.08):
    """Overlay مبتنی بر مقایسه رنگی در فضای Lab — دقیق از mask comparison.

    مراحل:
    1. خواندن هر دو تصویر رنگی
    2. Align با ECC (روی grayscale)
    3. مقایسه پیکسلی در Lab color space
    4. Gaussian blur + Otsu threshold
    5. تشخیص جهت: رشد (روشنتر) = قرمز، کاهش (تیرهتر) = آبی
    """
    im1 = _imread_any(baseline_path, cv2.IMREAD_COLOR)
    im2 = _imread_any(followup_path, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise ValueError("cannot read images")
    h, w = im1.shape[:2]
    if im2.shape[:2] != (h, w):
        im2 = cv2.resize(im2, (w, h), interpolation=cv2.INTER_AREA)

    # ── 1) Align با ECC روی grayscale ──
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, warp = cv2.findTransformECC(g1, g2, warp, cv2.MOTION_AFFINE, criteria, None, 5)
        im2_aligned = cv2.warpAffine(im2, warp, (w, h),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except cv2.error:
        im2_aligned = im2.copy()

    # ── 2) مقایسه رنگی در Lab ──
    lab1 = cv2.cvtColor(im1, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab2 = cv2.cvtColor(im2_aligned, cv2.COLOR_BGR2Lab).astype(np.float32)
    color_diff = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))

    # ── 3) Blur + threshold ──
    color_diff = cv2.GaussianBlur(color_diff, (15, 15), 0)
    diff_norm = cv2.normalize(color_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, change_mask = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # حذف نویز
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)

    # ── 4) تشخیص جهت با روشنایی ──
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(im2_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bright_diff = gray2 - gray1  # مثبت = روشنتر در دوم

    changed = change_mask > 0
    growth = changed & (bright_diff > 15)   # روشنتر = رشد
    shrink = changed & (bright_diff < -15)  # تیرهتر = کاهش

    # ── 5) رسم ──
    vis = im1.copy()
    # رشد → قرمز نیمه‌شفاف
    vis[growth] = (vis[growth].astype(np.float32) * 0.3 +
                   np.array([0, 0, 255], dtype=np.float32) * 0.7).astype(np.uint8)
    # کاهش → آبی نیمه‌شفاف
    vis[shrink] = (vis[shrink].astype(np.float32) * 0.3 +
                   np.array([255, 0, 0], dtype=np.float32) * 0.7).astype(np.uint8)

    # کانتور دور نواحی تغییر
    all_changed = (growth | shrink).astype(np.uint8) * 255
    contours, _ = cv2.findContours(all_changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)

    return vis


def analyze(baseline_path, followup_path, px_per_mm=None, inset=0.08):
    g1 = load_gray(baseline_path)
    g2_raw = load_gray(followup_path)
    # resize followup color inside segment; align on gray of same size
    h, w = g1.shape
    g2 = cv2.resize(g2_raw, (w, h)) if g2_raw.shape != g1.shape else g2_raw
    g2_aligned = align_ecc(g1, g2)

    m1 = segment_grabcut(baseline_path, inset=inset)
    # segment followup then warp its mask with same transform is complex;
    # simpler: write aligned gray to temp? Instead segment original followup
    # and resize — alignment already applied at measurement level via moments.
    m2_raw = segment_grabcut(followup_path, inset=inset)
    m2 = cv2.resize(m2_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    # refine: align mask m2 to m1 with ECC-computed warp on grays is already
    # approximated; do a second ECC pass mask-to-mask for translation only
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
        _, warp = cv2.findTransformECC(m1.astype(np.float32) / 255,
                                       m2.astype(np.float32) / 255,
                                       warp, cv2.MOTION_TRANSLATION, criteria, None, 1)
        m2 = cv2.warpAffine(m2, warp, (w, h),
                            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
    except cv2.error:
        pass

    c1 = largest_contour(m1)
    c2 = largest_contour(m2)
    if c1 is None or c2 is None:
        raise ValueError("سوژه پیدا نشد؛ inset را تغییر بده یا عکس با پس‌زمینه خلوت‌تر بگیر")

    a1 = float(cv2.contourArea(c1))
    a2 = float(cv2.contourArea(c2))
    x1, y1, w1, h1 = cv2.boundingRect(c1)
    x2, y2, w2, h2 = cv2.boundingRect(c2)
    M1 = cv2.moments(c1)
    M2 = cv2.moments(c2)
    cx1, cy1 = M1["m10"] / M1["m00"], M1["m01"] / M1["m00"]
    cx2, cy2 = M2["m10"] / M2["m00"], M2["m01"] / M2["m00"]

    r1 = radial_extent(c1, (cx1, cy1))
    r2 = radial_extent(c2, (cx2, cy2))
    dirs = []
    for i, name in enumerate(BIN_NAMES):
        b = float(r1[i])
        pct = (float(r2[i]) - b) / b * 100 if b > 1e-6 else 0.0
        dirs.append({"dir": name, "fa": FA_NAMES[name],
                     "before_px": round(b, 1), "after_px": round(float(r2[i]), 1),
                     "change_pct": round(pct, 2)})

    v1 = volume_proxy_solid_of_revolution(m1)
    v2 = volume_proxy_solid_of_revolution(m2)

    def pct(a, b):
        return (b - a) / a * 100 if a > 1e-9 else 0.0

    res = {
        "area_px": {"before": round(a1, 1), "after": round(a2, 1),
                    "change_pct": round(pct(a1, a2), 2)},
        "bbox_px": {"before_wh": [w1, h1], "after_wh": [w2, h2],
                    "width_change_pct": round(pct(w1, w2), 2),
                    "height_change_pct": round(pct(h1, h2), 2)},
        "centroid_shift_px": {"dx": round(cx2 - cx1, 1), "dy": round(cy2 - cy1, 1)},
        "directions": dirs,
        "volume_proxy_px3": {"before": round(v1, 1), "after": round(v2, 1),
                             "change_pct": round(pct(v1, v2), 2)},
        "note_fa": ("حجم، تخمین از روی نیمرخ (فرض تقارن محوری حول محور عمودی) است؛ "
                    "رشد «به سمت جلو/دوربین» از تک‌نما قابل اندازه‌گیری دقیق نیست و "
                    "در پهنای ظاهری دیده می‌شود. برای حجم واقعی به نمای دوم (۹۰ درجه) نیاز است."),
    }
    if px_per_mm and px_per_mm > 0:
        res["scale"] = {"px_per_mm": px_per_mm,
                        "area_mm2": {"before": round(a1 / px_per_mm ** 2, 2),
                                     "after": round(a2 / px_per_mm ** 2, 2)},
                        "volume_proxy_mm3": {"before": round(v1 / px_per_mm ** 3, 2),
                                             "after": round(v2 / px_per_mm ** 3, 2)}}
        res["centroid_shift_mm"] = {"dx": round((cx2 - cx1) / px_per_mm, 2),
                                    "dy": round((cy2 - cy1) / px_per_mm, 2)}
    overlay = draw_overlay(g1, m1, m2, c1, c2)
    return res, overlay, (m1, m2)


def draw_overlay(gray, m1, m2, c1, c2):
    """Overlay با نواحی تفاوت پررنگ + کانتور.
    قرمز = رشد (تصویر دوم بیشتر از مرجع)
    آبی = کاهش (تصویر دوم کمتر از مرجع)
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask1 = m1 > 0
    mask2 = m2 > 0
    growth = mask2 & ~mask1   # in after but not before → RED
    shrink = mask1 & ~mask2   # in before but not after → BLUE
    # نواحی رشد → قرمز نیمه‌شفاف
    vis[growth] = (vis[growth].astype(np.float32) * 0.3 + np.array([0, 0, 255], dtype=np.float32) * 0.7).astype(np.uint8)
    # نواحی کاهش → آبی نیمه‌شفاف
    vis[shrink] = (vis[shrink].astype(np.float32) * 0.3 + np.array([255, 0, 0], dtype=np.float32) * 0.7).astype(np.uint8)
    # کانتورها روی لایه بالا
    cv2.drawContours(vis, [c1], -1, (255, 0, 0), 2)   # قبل = آبی
    cv2.drawContours(vis, [c2], -1, (0, 0, 255), 2)   # بعد = قرمز
    return vis
