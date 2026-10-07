"""Core vision pipeline: align two fixed-angle photos, segment the object,
measure area / directional / volume-proxy changes. Only cv2 + numpy."""
import cv2
import numpy as np
import math


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


def _norm_color(img, max_side=1200):
    """نرمال‌سازی عکس رنگی به فضای مختصات overlay تحلیل (max_side=1200)."""
    s = min(1.0, max_side / max(img.shape[:2]))
    if s < 1.0:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
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
    return _segment_grabcut_arr(img, inset, max_side, iters)


def segment_grabcut_arr(img, inset=0.08, max_side=1200, iters=4):
    """سگمنت‌گذاری GrabCut روی آرایه (به جای مسیر فایل)."""
    return _segment_grabcut_arr(img, inset, max_side, iters)


def _segment_grabcut_arr(img, inset=0.08, max_side=1200, iters=4):
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
    # باگ: GrabCut به‌طور پیش‌فرض غیرقطعی است (GMM random init) و تا ۹٪
    # در مساحت سوژه نوسان میکند → اعداد تحلیل هر بار متفاوت میشد.
    # seed ثابت → نتایج قابل بازتولید
    cv2.setRNGSeed(42)
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


def make_overlay_from_arrays(im1, im2):
    """هسته مشترک overlay از آرایه‌های هم‌اندازه (هم‌راستا + Lab + Otsu).

    ورودی‌ها باید هم‌اندازه باشند. برمی‌گرداند:
    (vis, regions_full) که regions_full در مختصات همان ابعاد است.
    """

    # ── 1) Align با ORB + Homography ──
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    h, w = im1.shape[:2]
    im2_aligned, valid_mask, H = _align_homography(im2, g1, g2, w, h)

    # ── 2) مقایسه رنگی در Lab (فقط کانال‌های a,b) ──
    lab1 = cv2.cvtColor(im1, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab2 = cv2.cvtColor(im2_aligned, cv2.COLOR_BGR2Lab).astype(np.float32)
    color_diff = np.sqrt((lab1[:, :, 1] - lab2[:, :, 1]) ** 2
                         + (lab1[:, :, 2] - lab2[:, :, 2]) ** 2)

    # ── 3) Blur + threshold ──
    color_diff = cv2.GaussianBlur(color_diff, (15, 15), 0)
    color_diff = np.where(valid_mask > 0, color_diff, 0)
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

    return vis, growth, shrink, int(growth.sum()), int(shrink.sum())


def _read_norm_pair(baseline_path, followup_path, max_side=1200):
    """خواندن دو عکس و نرمال‌سازی به یک فضای مختصات (max_side).

    عکس دوم به ابعاد عکس اول resize می‌شود. برمی‌گرداند (im1, im2).
    همه خروجی‌های نمایشی (overlay تحلیل، overlay پیشنمایش، مختصات
    regions) در همین فضا هستند تا شماره‌ها دقیقاً روی ناحیه تغییر بیفتند.
    """
    im1 = _imread_any(baseline_path, cv2.IMREAD_COLOR)
    im2 = _imread_any(followup_path, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise ValueError("cannot read images")
    im1 = _norm_color(im1, max_side)
    im2 = _norm_color(im2, max_side)
    h, w = im1.shape[:2]
    if im2.shape[:2] != (h, w):
        im2 = cv2.resize(im2, (w, h), interpolation=cv2.INTER_AREA)
    return im1, im2


def make_overlay(baseline_path, followup_path, inset=0.08):
    """پیشنمایش overlay (Lab + Otsu) در فضای مختصات نرمال‌شده تحلیل."""
    im1, im2 = _read_norm_pair(baseline_path, followup_path)
    vis, _, _, _, _ = make_overlay_from_arrays(im1, im2)
    return vis


def _align_homography(im2, g1, g2, w, h):
    """هم‌راستاسازی im2 روی g1 با ORB + Homography + RANSAC.

    برمیگرداند: (im2_aligned, valid_mask, H)
    valid_mask: نواحی از تصویر که داده معتبر دارن (حاشیه‌های warp شده مستثنی شدن)
    """
    orb = cv2.ORB_create(nfeatures=8000, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)

    im2_aligned = im2.copy()
    valid_mask = np.full((h, w), 255, dtype=np.uint8)
    H = None

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return im2_aligned, valid_mask, H

    # Lowe's ratio test — دقیق‌تر از crossCheck
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return im2_aligned, valid_mask, H

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
    if H is None:
        return im2_aligned, valid_mask, H

    im2_aligned = cv2.warpPerspective(im2, H, (w, h), flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)

    # ماسک ناحیه معتبر: گوشه‌های im2 تبدیل میشن به مختصات im1
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_mapped = cv2.perspectiveTransform(corners, H)
    valid_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(valid_mask, [np.int32(corners_mapped)], 255)
    # حاشیه داخلی برای دقت لبه‌ها
    ins = 15
    valid_mask[:ins, :] = 0
    valid_mask[-ins:, :] = 0
    valid_mask[:, :ins] = 0
    valid_mask[:, -ins:] = 0
    return im2_aligned, valid_mask, H


def extract_regions(baseline_path, followup_path, min_area=200, inset=0.08):
    """استخراج مناطق تغییر به صورت اجزای مجزا برای گزارش تفکیکی.

    مختصات در همان فضای overlay تحلیل است (نرمال‌شده به max_side=1200)
    تا شماره‌ها روی تصویر همپوشانی دقیق بنشینند.

    برمیگرداند: (regions, stats)
    regions: لیست دیکشنری با شماره، مساحت، مرکز، جهت، درصد
    stats: مساحت کل رشد/کاهش
    """
    im1, im2 = _read_norm_pair(baseline_path, followup_path)
    h, w = im1.shape[:2]
    _, growth, shrink, total_growth_px, total_shrink_px = make_overlay_from_arrays(im1, im2)

    regions = []
    # اجزای مجزا در ماسک رشد و کاهش
    growth_u8 = growth.astype(np.uint8) * 255
    shrink_u8 = shrink.astype(np.uint8) * 255
    for rtype, mask_u8 in (("growth", growth_u8), ("shrink", shrink_u8)):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2
            cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2
            # مرکز کانتور برای جهت
            mask_i = (labels == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            # جهت: زاویه مرکز نسبت به مرکز تصویر
            img_cx, img_cy = w / 2, h / 2
            dx, dy = cx - img_cx, img_cy - cy  # y معکوس چون بالا=منفی
            ang = math.degrees(math.atan2(dy, dx)) % 360
            dirs = ["شرق", "شمالشرق", "شمال", "شمالغرب", "غرب", "جنوبغرب", "جنوب", "جنوبشرق"]
            dir_idx = int(((ang + 22.5) % 360) / 45)
            direction = dirs[dir_idx]

            total_px = total_growth_px if rtype == "growth" else total_shrink_px
            pct = round(area / total_px * 100, 1) if total_px else 0.0
            regions.append({
                "area_px": area,
                "cx": round(cx, 0),
                "cy": round(cy, 0),
                "direction": direction,
                "pct": pct,
                "type": rtype,
                "sign": "+" if rtype == "growth" else "-",
                "bbox": [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                         int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])],
            })

    # مرتبسازی نزولی بر اساس مساحت
    regions.sort(key=lambda r: r["area_px"], reverse=True)
    # فقط ۱۰ نقطه با بزرگترین تغییرات
    regions = regions[:10]
    for i, r in enumerate(regions, 1):
        r["number"] = i

    stats_out = {
        "total_growth_px": total_growth_px,
        "total_shrink_px": total_shrink_px,
        "region_count": len(regions),
    }
    return regions, stats_out


def _union_contours(mask):
    """اجتماع همه کانتورهای ≥۵٪ بزرگ‌ترین قطعه — سوژه چندتکه از دست نره."""
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    big = max(cv2.contourArea(cnts[0]) * 0.05, 50)
    keep = [c for c in cnts if cv2.contourArea(c) >= big]
    out = np.zeros_like(m)
    cv2.drawContours(out, keep, -1, 1, thickness=cv2.FILLED)
    return out


def volume_proxy_in_bbox(mask_full, bbox):
    """حجم تخمینی محدود به bbox: جمع pi*(w/2)^2 روی سطرهای داخل ناحیه."""
    x, y, w, h = [int(v) for v in bbox]
    H, W = mask_full.shape[:2]
    x0, x1 = max(0, x), min(W, x + w)
    y0, y1 = max(0, y), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    vol = 0.0
    for yy in range(y0, y1):
        row = np.where(mask_full[yy, x0:x1] > 0)[0]
        if row.size < 2:
            continue
        wd = float(row.max() - row.min())
        vol += np.pi * (wd / 2.0) ** 2
    return vol


_AREA_ZERO = 0.5
_AREA_TINY = 200.0


def _area_change(before, after):
    """نوع تغییر از روی مساحت اندازه‌گیری‌شده (مرجع در برابر دوم)، نه ماسک رنگ.

    برمی‌گرداند: (type, sign, change_pct|None, kind, label|None)
    kind یکی از: pct | emerged | removed | empty | tiny_base
    """
    b = float(before)
    a = float(after)
    if b <= _AREA_ZERO and a <= _AREA_ZERO:
        return ("none", "", None, "empty", "—")
    if b <= _AREA_ZERO:
        return ("growth", "+", None, "emerged", "ظهور (نقطه جدید)")
    if a <= _AREA_ZERO:
        return ("shrink", "−", None, "removed", "حذف کامل")
    if b < _AREA_TINY:
        if a > b:
            return ("growth", "+", None, "tiny_base", "— (مبنای کوچک)")
        if a < b:
            return ("shrink", "−", None, "tiny_base", "— (مبنای کوچک)")
        return ("none", "", None, "empty", "—")
    if a > b:
        return ("growth", "+", round((a - b) / b * 100, 2), "pct", None)
    if a < b:
        return ("shrink", "−", round((a - b) / b * 100, 2), "pct", None)
    return ("none", "", 0.0, "pct", None)


def _vol_change(before, after):
    """برچسب تغییر حجم؛ آستانه «مبنای کوچک» ندارد (واحد px³ با مساحت فرق دارد)."""
    b = float(before)
    a = float(after)
    if b <= _AREA_ZERO and a <= _AREA_ZERO:
        return (None, "empty", "—")
    if b <= _AREA_ZERO:
        return (None, "emerged", "ظهور")
    if a <= _AREA_ZERO:
        return (None, "removed", "حذف")
    return (round((a - b) / b * 100, 2) if b > 1e-9 else 0.0, "pct", None)


def _analyze_scoped(baseline_path, followup_path, px_per_mm, inset, selected):
    """تحلیل محدود به نقاط انتخاب‌شده کاربر (شماره‌های extract_regions).

    مساحت مرجع/دوم هر نقطه = پیکسل‌های سوژه (GrabCut) داخل bbox همان نقطه؛
    ماسک دوم با H هم‌راستا می‌شود تا مقایسه در یک قاب باشد (مثل overlay).
    """
    try:
        sel = sorted({int(s) for s in selected if str(s).strip()})
    except Exception:
        sel = []
    if not sel:
        raise ValueError("اول نقطه‌ای برای تحلیل انتخاب کن")
    regions_all, _ = extract_regions(baseline_path, followup_path)
    regions = [r for r in regions_all if r["number"] in sel]
    if not regions:
        raise ValueError("نقاط انتخاب‌شده معتبر نیست؛ دوباره «نمایش همپوشانی» را بزن")

    im1, im2 = _read_norm_pair(baseline_path, followup_path)
    h1, w1 = im1.shape[:2]
    max_side = 1200
    m1 = segment_grabcut_arr(im1, inset=inset, max_side=max_side + 1)
    m2 = segment_grabcut_arr(im2, inset=inset, max_side=max_side + 1)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    _, _, H = _align_homography(im2, g1, g2, w1, h1)
    m2w = cv2.warpPerspective(m2, H, (w1, h1), flags=cv2.INTER_NEAREST) if H is not None else m2
    m1_full = _union_contours(m1)
    m2_full = _union_contours(m2w)
    if m1_full is None or m2_full is None:
        raise ValueError("سوژه پیدا نشد؛ inset را تغییر بده یا عکس با پس‌زمینه خلوت‌تر بگیر")

    def pct(a, b):
        return (b - a) / a * 100 if a > 1e-9 else 0.0

    out_regions = []
    for r in regions:
        bx, by, bw, bh = [int(v) for v in r["bbox"]]
        x0, x1 = max(0, bx), min(w1, bx + bw)
        y0, y1 = max(0, by), min(h1, by + bh)
        a_before = float(m1_full[y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0.0
        a_after = float(m2_full[y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0.0
        v_before = volume_proxy_in_bbox(m1_full, r["bbox"])
        v_after = volume_proxy_in_bbox(m2_full, r["bbox"])
        rtype, sign, cpct, ckind, clabel = _area_change(a_before, a_after)
        vpct, vkind, vlabel = _vol_change(v_before, v_after)
        if ckind in ("tiny_base", "empty"):
            # مبنای مساحت غیرقابل اتکاست → درصد حجم هم اعتبار ندارد
            vpct, vkind, vlabel = None, ckind, clabel
        entry = {
            "number": r["number"], "type": rtype, "sign": sign,
            "chroma_type": r["type"], "chroma_sign": r["sign"],
            "direction": r["direction"], "pct": r["pct"],
            "cx": r["cx"], "cy": r["cy"], "bbox": r["bbox"],
            "area_px": {"before": round(a_before, 1), "after": round(a_after, 1),
                        "change_pct": cpct, "change_kind": ckind,
                        "change_label": clabel},
            "volume_proxy_px3": {"before": round(v_before, 1), "after": round(v_after, 1),
                                 "change_pct": vpct, "change_kind": vkind,
                                 "change_label": vlabel},
        }
        if px_per_mm and px_per_mm > 0:
            entry["area_mm2"] = {"before": round(a_before / px_per_mm ** 2, 2),
                                 "after": round(a_after / px_per_mm ** 2, 2)}
            entry["volume_proxy_mm3"] = {"before": round(v_before / px_per_mm ** 3, 2),
                                         "after": round(v_after / px_per_mm ** 3, 2)}
        out_regions.append(entry)

    scoped_regions = [e for e in out_regions
                      if e["area_px"].get("change_kind") != "empty"]
    tot_ab = sum(e["area_px"]["before"] for e in scoped_regions)
    tot_aa = sum(e["area_px"]["after"] for e in scoped_regions)
    tot_vb = sum(e["volume_proxy_px3"]["before"] for e in scoped_regions)
    tot_va = sum(e["volume_proxy_px3"]["after"] for e in scoped_regions)
    _, _, tot_cpct, tot_ckind, tot_clabel = _area_change(tot_ab, tot_aa)
    tot_vpct, tot_vkind, tot_vlabel = _vol_change(tot_vb, tot_va)
    res = {
        "scoped": True,
        "selected": [e["number"] for e in out_regions],
        "regions": out_regions,
        "area_px": {"before": round(tot_ab, 1), "after": round(tot_aa, 1),
                    "change_pct": tot_cpct, "change_kind": tot_ckind,
                    "change_label": tot_clabel},
        "volume_proxy_px3": {"before": round(tot_vb, 1), "after": round(tot_va, 1),
                             "change_pct": tot_vpct, "change_kind": tot_vkind,
                             "change_label": tot_vlabel},
        "directions": [],
        "note_fa": ("تحلیل محدود به نقاط انتخاب‌شده است؛ مساحت و حجم هر نقطه فقط "
                    "داخل همان ناحیه سنجیده شده و بقیه تصویر نادیده گرفته شده است. "
                    "ستون «نوع» از مقایسه مساحت مرجع و دوم به‌دست می‌آید "
                    "(نه از رنگ اختلاف)؛ «ظهور» یعنی نقطه در مرجع نبود، "
                    "«حذف» یعنی در دوم نیست، و «مبنای کوچک» یعنی مرجع زیر ۲۰۰px "
                    "است و درصد قابل اتکا نیست."),
    }
    if px_per_mm and px_per_mm > 0:
        res["scale"] = {"px_per_mm": px_per_mm,
                        "area_mm2": {"before": round(tot_ab / px_per_mm ** 2, 2),
                                     "after": round(tot_aa / px_per_mm ** 2, 2)},
                        "volume_proxy_mm3": {"before": round(tot_vb / px_per_mm ** 3, 2),
                                             "after": round(tot_va / px_per_mm ** 3, 2)}}
    # overlay پایه (مثل تحلیل کلی) — شماره‌ها در فرانت‌اند رسم می‌شن
    m1_ov = cv2.resize(m1, (w1, h1), interpolation=cv2.INTER_NEAREST)
    m2_ov = cv2.resize(m2, (w1, h1), interpolation=cv2.INTER_NEAREST)
    if H is not None:
        m2_ov = cv2.warpPerspective(m2_ov, H, (w1, h1), flags=cv2.INTER_NEAREST)
    c1_ov = largest_contour(m1_ov)
    c2_ov = largest_contour(m2_ov)
    overlay = draw_overlay(g1, m1_ov, m2_ov, c1_ov, c2_ov)
    return res, overlay, (m1, m2)


def analyze(baseline_path, followup_path, px_per_mm=None, inset=0.08, selected=None):
    """تحلیل کامل یا محدود به نقاط انتخابی (selected = شماره‌های extract_regions).

    selected=None → رفتار قبلی (کل سوژه). در غیر این صورت فقط نقاط انتخابی
    تحلیل می‌شن و بدون انتخاب معتبر خطا میده.
    """
    if selected is not None:
        return _analyze_scoped(baseline_path, followup_path, px_per_mm, inset, selected)
    im1 = _imread_any(baseline_path, cv2.IMREAD_COLOR)
    im2 = _imread_any(followup_path, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise ValueError("cannot read images")

    # نرمال‌سازی هر دو عکس به max_side یکسان — GrabCut و metrics روی این سایز
    # کار میکنن تا ماسک و gray همیشه هم‌اندازه بمونن (باگ سایز ماسک)
    max_side = 1200

    def _norm(img):
        s = min(1.0, max_side / max(img.shape[:2]))
        if s < 1.0:
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                             interpolation=cv2.INTER_AREA)
        return img

    im1, im2 = _norm(im1), _norm(im2)
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    # هم‌راستاسازی ORB + Homography (۱۱ برابر دقیق‌تر از ECC affine)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    if (h2, w2) != (h1, w1):
        im2_r = cv2.resize(im2, (w1, h1), interpolation=cv2.INTER_AREA)
        g2_r = cv2.cvtColor(im2_r, cv2.COLOR_BGR2GRAY)
    else:
        im2_r, g2_r = im2, g2
    im2_aligned, valid_mask, H = _align_homography(im2_r, g1, g2_r, w1, h1)

    # سگمنت‌گذاری مستقل روی هر عکس در مختصات خودش — resize به یک سایز
    # باعث کشیدگی و اندازه‌گیری غلط میشه. هر عکس با aspect اصلی خودش.
    m1 = segment_grabcut_arr(im1, inset=inset, max_side=max_side + 1)
    m2 = segment_grabcut_arr(im2, inset=inset, max_side=max_side + 1)

    # حذف قطعات کوچک نویزی — _union_contours مشترک (اجتماع ≥۵٪ بزرگ‌ترین)
    m1_full = _union_contours(m1)
    m2_full = _union_contours(m2)
    if m1_full is None or m2_full is None:
        raise ValueError("سوژه پیدا نشد؛ inset را تغییر بده یا عکس با پس‌زمینه خلوت‌تر بگیر")

    c1 = largest_contour(m1)
    c2 = largest_contour(m2)
    if c1 is None or c2 is None:
        raise ValueError("سوژه پیدا نشد؛ inset را تغییر بده یا عکس با پس‌زمینه خلوت‌تر بگیر")

    a1 = float(m1_full.sum())
    a2 = float(m2_full.sum())
    bx1, by1, bw1, bh1 = cv2.boundingRect(c1)
    bx2, by2, bw2, bh2 = cv2.boundingRect(c2)
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

    v1 = volume_proxy_solid_of_revolution(m1_full)
    v2 = volume_proxy_solid_of_revolution(m2_full)

    def pct(a, b):
        return (b - a) / a * 100 if a > 1e-9 else 0.0

    res = {
        "area_px": {"before": round(a1, 1), "after": round(a2, 1),
                    "change_pct": round(pct(a1, a2), 2)},
        "bbox_px": {"before_wh": [bw1, bh1], "after_wh": [bw2, bh2],
                    "width_change_pct": round(pct(bw1, bw2), 2),
                    "height_change_pct": round(pct(bh1, bh2), 2)},
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
    # overlay: ماسک‌ها رو هم‌اندازه‌ی im1 میکنیم و m2 رو با H هم‌راستا
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    m1_ov = cv2.resize(m1, (w1, h1), interpolation=cv2.INTER_NEAREST)
    m2_ov = cv2.resize(m2, (w1, h1), interpolation=cv2.INTER_NEAREST)
    if H is not None:
        m2_ov = cv2.warpPerspective(m2_ov, H, (w1, h1), flags=cv2.INTER_NEAREST)
    c1_ov = largest_contour(m1_ov)
    c2_ov = largest_contour(m2_ov)
    overlay = draw_overlay(g1, m1_ov, m2_ov, c1_ov, c2_ov)
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
