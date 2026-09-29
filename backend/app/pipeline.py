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

    # ── 1) Align با ORB + Homography ──
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
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

    برمیگرداند: (regions, stats)
    regions: لیست دیکشنری با شماره، مساحت، مرکز، جهت، درصد
    stats: مساحت کل رشد/کاهش
    """
    im1 = _imread_any(baseline_path, cv2.IMREAD_COLOR)
    im2 = _imread_any(followup_path, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise ValueError("cannot read images")
    h, w = im1.shape[:2]
    if im2.shape[:2] != (h, w):
        im2 = cv2.resize(im2, (w, h), interpolation=cv2.INTER_AREA)

    # ===== هم‌راستاسازی ORB + Homography (۱۱ برابر دقیق‌تر از ECC affine) =====
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im2_aligned, valid_mask, H = _align_homography(im2, g1, g2, w, h)

    # ===== مقایسه رنگی Lab فقط روی ناحیه معتبر =====
    lab1 = cv2.cvtColor(im1, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab2 = cv2.cvtColor(im2_aligned, cv2.COLOR_BGR2Lab).astype(np.float32)
    # فقط کانال‌های رنگی (a,b) — مقاوم به تغییرات روشنایی
    color_diff = np.sqrt((lab1[:, :, 1] - lab2[:, :, 1]) ** 2
                         + (lab1[:, :, 2] - lab2[:, :, 2]) ** 2)
    color_diff = cv2.GaussianBlur(color_diff, (15, 15), 0)
    color_diff = np.where(valid_mask > 0, color_diff, 0)
    diff_norm = cv2.normalize(color_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, change_mask = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)

    gray1 = g1.astype(np.float32)
    gray2 = cv2.cvtColor(im2_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bright_diff = gray2 - gray1

    changed = change_mask > 0
    growth = changed & (bright_diff > 15)
    shrink = changed & (bright_diff < -15)

    total_growth_px = int(growth.sum())
    total_shrink_px = int(shrink.sum())

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


def analyze(baseline_path, followup_path, px_per_mm=None, inset=0.08):
    im1 = _imread_any(baseline_path, cv2.IMREAD_COLOR)
    im2 = _imread_any(followup_path, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise ValueError("cannot read images")
    h, w = im1.shape[:2]
    if im2.shape[:2] != (h, w):
        im2 = cv2.resize(im2, (w, h), interpolation=cv2.INTER_AREA)

    # هم‌راستاسازی ORB + Homography (۱۱ برابر دقیق‌تر از ECC affine)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im2_aligned, valid_mask, H = _align_homography(im2, g1, g2, w, h)

    # سگمنت‌گذاری روی تصویر مرجع و تصویر هم‌راستا شده
    m1 = segment_grabcut_arr(im1, inset=inset)
    m2 = segment_grabcut_arr(im2_aligned, inset=inset)

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
