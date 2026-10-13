# -*- coding: utf-8 -*-
"""Core vision pipeline v2 — تشخیص تغییرات نقاط انتخاب‌شده بین دو عکس.

اصلاحات نسخه ۲ نسبت به نسخه ۱:
  - حذف GrabCut: روی صحنه‌های «سوژه روشن روی پس‌زمینه پرنقش» ماسک برعکس
    می‌داد (۵۳٪/۴۸٪ به‌جای ~۱۳٪) و drift می‌کرد.
  - حذف MINMAX + Otsu: نویز JPEG را به‌عنوان تغییر بزرگ نشان می‌داد.
  - هم‌راستاسازی SIFT + similarity + RANSAC به‌جای ORB+Homography:
    ORB روی بافت تکراری فرش تطبیق کاذب می‌دهد.
  - حذف نور سراسری و محلی با نرمال‌سازی L − blur(L) (نه CLAHE).
  - آستانه مطلق کرومای Lab + آستانه مطلق روشنایی محلی.
  - مسیر روشنایی فقط روی اجسام روشن فعال است.
  - seed ثابت → نتایجات قابل بازتولید.
"""
import cv2
import numpy as np
import math

# ── پارامترها (قابل کالیبراسیون) ──
MAX_SIDE = 1200
SIFT_FEATURES = 5000
MATCH_RATIO = 0.70
RANSAC_THRESH = 3.0
COVERAGE_ERODE = 15
LOCAL_GRID = 24          # نرمال‌سازی محلی روشنایی (حذف نور سراسری + محلی)
CHROMA_MIN = 15.0        # آستانه مطلق کرومای Lab
L_ABS_MIN = 15.0         # آستانه مطلق تغییر روشنایی محلی
L_DIR_MIN = 6.0          # آستانه جهت رشد/کاهش
BRIGHT_MIN = 170         # حداقل روشنایی برای فعال بودن مسیر L
BLUR = 15
MORPH_K = 9
MIN_AREA = 200           # حداقل مساحت یک ناحیه تغییر واقعی
TOP_N = 10

BIN_NAMES = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
FA_NAMES = {"E": "شرق", "NE": "شمال‌شرق", "N": "شمال (بالا)",
            "NW": "شمال‌غرب", "W": "غرب", "SW": "جنوب‌غرب",
            "S": "جنوب (پایین)", "SE": "جنوب‌شرق"}

_AREA_ZERO = 0.5
_AREA_TINY = 200.0


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
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size:
            return cv2.imdecode(data, flags)
    except Exception:
        pass
    return None


def _norm_size(img, max_side=MAX_SIDE):
    """نرمال‌سازی عکس رنگی به max_side یکسان."""
    s = min(1.0, max_side / max(img.shape[:2]))
    if s < 1.0:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
    return img


def load_gray(path, max_side=MAX_SIDE):
    img = _imread_any(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return _norm_size(img, max_side)


# ─────────────────────────────────────────────────────────────────────────────
# مرحله ۱+۲: نرمال‌سازی نور و هم‌راستاسازی
# ─────────────────────────────────────────────────────────────────────────────

def _light_norm(img):
    """حذف اختلاف نور سراسری و محلی: L_new = 128 + (L − blur(L)).

    از CLAHE استفاده نمی‌شود چون نویز JPEG را تقویت می‌کند. میانگیر‌گیری
    محلی (LOCAL_GRID) نویز JPEG را ≤۵ واحد و تغییرات واقعی را ≥۱۷ واحد
    نگه می‌دارد. a,b به مرکز صفر منتقل می‌شوند تا کروما مستقل از آفست باشد.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    diff = L - cv2.blur(L, (LOCAL_GRID, LOCAL_GRID))
    lab = lab.astype(np.float32)
    lab[:, :, 0] = 128.0 + diff
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def _align(im1, im2, seed=42):
    """هم‌راستاسازی im2 روی im1 با SIFT + similarity (scale/rot/trans) + RANSAC.

    برمی‌گرداند: (im2_aligned, valid_mask, A)
    seed ثابت → نتایجات قابل بازتولید.
    """
    cv2.setRNGSeed(seed)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise ValueError("نقاط مشترک کافی برای هم‌راستاسازی پیدا نشد")
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < MATCH_RATIO * n.distance]
    if len(good) < 8:
        raise ValueError("تطبیق کافی برای هم‌راستاسازی پیدا نشد")
    P1 = np.float32([kp1[m.queryIdx].pt for m in good])
    P2 = np.float32([kp2[m.trainIdx].pt for m in good])
    A, inl = cv2.estimateAffinePartial2D(P1, P2, method=cv2.RANSAC,
                                         ransacReprojThreshold=RANSAC_THRESH,
                                         maxIters=8000, confidence=0.999)
    if A is None:
        raise ValueError("هم‌راستاسازی ناموفق بود")
    inl_frac = float(inl.sum()) / len(good) if inl is not None else 0.0
    if inl_frac < 0.15:
        raise ValueError(f"هم‌راستاسازی غیرقابل اتکا بود ({inl_frac:.0%} درون‌خط)")
    h, w = im1.shape[:2]
    im2_al = cv2.warpAffine(im2, A, (w, h), flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # ماسک معتبر از هندسه‌ی warp (نه روشنایی): یک ماسک یک‌دست را با همان
    # تبدیل warp می‌کنیم — هر جا خروجی غیرصفر باشد، نگاشت داخل تصویر مبدأ بوده.
    ones = np.ones((im2.shape[0], im2.shape[1]), np.uint8)
    cov = cv2.warpAffine(ones, A, (w, h), flags=cv2.WARP_INVERSE_MAP + cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cov = cv2.erode(cov, np.ones((COVERAGE_ERODE, COVERAGE_ERODE), np.uint8))
    return im2_al, cov.astype(bool), A


# ─────────────────────────────────────────────────────────────────────────────
# مرحله ۳+۴: استخراج ماسک تغییر و نقاط برتر
# ─────────────────────────────────────────────────────────────────────────────

def _change_masks(im1, im2_al, valid):
    """برمی‌گرداند: (growth, shrink, Ld, la, lb)"""
    la = _light_norm(im1)
    lb = _light_norm(im2_al)

    chroma = np.sqrt((la[:, :, 1] - lb[:, :, 1]) ** 2 + (la[:, :, 2] - lb[:, :, 2]) ** 2)
    chroma = cv2.GaussianBlur(np.where(valid, chroma, 0), (BLUR, BLUR), 0)

    Ld = lb[:, :, 0] - la[:, :, 0]
    Ld = cv2.GaussianBlur(Ld, (BLUR, BLUR), 0)

    # مسیر روشنایی فقط روی مناطق روشن (اجسام سفید) — فرش تیره از این مسیر
    # وارد نمی‌شود چون خطای هم‌راستاسازی روی بافت تکراری Ldiff بزرگ می‌دهد.
    g1g = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2g = cv2.cvtColor(im2_al, cv2.COLOR_BGR2GRAY)
    bright = ((g1g > BRIGHT_MIN) & (g2g > BRIGHT_MIN)) & valid

    changed_abs = (chroma > CHROMA_MIN) | ((np.abs(Ld) > L_ABS_MIN) & bright)
    m = changed_abs.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    n, labels, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            keep[labels == i] = 1

    changed = keep > 0
    growth = changed & (Ld > L_DIR_MIN)
    shrink = changed & (Ld < -L_DIR_MIN)
    return growth, shrink, Ld, la, lb


def _regions_from_mask(mask, rtype, total_px, w, h):
    """اجزای مجزا یک ماسک → لیست دیکشنری."""
    out = []
    mu8 = mask.astype(np.uint8)
    n, labels, st, _ = cv2.connectedComponentsWithStats(mu8, 8)
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA:
            continue
        mi = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        cx = M["m10"] / M["m00"] if M["m00"] else st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] / 2
        cy = M["m01"] / M["m00"] if M["m00"] else st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] / 2
        # جهت: زاویه مرکز نسبت به مرکز تصویر
        dx, dy = cx - w / 2, h / 2 - cy  # y معکوس چون بالا=منفی
        ang = math.degrees(math.atan2(dy, dx)) % 360
        dir_idx = int(((ang + 22.5) % 360) / 45)
        direction = FA_NAMES[BIN_NAMES[dir_idx]]
        out.append({
            "area_px": area,
            "cx": round(cx, 0), "cy": round(cy, 0),
            "type": rtype, "sign": "+" if rtype == "growth" else "-",
            "direction": direction,
            "bbox": [int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP]),
                     int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])],
            "pct": round(area / total_px * 100, 1) if total_px else 0.0,
        })
    return out


def extract_regions(baseline_path, followup_path, min_area=MIN_AREA, inset=0.08):
    """استخراج مناطق تغییر به صورت اجزای مجزا برای گزارش تفکیکی.

    مختصات در فضای عکس مرجع نرمال‌شده (max_side) است.

    برمی‌گرداند: (regions, stats)
    regions: لیست دیکشنری با شماره، مساحت، مرکز، جهت، درصد
    stats: مساحت کل رشد/کاهش
    """
    global MIN_AREA
    MIN_AREA = max(min_area, 1)
    im1 = _imread_any(baseline_path, cv2.IMREAD_COLOR)
    im2 = _imread_any(followup_path, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise ValueError("cannot read images")
    im1 = _norm_size(im1)
    im2 = _norm_size(im2)
    h, w = im1.shape[:2]

    im2_al, valid, _ = _align(im1, im2)
    growth, shrink, Ld, la, lb = _change_masks(im1, im2_al, valid)

    tot_g = int(growth.sum())
    tot_s = int(shrink.sum())
    regions = _regions_from_mask(growth, "growth", tot_g, w, h) + \
        _regions_from_mask(shrink, "shrink", tot_s, w, h)
    regions.sort(key=lambda r: r["area_px"], reverse=True)
    regions = regions[:TOP_N]
    for i, r in enumerate(regions, 1):
        r["number"] = i

    stats = {
        "total_growth_px": tot_g,
        "total_shrink_px": tot_s,
        "total_px": tot_g + tot_s,
        "note_fa": ("تحلیل بر اساس نقاط مشترک تصاویر و حذف اثر نور و لرزش "
                    "انجام شده است. نقاط قرمز = رشد، آبی = کاهش. اگر تغییری "
                    "بالای آستانه نباشد هیچ نقطه‌ای نشان داده نمی‌شود."),
    }
    if not regions:
        stats["empty"] = True
        stats["note_fa"] = ("هیچ تغییر قابل‌توجهی بالای آستانه پیدا نشد. "
                            "دو عکس از نظر تغییرات جسم یکسان هستند.")
    return regions, stats


# ─────────────────────────────────────────────────────────────────────────────
# Overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(base_bgr, growth, shrink, numbered=True):
    """رسم نقاط تغییر روی عکس مرجع: قرمز = رشد، آبی = کاهش."""
    vis = base_bgr.copy()
    if growth.any():
        vis[growth] = (vis[growth].astype(np.float32) * 0.3 +
                       np.array([0, 0, 255], np.float32) * 0.7).astype(np.uint8)
    if shrink.any():
        vis[shrink] = (vis[shrink].astype(np.float32) * 0.3 +
                       np.array([255, 0, 0], np.float32) * 0.7).astype(np.uint8)
    allc = (growth | shrink).astype(np.uint8)
    if allc.any():
        cnts, _ = cv2.findContours(allc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
        if numbered:
            n = 1
            for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:TOP_N]:
                M = cv2.moments(c)
                if M["m00"] <= 0:
                    continue
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.putText(vis, str(n), (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
                n += 1
    return vis


def make_overlay(baseline_path, followup_path):
    """ساخت تصویر تطبیقی شماره‌دار."""
    im1 = _imread_any(baseline_path, cv2.IMREAD_COLOR)
    im2 = _imread_any(followup_path, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise ValueError("cannot read images")
    im1 = _norm_size(im1)
    im2 = _norm_size(im2)
    im2_al, valid, _ = _align(im1, im2)
    growth, shrink, _, _, _ = _change_masks(im1, im2_al, valid)
    return draw_overlay(im1, growth, shrink)


def make_overlay_from_arrays(im1, im2):
    """نگه‌داشتن امضای قدیمی برای main.py — برمی‌گرداند (vis, growth, shrink, g, s)."""
    im2_al, valid, _ = _align(im1, im2)
    growth, shrink, _, _, _ = _change_masks(im1, im2_al, valid)
    vis = draw_overlay(im1, growth, shrink)
    return vis, growth, shrink, int(growth.sum()), int(shrink.sum())


# ─────────────────────────────────────────────────────────────────────────────
# گزارش
# ─────────────────────────────────────────────────────────────────────────────

def _area_change(g_before, g_after):
    """مساحت روشن (پروکسی سوژه) در یک bbox."""
    b, a = float((g_before > 150).sum()), float((g_after > 150).sum())
    if b <= _AREA_ZERO and a <= _AREA_ZERO:
        return b, a, None, "empty", "—"
    if b <= _AREA_ZERO:
        return b, a, None, "emerged", "ظهور (نقطه جدید)"
    if a <= _AREA_ZERO:
        return b, a, None, "removed", "حذف کامل"
    if b < _AREA_TINY:
        return b, a, None, "tiny_base", "— (مبنای کوچک)"
    pctv = round((a - b) / b * 100, 2)
    return b, a, pctv, "pct", None


def _analyze_scoped(baseline_path, followup_path, px_per_mm, inset, selected):
    """گزارش محدود به نقاط انتخاب‌شده."""
    regions, _ = extract_regions(baseline_path, followup_path, min_area=MIN_AREA, inset=inset)
    if not regions:
        raise ValueError("هیچ نقطه تغییر بالای آستانه پیدا نشد — نقطه‌ای برای تحلیل نیست")

    im1 = _imread_any(baseline_path, cv2.IMREAD_COLOR)
    im2 = _imread_any(followup_path, cv2.IMREAD_COLOR)
    im1 = _norm_size(im1)
    im2 = _norm_size(im2)
    im2_al, valid, _ = _align(im1, im2)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2_al, cv2.COLOR_BGR2GRAY)

    sel_nums = []
    for s in selected:
        try:
            v = int(str(s).strip())
        except ValueError:
            continue
        if 1 <= v <= len(regions):
            sel_nums.append(v)
    if not sel_nums:
        raise ValueError("هیچ‌یک از نقاط انتخاب‌شده معتبر نیست")

    out = []
    for r in regions:
        if r["number"] not in sel_nums:
            continue
        bx, by, bw, bh = [int(v) for v in r["bbox"]]
        x0, x1 = max(0, bx), min(g1.shape[1], bx + bw)
        y0, y1 = max(0, by), min(g1.shape[0], by + bh)
        b, a, cpct, kind, label = _area_change(g1[y0:y1, x0:x1], g2[y0:y1, x0:x1])
        entry = {
            "number": r["number"], "type": r["type"], "sign": r["sign"],
            "direction": r.get("direction", ""), "pct": r["pct"],
            "cx": r["cx"], "cy": r["cy"], "bbox": r["bbox"],
            "area_px": {"before": round(b, 1), "after": round(a, 1),
                        "change_pct": cpct, "change_kind": kind,
                        "change_label": label},
        }
        if px_per_mm and px_per_mm > 0:
            entry["area_mm2"] = {"before": round(b / px_per_mm ** 2, 2),
                                 "after": round(a / px_per_mm ** 2, 2)}
        out.append(entry)

    scoped = [e for e in out if e["area_px"].get("change_kind") != "empty"]
    tot_b = sum(e["area_px"]["before"] for e in scoped)
    tot_a = sum(e["area_px"]["after"] for e in scoped)
    tot_cpct, tot_kind, tot_label = None, "empty", "—"
    if tot_b > _AREA_ZERO:
        tot_cpct = round((tot_a - tot_b) / tot_b * 100, 2)
        tot_kind, tot_label = "pct", None

    growth, shrink, Ld, _, _ = _change_masks(im1, im2_al, valid)
    overlay = draw_overlay(im1, growth, shrink)

    res = {
        "scoped": True,
        "selected": [e["number"] for e in out],
        "regions": out,
        "area_px": {"before": round(tot_b, 1), "after": round(tot_a, 1),
                    "change_pct": tot_cpct, "change_kind": tot_kind,
                    "change_label": tot_label},
        # برای حفظ سازگاری با frontend/pdfgen (همیشه صفر — حجم حذف شده)
        "volume_proxy_px3": {"before": 0.0, "after": 0.0, "change_pct": 0.0},
        "bbox_px": {"before_wh": [0, 0], "after_wh": [0, 0],
                    "width_change_pct": 0.0, "height_change_pct": 0.0},
        "directions": [],
        "note_fa": ("تحلیل محدود به نقاط انتخاب‌شده است. مساحت هر نقطه از "
                    "پیکسل‌های روشن (سوژه) داخل همان ناحیه سنجیده شده است. "
                    "ستون «نوع» از مقایسه مساحت مرجع و دوم به‌دست می‌آید. "
                    "حجم تخمینی حذف شده است چون بدون سگمنت‌گذاری کامل جسم "
                    "معنای فیزیکی نداشت."),
    }
    if px_per_mm and px_per_mm > 0:
        res["scale"] = {"px_per_mm": px_per_mm,
                        "area_mm2": {"before": round(tot_b / px_per_mm ** 2, 2),
                                     "after": round(tot_a / px_per_mm ** 2, 2)}}
    return res, overlay, (growth, shrink)


def analyze(baseline_path, followup_path, px_per_mm=None, inset=0.08, selected=None):
    """تحلیل کامل یا محدود به نقاط انتخابی (selected = شماره‌های extract_regions)."""
    if selected is not None:
        return _analyze_scoped(baseline_path, followup_path, px_per_mm, inset, selected)

    im1 = _imread_any(baseline_path, cv2.IMREAD_COLOR)
    im2 = _imread_any(followup_path, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise ValueError("cannot read images")
    im1 = _norm_size(im1)
    im2 = _norm_size(im2)

    im2_al, valid, _ = _align(im1, im2)
    growth, shrink, Ld, _, _ = _change_masks(im1, im2_al, valid)

    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2_al, cv2.COLOR_BGR2GRAY)
    tot_b = float((g1 > 150).sum())
    tot_a = float((g2 > 150).sum())

    def pct(a, b):
        return (b - a) / a * 100 if a > 1e-9 else 0.0

    overlay = draw_overlay(im1, growth, shrink)
    res = {
        "scoped": False,
        "area_px": {"before": round(tot_b, 1), "after": round(tot_a, 1),
                    "change_pct": round(pct(tot_b, tot_a), 2)},
        "bbox_px": {"before_wh": [0, 0], "after_wh": [0, 0],
                    "width_change_pct": 0.0, "height_change_pct": 0.0},
        "directions": [],
        "volume_proxy_px3": {"before": 0.0, "after": 0.0, "change_pct": 0.0},
        "note_fa": ("تحلیل بر اساس نقاط مشترک تصاویر و حذف اثر نور و لرزش "
                    "انجام شده است. نقاط قرمز = رشد، آبی = کاهش. "
                    "حجم تخمینی حذف شده است چون بدون سگمنت‌گذاری کامل جسم "
                    "معنای فیزیکی نداشت."),
    }
    if px_per_mm and px_per_mm > 0:
        res["scale"] = {"px_per_mm": px_per_mm,
                        "area_mm2": {"before": round(tot_b / px_per_mm ** 2, 2),
                                     "after": round(tot_a / px_per_mm ** 2, 2)}}
    return res, overlay, (growth, shrink)
