"""تولید گزارش تحلیلی PDF با فونت فارسی (Vazirmatn) و چیدمان راست‌به‌چپ."""
import os
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
)

FONT_DIR = "C:/Windows/Fonts"
_VARIANTS = {
    "Vazir": ["Vazirmatn-Regular.ttf", "Vazirmatn-Regular.ttf"],
    "Vazir-Bold": ["Vazirmatn-Bold.ttf", "Vazirmatn-SemiBold.ttf"],
}
_registered = False


def _register_fonts():
    global _registered
    if _registered:
        return
    for name, (primary, fallback) in _VARIANTS.items():
        path = os.path.join(FONT_DIR, primary)
        if not os.path.exists(path):
            path = os.path.join(FONT_DIR, fallback)
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont(name, path))
    _registered = True


def _fa_digits(text):
    if text is None:
        return "—"
    return str(text).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


# کاراکترهای کنترل RTL یونیکد
_RLM = "‮"   # RIGHT-TO-LEFT MARK
_RLE = "‫"   # RIGHT-TO-LEFT EMBEDDING
_PDF = "‬"   # POP DIRECTIONAL FORMATTING

_reshaper = None
def _get_reshaper():
    global _reshaper
    if _reshaper is None:
        try:
            import arabic_reshaper
            _reshaper = arabic_reshaper.reshape
        except Exception:
            _reshaper = lambda x: x  # بدون تغییر
    return _reshaper

def _bidi(text):
    try:
        from bidi.algorithm import get_display
        return get_display(text)
    except Exception:
        return text


def _rtl(text):
    """شکل‌دهی + bidi برای رندر درست فارسی در reportlab (LTR داخلی)."""
    if text is None:
        return ""
    s = str(text)
    reshaped = _get_reshaper()(s)
    return _bidi(reshaped)


def _pct(v):
    try:
        return _fa_digits(f"{float(v):.2f}") + "٪"
    except (TypeError, ValueError):
        return "—"


def build_report(
    project: str,
    photo_a: dict,
    photo_b: dict,
    analysis: dict,
    img_a_path: str,
    img_b_path: str,
    overlay_path: str = None,
) -> bytes:
    """ساخت PDF گزارش. photo_a = مرجع، photo_b = دوم."""
    _register_fonts()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="Buildup Tracker Report",
    )
    st_title = ParagraphStyle("t", fontName="Vazir-Bold", fontSize=17, leading=24,
                              alignment=2, wordWrap="RTL")
    st_h = ParagraphStyle("h", fontName="Vazir-Bold", fontSize=12.5, leading=20,
                          alignment=2, spaceBefore=10, wordWrap="RTL")
    st_b = ParagraphStyle("b", fontName="Vazir", fontSize=10.5, leading=18,
                          alignment=2, wordWrap="RTL")
    st_small = ParagraphStyle("s", fontName="Vazir", fontSize=9, leading=14,
                              alignment=2, wordWrap="RTL")
    story = []
    story.append(Paragraph(_rtl("📷 گزارش تحلیلی Buildup Tracker"), st_title))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(_rtl(f"پروژه: {project}"), st_h))
    a_at = (photo_a or {}).get("at", {}).get("fa", "—")
    b_at = (photo_b or {}).get("at", {}).get("fa", "—")
    story.append(Paragraph(
        _rtl(f"عکس مرجع: {a_at}   |   عکس دوم: {b_at}"), st_b))
    story.append(Spacer(1, 3 * mm))

    def _img(path, max_w=150 * mm, max_h=120 * mm):
        if not path or not os.path.exists(path):
            return None
        w, h = 400, 300
        try:
            from PIL import Image as PILImage
            im = PILImage.open(path)
            w, h = im.size
        except Exception:
            pass
        scale = min(max_w / w, max_h / h, 1.0)
        return RLImage(path, width=w * scale, height=h * scale)

    # عکس‌ها کنار هم
    cells = []
    for p, cap in [(img_a_path, f"مرجع — {a_at}"), (img_b_path, f"دوم — {b_at}")]:
        im = _img(p, max_w=72 * mm, max_h=100 * mm)
        if im is not None:
            cells.append([Paragraph(_rtl(cap), st_small), im])
    if len(cells) >= 2:
        t = Table([[cells[0][1], cells[1][1]],
                   [cells[0][0], cells[1][0]]],
                  colWidths=[82 * mm, 82 * mm])
        t.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 4 * mm))

    # خلاصه
    story.append(Paragraph(_rtl("خلاصه تغییرات"), st_h))
    ap = analysis.get("area_px", {})
    bb = analysis.get("bbox_px", {})
    vp = analysis.get("volume_proxy_px3", {})
    sc = analysis.get("scale") or {}
    rows = [
        [_rtl("شاخص"), _rtl("مرجع"), _rtl("دوم"), _rtl("تغییر")],
        [_rtl("مساحت نیمرخ (px²)"), _fa_digits(ap.get("before")), _fa_digits(ap.get("after")), _pct(ap.get("change_pct"))],
        [_rtl("پهنا (px)"), _fa_numbers_list(bb.get("before_wh")), _fa_numbers_list(bb.get("after_wh")), _pct(bb.get("width_change_pct"))],
        [_rtl("ارتفاع (px)"), "—", "—", _pct(bb.get("height_change_pct"))],
        [_rtl("حجم تخمینی (px³)"), _fa_digits(vp.get("before")), _fa_digits(vp.get("after")), _pct(vp.get("change_pct"))],
    ]
    if sc.get("area_mm2"):
        rows.append([_rtl("مساحت (mm²)"), _fa_digits(sc["area_mm2"].get("before")),
                     _fa_digits(sc["area_mm2"].get("after")), "—"])
    if sc.get("volume_proxy_mm3"):
        rows.append([_rtl("حجم تخمینی (mm³)"), _fa_digits(sc["volume_proxy_mm3"].get("before")),
                     _fa_digits(sc["volume_proxy_mm3"].get("after")), "—"])
    tbl = Table(rows, colWidths=[52 * mm, 38 * mm, 38 * mm, 28 * mm])
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Vazir"),
        ("FONTNAME", (0, 0), (-1, 0), "Vazir-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f0")),
    ]
    tbl.setStyle(TableStyle(style))
    story.append(tbl)
    story.append(Spacer(1, 4 * mm))

    # رشد جهتی
    dirs = analysis.get("directions") or []
    if dirs:
        story.append(Paragraph(_rtl("رشد جهتی"), st_h))
        drows = [[_rtl("جهت"), _rtl("مرجع (px)"), _rtl("دوم (px)"), _rtl("تغییر")]]
        for d in dirs:
            drows.append([_rtl(str(d.get("fa", ""))), _fa_digits(d.get("before_px")),
                          _fa_digits(d.get("after_px")), _pct(d.get("change_pct"))])
        dtbl = Table(drows, colWidths=[42 * mm, 44 * mm, 44 * mm, 28 * mm])
        dtbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Vazir"),
            ("FONTNAME", (0, 0), (-1, 0), "Vazir-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#cccccc")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f0")),
        ]))
        story.append(dtbl)
        story.append(Spacer(1, 4 * mm))

    # تصویر تطبیقی
    if overlay_path:
        im = _img(overlay_path, max_w=140 * mm, max_h=130 * mm)
        if im is not None:
            story.append(Paragraph(_rtl("تصویر تطبیقی"), st_h))
            story.append(im)
            story.append(Paragraph(_rtl("(آبی = مرجع، قرمز = دوم)"), st_small))
    note = analysis.get("note_fa")
    if note:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(_rtl(f"یادداشت: {note}"), st_small))

    doc.build(story)
    return buf.getvalue()


def _fa_numbers_list(v):
    if isinstance(v, list):
        return " × ".join(_fa_digits(x) for x in v)
    return _fa_digits(v)
