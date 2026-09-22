"""مدیریت پروژه‌ها، عکس‌ها، کامنت‌ها و تأییدها."""
import json
import os
import time
from urllib.parse import quote

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects")
os.makedirs(DATA_DIR, exist_ok=True)

try:
    import jdatetime
    _HAS_JDATE = True
except Exception:
    _HAS_JDATE = False

_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def _safe_name(name: str) -> str:
    """نام پروژه را به نام پوشه امن تبدیل می‌کند (یونیکد و فاصله نگه می‌دارد)."""
    return "".join(c for c in (name or "").strip() if c not in '/\\:*?"<>|') or "پروژه"


def _proj_dir(safe: str) -> str:
    return os.path.join(DATA_DIR, safe)


def _meta_path(safe: str) -> str:
    return os.path.join(_proj_dir(safe), "meta.json")


def _load_meta(safe: str) -> dict:
    p = _meta_path(safe)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"project": safe, "created_at": time.time(), "photos": []}


def _save_meta(safe: str, meta: dict) -> None:
    d = _proj_dir(safe)
    os.makedirs(d, exist_ok=True)
    tmp = _meta_path(safe) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _meta_path(safe))


def ensure_project(name: str) -> dict:
    safe = _safe_name(name)
    d = _proj_dir(safe)
    os.makedirs(d, exist_ok=True)
    meta = _load_meta(safe)
    meta.setdefault("project", safe)
    meta.setdefault("created_at", time.time())
    meta.setdefault("photos", [])
    _save_meta(safe, meta)
    return meta


def list_projects() -> list:
    out = []
    if not os.path.isdir(DATA_DIR):
        return out
    for name in sorted(os.listdir(DATA_DIR)):
        p = os.path.join(DATA_DIR, name)
        if not os.path.isdir(p):
            continue
        meta = _load_meta(name)
        out.append({
            "name": meta.get("project", name),
            "photo_count": len(meta.get("photos", [])),
            "created_at": meta.get("created_at"),
        })
    return out


def delete_project(name: str) -> bool:
    import shutil
    safe = _safe_name(name)
    d = _proj_dir(safe)
    if not os.path.isdir(d):
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


def rename_project(old: str, new: str) -> dict:
    """تغییر نام پروژه (پوشه + متادیتا). نام جدید یکتا می‌شود."""
    old_safe = _safe_name(old)
    new_safe = _safe_name(new)
    if not new_safe:
        raise ValueError("نام جدید خالی است")
    d_old = _proj_dir(old_safe)
    if not os.path.isdir(d_old):
        raise ValueError("پروژه قدیمی وجود ندارد")
    if old_safe == new_safe:
        return _load_meta(old_safe)
    d_new = _proj_dir(new_safe)
    base, n = new_safe, 1
    while os.path.exists(d_new):
        n += 1
        d_new = _proj_dir(f"{new_safe} ({n})")
    os.rename(d_old, d_new)
    meta = _load_meta(os.path.basename(d_new))
    meta["project"] = os.path.basename(d_new)
    _save_meta(os.path.basename(d_new), meta)
    return meta


def shamsi_now() -> dict:
    """برچسب زمانی شمسی (یا میلادی در صورت نبود jdatetime)."""
    if _HAS_JDATE:
        now = jdatetime.datetime.now()
        return {
            "date_fa": now.strftime("%Y/%m/%d").translate(_FA_DIGITS),
            "time_fa": now.strftime("%H:%M").translate(_FA_DIGITS),
            "weekday": _fa_weekday(now),
            "iso": now.isoformat(),
        }
    t = time.localtime()
    return {
        "date_fa": f"{t.tm_year}/{t.tm_mon:02d}/{t.tm_mday:02d}",
        "time_fa": f"{t.tm_hour:02d}:{t.tm_min:02d}",
        "weekday": "",
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def add_photo(safe: str, filename: str, source: str = "upload", author: str = "") -> dict:
    """افزودن رکورد عکس به متادیتا."""
    meta = _load_meta(safe)
    photos = meta.setdefault("photos", [])
    entry = {
        "index": len(photos),
        "file": filename,
        "source": source,
        "author": author,
        "at": shamsi_now(),
        "approved": False,
        "comments": [],
    }
    photos.append(entry)
    _save_meta(safe, meta)
    return entry


_FA_WEEKDAYS = {
    "Saturday": "شنبه", "Sunday": "یکشنبه", "Monday": "دوشنبه", "Tuesday": "سه‌شنبه",
    "Wednesday": "چهارشنبه", "Thursday": "پنجشنبه", "Friday": "جمعه",
}


def _fa_weekday(d) -> str:
    try:
        return _FA_WEEKDAYS.get(d.date().strftime("%A"), "")
    except Exception:
        return ""


def format_stamp(entry: dict) -> dict:
    """برچسب زمانی کامل و نرمال‌شده برای نمایش: تاریخ شمسی، روز هفته، ساعت ۲۴ ساعته."""
    at = entry.get("at", {}) or {}
    date_fa = at.get("date_fa", "")
    time_fa = at.get("time_fa", "")
    weekday = at.get("weekday", "")
    iso = str(at.get("iso") or "")
    # fallback: روز هفته را از iso می‌سازیم (اول شمسی، بعد میلادی)
    if not weekday and iso:
        d = None
        if _HAS_JDATE:
            try:
                d = jdatetime.datetime.fromisoformat(iso)
            except Exception:
                d = None
        if d is None:
            try:
                from datetime import datetime as _dt
                d = _dt.fromisoformat(iso)
                if _HAS_JDATE:
                    d = jdatetime.datetime.fromgregorian(date=d.date(), time=d.time())
            except Exception:
                d = None
        if d is not None:
            weekday = _fa_weekday(d)
            if not date_fa and _HAS_JDATE:
                date_fa = d.strftime("%Y/%m/%d").translate(_FA_DIGITS)
                time_fa = time_fa or d.strftime("%H:%M").translate(_FA_DIGITS)
    return {"date_fa": date_fa, "time_fa": time_fa, "weekday": weekday}


def get_photos(safe: str) -> list:
    meta = _load_meta(safe)
    photos = meta.get("photos", [])
    # فیلتر: فقط فایل‌هایی که روی دیسک هستند
    d = _proj_dir(safe)
    enc = quote(safe, safe="")
    out = []
    for p in photos:
        fname = p.get("file", "")
        if os.path.isfile(os.path.join(d, fname)):
            q = dict(p)
            q["url"] = f"/api/projects/{enc}/photos/{quote(fname, safe='')}/raw"
            q["at"] = format_stamp(p)
            out.append(q)
    return out


def find_photo(safe: str, filename: str) -> dict | None:
    for p in get_photos(safe):
        if p.get("file") == filename:
            return p
    return None


def set_approved(safe: str, filename: str, approved: bool) -> dict:
    """تأیید/عدم تأیید عکس توسط ادمین."""
    meta = _load_meta(safe)
    for p in meta.get("photos", []):
        if p.get("file") == filename:
            p["approved"] = bool(approved)
            _save_meta(safe, meta)
            return p
    raise KeyError("عکس پیدا نشد")


def add_comment(safe: str, filename: str, author: str, text: str) -> dict:
    """افزودن کامنت به یک عکس."""
    meta = _load_meta(safe)
    for p in meta.get("photos", []):
        if p.get("file") == filename:
            c = {"author": author, "text": text, "at": shamsi_now()}
            p.setdefault("comments", []).append(c)
            _save_meta(safe, meta)
            return c
    raise KeyError("عکس پیدا نشد")


def delete_comment(safe: str, filename: str, index: int) -> bool:
    meta = _load_meta(safe)
    for p in meta.get("photos", []):
        if p.get("file") == filename:
            cs = p.get("comments", [])
            if 0 <= index < len(cs):
                del cs[index]
                _save_meta(safe, meta)
                return True
    return False


def approved_photos(safe: str) -> list:
    """فقط عکس‌های تأییدشده (برای انتخاب مرجع)."""
    return [p for p in get_photos(safe) if p.get("approved")]
