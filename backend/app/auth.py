"""احراز هویت و دسترسی‌ها: نقش‌ها + ماتریس دسترسی قابل تنظیم توسط ادمین."""
import json
import os
import secrets
import hashlib
import time

AUTH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "auth")
os.makedirs(AUTH_DIR, exist_ok=True)
USERS_PATH = os.path.join(AUTH_DIR, "users.json")

# نشست‌های فعال: token -> {username, is_admin, perms, exp}
SESSIONS = {}
SESSION_TTL = 12 * 3600  # ۱۲ ساعت

# کلیدهای دسترسی (perms). ادمین همه را دارد.
PERM_KEYS = [
    ("camera", "تصویربرداری مستقیم"),
    ("gallery", "مشاهده گالری پروژه"),
    ("analyze", "اجرای تحلیل"),
    ("report", "گزارش PDF"),
    ("delete_photo", "حذف عکس"),
    ("comment", "ثبت کامنت"),
    ("approve_photo", "تأیید عکس"),
    ("manage_projects", "ایجاد/ویرایش/حذف پروژه"),
    ("manage_users", "مدیریت کاربران"),
]
ALL_PERMS = [k for k, _ in PERM_KEYS]


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _load_users() -> dict:
    if os.path.exists(USERS_PATH):
        try:
            with open(USERS_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_users(users: dict) -> None:
    tmp = USERS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    os.replace(tmp, USERS_PATH)


def _default_perms(is_admin: bool) -> dict:
    return {k: True for k in ALL_PERMS} if is_admin else {
        "camera": True, "gallery": True, "analyze": True, "comment": True,
    }


def _public_view(u: dict) -> dict:
    """نمای عمومی کاربر برای ارسال به کلاینت."""
    return {
        "username": u.get("username"),
        "is_admin": bool(u.get("is_admin", False)),
        "perms": u.get("perms", {}),
    }


# --- راه‌اندازی اولیه ---
# نام کاربری ادمین اولیه (در صورت نبود هیچ کاربری، از این متغیر استفاده می‌شود)
DEFAULT_ADMIN_USERNAME = os.environ.get("FSFCARE_ADMIN", "admin")
# رمز عبور ادمین اولیه — حتماً از متغیر محیطی تنظیم شود.
# اگر تنظیم نشد، یک رمز تصادفی امن تولید و فقط یک‌بار در لاگ سرور چاپ می‌شود.
DEFAULT_ADMIN_PASSWORD = os.environ.get("FSFCARE_ADMIN_PASSWORD", secrets.token_urlsafe(24))


def ensure_default_admin():
    """ساخت ادمین اولیه در صورت نبود هیچ کاربری.

    رمز پیش‌فرض از متغیر محیطی FSFCARE_ADMIN_PASSWORD خوانده می‌شود.
    اگر تنظیم نشده باشد، یک رمز تصادفی امن تولید شده و در لاگ سرور چاپ
    می‌گردد تا ادمین بتواند اولین ورود را انجام دهد و سپس رمز را تغییر دهد.
    """
    users = _load_users()
    if not users:
        users[DEFAULT_ADMIN_USERNAME] = {
            "password_hash": _hash(DEFAULT_ADMIN_PASSWORD),
            "is_admin": True,
            "perms": _default_perms(True),
            "created_at": time.time(),
        }
        _save_users(users)
        print("========================================")
        print("رمز ادمین اولیه ساخته شد: " + DEFAULT_ADMIN_PASSWORD)
        print("همین حالا وارد شوید و آن را تغییر دهید!")
        print("========================================", flush=True)
    return users


def _migrate(users: dict) -> bool:
    """تضمین وجود فیلد perms برای یوزرهای قدیمی."""
    changed = False
    for u in users.values():
        if "perms" not in u or not isinstance(u.get("perms"), dict):
            u["perms"] = _default_perms(bool(u.get("is_admin", False)))
            changed = True
        else:
            for k in ALL_PERMS:
                u["perms"].setdefault(k, False)
            if u.get("is_admin"):
                u["perms"] = {k: True for k in ALL_PERMS}
                changed = True
    return changed


def authenticate(username: str, password: str):
    """بررسی یوزر/پسورد؛ در صورت صحت توکن نشست برمی‌گرداند."""
    users = ensure_default_admin()
    if _migrate(users):
        _save_users(users)
    u = users.get(username)
    if not u or u.get("password_hash") != _hash(password):
        return None
    token = secrets.token_hex(24)
    SESSIONS[token] = {
        "username": username,
        "is_admin": bool(u.get("is_admin", False)),
        "perms": u.get("perms", _default_perms(bool(u.get("is_admin", False)))),
        "exp": time.time() + SESSION_TTL,
    }
    return token


def session_user(token: str):
    if not token:
        return None
    s = SESSIONS.get(token)
    if not s:
        return None
    if time.time() > s["exp"]:
        SESSIONS.pop(token, None)
        return None
    return s


def logout(token: str):
    SESSIONS.pop(token, None)


def list_users() -> list:
    users = ensure_default_admin()
    if _migrate(users):
        _save_users(users)
    return [_public_view(v) | {"username": k} for k, v in sorted(users.items())]


def add_user(username: str, password: str, is_admin: bool = False, perms: dict = None) -> bool:
    users = ensure_default_admin()
    username = (username or "").strip()
    if not username or not password:
        return False
    if username in users:
        return False
    p = _default_perms(bool(is_admin))
    if perms:
        for k in ALL_PERMS:
            if k in perms:
                p[k] = bool(perms[k])
    users[username] = {
        "password_hash": _hash(password),
        "is_admin": bool(is_admin),
        "perms": p,
        "created_at": time.time(),
    }
    _save_users(users)
    return True


def delete_user(username: str) -> bool:
    users = ensure_default_admin()
    if username == "admin":
        return False  # ادمین اصلی قابل حذف نیست
    if username not in users:
        return False
    del users[username]
    _save_users(users)
    return True


def set_user_perms(username: str, perms: dict) -> bool:
    """تنظیم دسترسی‌های یک کاربر (فقط ادمین)."""
    users = ensure_default_admin()
    u = users.get(username)
    if not u:
        return False
    p = u.get("perms", {})
    for k in ALL_PERMS:
        if k in perms:
            p[k] = bool(perms[k])
    u["perms"] = p
    if u.get("is_admin"):
        u["perms"] = {k: True for k in ALL_PERMS}
    _save_users(users)
    # به‌روزرسانی نشست‌های فعال همان کاربر
    for s in SESSIONS.values():
        if s.get("username") == username:
            s["perms"] = u["perms"]
    return True


def toggle_user_admin(username: str, is_admin: bool) -> bool:
    users = ensure_default_admin()
    u = users.get(username)
    if not u or username == "admin":
        return False
    u["is_admin"] = bool(is_admin)
    u["perms"] = _default_perms(bool(is_admin))
    _save_users(users)
    for s in SESSIONS.values():
        if s.get("username") == username:
            s["is_admin"] = u["is_admin"]
            s["perms"] = u["perms"]
    return True


def change_password(username: str, old_password: str, new_password: str) -> bool:
    users = ensure_default_admin()
    u = users.get(username)
    if not u or u.get("password_hash") != _hash(old_password):
        return False
    u["password_hash"] = _hash(new_password)
    _save_users(users)
    return True


# مقداردهی اولیه
ensure_default_admin()
