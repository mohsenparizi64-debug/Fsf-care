"""fsfmonitor — بک‌اند FastAPI. اجرا: python -m uvicorn app.main:app --host 0.0.0.0 --port 8095 (از داخل backend/)"""
import os
import uuid
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.pipeline import analyze
from app import projects as prj
from app import auth
from app import pdfgen

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "frontend")
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="fsfmonitor")


# ---------- HTTP -> HTTPS ----------
# Cloudflare TLS را خاتمه می‌دهد و همیشه با HTTP به مبدا می‌رسد،
# پس به جای scheme واقعی، هدر X-Forwarded-Proto بررسی می‌شود.
# /api/health مستثنی است تا watchdog محلی (HTTP) کار خود را ادامه دهد.
@app.middleware("http")
async def force_https(request: Request, call_next):
    if request.url.path == "/api/health":
        return await call_next(request)
    proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    if proto == "http":
        return JSONResponse(
            {"detail": "HTTPS الزامی است"},
            status_code=301,
            headers={"Location": request.url.replace(scheme="https").__str__()},
        )
    return await call_next(request)


# ---------- کمکی‌های احراز هویت ----------

def _bearer(request: Request) -> str:
    h = request.headers.get("authorization") or ""
    if h.lower().startswith("bearer "):
        return h.split(" ", 1)[1].strip()
    return ""


def require_user(request: Request):
    """نیاز به نشست معتبر؛ دیکشنری کاربر را برمی‌گرداند."""
    user = auth.session_user(_bearer(request))
    if not user:
        raise HTTPException(401, "نشست نامعتبر است؛ دوباره وارد شوید")
    return user


def require_admin(request: Request):
    user = require_user(request)
    if not user.get("is_admin"):
        raise HTTPException(403, "این عملیات فقط برای ادمین مجاز است")
    return user


def require_perm(request: Request, key: str):
    """نیاز به دسترسی خاص. ادمین همیشه دارد."""
    user = require_user(request)
    if user.get("is_admin"):
        return user
    perms = user.get("perms", {})
    if not perms.get(key, False):
        raise HTTPException(403, "شما دسترسی این عملیات را ندارید")
    return user


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


JDTIME = prj._HAS_JDATE


@app.get("/api/health")
def health():
    return {"ok": True, "service": "fsfmonitor", "version": "0.8.0", "jdatetime": JDTIME}


# ---------- احراز هویت ----------

@app.post("/api/login")
async def api_login(request: Request):
    body = await request.json()
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))
    remember = bool(body.get("remember", False))
    token = auth.authenticate(username, password)
    if not token:
        raise HTTPException(401, "نام کاربری یا رمز عبور اشتباه است")
    return {"ok": True, "token": token,
            "remember": remember,
            "is_admin": auth.SESSIONS[token]["is_admin"],
            "perms": auth.SESSIONS[token].get("perms", {}),
            "username": username}


@app.post("/api/logout")
def api_logout(request: Request):
    auth.logout(_bearer(request))
    return {"ok": True}


@app.get("/api/me")
def api_me(request: Request):
    user = auth.session_user(_bearer(request))
    if not user:
        return {"ok": False}
    return {"ok": True, "username": user["username"], "is_admin": user["is_admin"],
            "perms": user.get("perms", {})}


@app.get("/api/perms")
def api_list_perms(request: Request):
    """لیست کلیدهای دسترسی — برای ساخت جدول پنل ادمین."""
    require_user(request)
    return {"ok": True, "perms": [{"key": k, "label": label} for k, label in auth.PERM_KEYS]}


# ---------- مدیریت کاربران (ادمین) ----------

@app.get("/api/users")
def api_list_users(request: Request):
    require_perm(request, "manage_users")
    return {"ok": True, "users": auth.list_users()}


@app.post("/api/users")
async def api_add_user(request: Request):
    require_perm(request, "manage_users")
    body = await request.json()
    ok = auth.add_user(str(body.get("username", "")), str(body.get("password", "")),
                       bool(body.get("is_admin", False)),
                       body.get("perms") if isinstance(body.get("perms"), dict) else None)
    if not ok:
        raise HTTPException(400, "نام کاربری تکراری یا نامعتبر است")
    return {"ok": True}


@app.put("/api/users/{username}/perms")
async def api_set_perms(request: Request, username: str):
    require_perm(request, "manage_users")
    body = await request.json()
    ok = auth.set_user_perms(username, body if isinstance(body, dict) else {})
    if not ok:
        raise HTTPException(400, "تنظیم دسترسی ممکن نشد")
    return {"ok": True}


@app.post("/api/users/{username}/admin")
async def api_toggle_admin(request: Request, username: str):
    require_admin(request)
    body = await request.json()
    ok = auth.toggle_user_admin(username, bool(body.get("is_admin", False)))
    if not ok:
        raise HTTPException(400, "تغییر نقش ممکن نشد")
    return {"ok": True}


@app.delete("/api/users/{username}")
def api_delete_user(request: Request, username: str):
    require_perm(request, "manage_users")
    ok = auth.delete_user(username)
    if not ok:
        raise HTTPException(400, "حذف ممکن نشد (یوزر وجود ندارد یا ادمین اصلی است)")
    return {"ok": True}


@app.post("/api/password")
async def api_change_password(request: Request):
    user = require_user(request)
    body = await request.json()
    ok = auth.change_password(user["username"],
                              str(body.get("old_password", "")),
                              str(body.get("new_password", "")))
    if not ok:
        raise HTTPException(400, "رمز قدیمی اشتباه است")
    return {"ok": True}


# ---------- پروژه‌ها ----------

@app.get("/api/projects")
def api_list_projects(request: Request):
    require_user(request)
    return {"ok": True, "projects": prj.list_projects()}


@app.post("/api/projects")
async def api_create_project(request: Request):
    require_perm(request, "manage_projects")
    body = await request.json()
    name = str(body.get("name", "")).strip()
    if not name:
        raise HTTPException(400, "نام پروژه خالی است")
    meta = prj.ensure_project(name)
    return {"ok": True, "project": meta.get("project"), "created_at": meta.get("created_at")}


@app.put("/api/projects/{project}")
async def api_rename_project(request: Request, project: str):
    require_perm(request, "manage_projects")
    body = await request.json()
    new = str(body.get("name", "")).strip()
    if not new:
        raise HTTPException(400, "نام جدید خالی است")
    try:
        meta = prj.rename_project(project, new)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "project": meta.get("project")}


@app.delete("/api/projects/{project}")
def api_delete_project(request: Request, project: str):
    require_perm(request, "manage_projects")
    ok = prj.delete_project(project)
    return {"ok": ok}


@app.get("/api/projects/{project}/photos")
def api_get_photos(request: Request, project: str):
    require_perm(request, "gallery")
    return {"ok": True, "photos": prj.get_photos(prj._safe_name(project))}


@app.get("/api/projects/{project}/photos/approved")
def api_get_approved_photos(request: Request, project: str):
    """عکس‌های تأییدشده — برای انتخاب مرجع/شبح."""
    require_user(request)
    return {"ok": True, "photos": prj.approved_photos(prj._safe_name(project))}


@app.post("/api/projects/{project}/photos")
async def api_upload_photo(request: Request, project: str, file: UploadFile = File(...)):
    """ثبت یک عکس در خط زمانی پروژه با تاریخ و ساعت شمسی."""
    user = require_user(request)
    safe = prj._safe_name(project)
    if not (project or "").strip():
        raise HTTPException(400, "نام پروژه خالی است")
    prj.ensure_project(safe)
    stamp = prj.shamsi_now()
    fname = f"{stamp['date_fa'].replace('/', '').translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789'))}" \
            f"_{stamp['time_fa'].replace(':', '').translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789'))}.jpg"
    base, n = fname, 1
    target = os.path.join(prj._proj_dir(safe), base)
    while os.path.exists(target):
        n += 1
        target = os.path.join(prj._proj_dir(safe), f"{str(base)[:-4]}_{n}.jpg")
    data = await file.read()
    if not data:
        raise HTTPException(400, "فایل خالی است")
    with open(target, "wb") as f:
        f.write(data)
    entry = prj.add_photo(safe, os.path.basename(target), source="upload",
                          author=user.get("username", ""))
    return {
        "ok": True,
        "photo": entry,
        "url": f"/projects/{safe}/{os.path.basename(target)}",
    }


@app.delete("/api/projects/{project}/photos/{filename}")
def api_delete_photo(request: Request, project: str, filename: str):
    """حذف یک عکس از پروژه — نیاز به دسترسی delete_photo."""
    require_perm(request, "delete_photo")
    safe = prj._safe_name(project)
    d = prj._proj_dir(safe)
    path = os.path.join(d, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "عکس پیدا نشد")
    os.remove(path)
    meta = prj._load_meta(safe)
    meta["photos"] = [p for p in meta.get("photos", []) if p.get("file") != filename]
    prj._save_meta(safe, meta)
    return {"ok": True}


@app.post("/api/projects/{project}/photos/{filename}/approve")
async def api_approve_photo(request: Request, project: str, filename: str):
    """تأیید عکس توسط ادمین — فقط عکس‌های تأییدشده مرجع می‌شوند."""
    require_perm(request, "approve_photo")
    safe = prj._safe_name(project)
    body = None
    try:
        import json as _json
        raw = await request.body()
        body = _json.loads(raw) if raw else {}
    except Exception:
        body = {}
    approved = bool(body.get("approved", True))
    try:
        p = prj.set_approved(safe, filename, approved)
    except KeyError:
        raise HTTPException(404, "عکس پیدا نشد")
    return {"ok": True, "photo": p}


@app.get("/api/projects/{project}/photos/{filename}/raw")
def api_photo_raw(request: Request, project: str, filename: str):
    """عکس اصلی بدون گرید."""
    require_user(request)
    safe = prj._safe_name(project)
    path = os.path.join(prj.DATA_DIR, safe, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "عکس پیدا نشد")
    from fastapi.responses import FileResponse
    return FileResponse(path, media_type="image/jpeg")


# ---------- کامنت‌ها ----------

@app.get("/api/projects/{project}/photos/{filename}/comments")
def api_get_comments(request: Request, project: str, filename: str):
    require_user(request)
    safe = prj._safe_name(project)
    p = prj.find_photo(safe, filename)
    if not p:
        raise HTTPException(404, "عکس پیدا نشد")
    return {"ok": True, "comments": p.get("comments", [])}


@app.post("/api/projects/{project}/photos/{filename}/comments")
async def api_add_comment(request: Request, project: str, filename: str):
    require_perm(request, "comment")
    safe = prj._safe_name(project)
    body = await request.json()
    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(400, "متن کامنت خالی است")
    user = auth.session_user(_bearer(request))
    try:
        c = prj.add_comment(safe, filename, user.get("username", ""), text)
    except KeyError:
        raise HTTPException(404, "عکس پیدا نشد")
    return {"ok": True, "comment": c}


@app.delete("/api/projects/{project}/photos/{filename}/comments/{index}")
def api_delete_comment(request: Request, project: str, filename: str, index: int):
    require_perm(request, "comment")
    safe = prj._safe_name(project)
    ok = prj.delete_comment(safe, filename, int(index))
    if not ok:
        raise HTTPException(404, "کامنت پیدا نشد")
    return {"ok": True}


# ---------- تحلیل + گزارش PDF ----------

@app.post("/api/analyze")
async def api_analyze(
    request: Request,
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    px_per_mm: float = Form(0),
    inset: float = Form(0.08),
):
    require_perm(request, "analyze")
    uid = uuid.uuid4().hex[:8]
    p1 = os.path.join(UPLOAD_DIR, f"{uid}_before.jpg")
    p2 = os.path.join(UPLOAD_DIR, f"{uid}_after.jpg")
    with open(p1, "wb") as f:
        f.write(await before.read())
    with open(p2, "wb") as f:
        f.write(await after.read())
    try:
        res, overlay, _ = analyze(p1, p2,
                                  px_per_mm=px_per_mm or None,
                                  inset=float(inset))
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    cv2.imwrite(os.path.join(UPLOAD_DIR, f"{uid}_overlay.jpg"), overlay)
    res["ok"] = True
    res["overlay_url"] = f"/uploads/{uid}_overlay.jpg"
    res["_tmp"] = {"before": p1, "after": p2, "overlay": os.path.join(UPLOAD_DIR, f"{uid}_overlay.jpg")}
    return res


@app.post("/api/report")
async def api_report(request: Request):
    """گزارش تحلیلی PDF بین دو عکس یک پروژه."""
    require_perm(request, "report")
    body = await request.json()
    project = prj._safe_name(str(body.get("project", "")))
    ref_index = body.get("ref_index")
    other_index = body.get("other_index")
    if ref_index is None or other_index is None:
        raise HTTPException(400, "شاخص عکس‌ها لازم است")
    photos = prj.get_photos(project)
    if not (0 <= int(ref_index) < len(photos)) or not (0 <= int(other_index) < len(photos)):
        raise HTTPException(400, "شاخص عکس نامعتبر است")
    pa, pb = photos[int(ref_index)], photos[int(other_index)]
    pa_path = os.path.join(prj.DATA_DIR, project, pa["file"])
    pb_path = os.path.join(prj.DATA_DIR, project, pb["file"])

    uid = uuid.uuid4().hex[:8]
    try:
        res, overlay, _ = analyze(pa_path, pb_path, px_per_mm=None, inset=0.08)
    except Exception as e:
        raise HTTPException(400, f"خطا در تحلیل تصاویر: {e}")
    ov_path = os.path.join(UPLOAD_DIR, f"{uid}_overlay.jpg")
    cv2.imwrite(ov_path, overlay)

    try:
        pdf_bytes = pdfgen.build_report(
            project=project,
            photo_a=pa, photo_b=pb,
            analysis=res,
            img_a_path=pa_path, img_b_path=pb_path,
            overlay_path=ov_path,
        )
    except Exception as e:
        raise HTTPException(500, f"خطا در ساخت PDF: {e}")
    fname = f"report_{pa['file'][:-4]}_{pb['file'][:-4]}.pdf"
    from urllib.parse import quote as _quote
    fname_ascii = fname.encode("ascii", "ignore").decode("ascii").replace(" ", "_")
    if not fname_ascii or fname_ascii == ".pdf":
        fname_ascii = "report.pdf"
    disp = f'attachment; filename="{fname_ascii}"'
    try:
        disp += f"; filename*=UTF-8''{_quote(fname)}"
    except Exception:
        pass
    headers = {"Content-Disposition": disp}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
if os.path.isdir(prj.DATA_DIR):
    app.mount("/projects", StaticFiles(directory=prj.DATA_DIR), name="projects")
