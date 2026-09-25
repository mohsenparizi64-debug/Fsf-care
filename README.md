# fsfmonitor

وب‌اپ فارسی موبایل‌فرندلی برای سنجش تغییرات حجم و جهت رشد یک جسم
از روی مقایسه دو عکس با **زاویه ثابت**.

## ساختار پروژه

```text
fsfmonitor/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py        # FastAPI: سرو صفحه + API (۳۳ مسیر)
│   │   ├── auth.py        # احراز هویت توکنی + ماتریس ۹ سطح دسترسی
│   │   ├── projects.py    # پروژه‌ها، عکس‌ها، نظرات، تاریخ شمسی
│   │   ├── pipeline.py    # بینایی ماشین: تراز (ECC)، سگمنتیشن (GrabCut)، سنجش
│   │   └── pdfgen.py      # گزارش PDF فارسی RTL
│   ├── requirements.txt
│   └── server.bat         # اجرای ساده روی ویندوز
├── frontend/
│   └── index.html         # تک‌صفحه فارسی RTL، موبایل‌فرندلی
├── Dockerfile
└── run.bat
```

## اجرا

### لوکال (ویندوز)
```bat
cd backend
pip install -r requirements.txt
run_server.bat
```
سپس مرورگر را روی `http://127.0.0.1:8095` باز کنید.

### Docker
```bash
docker build -t fsfmonitor .
docker run -p 8095:8095 -v fsf_data:/app/backend fsfmonitor
```

### دسترسی از بیرون (دامنه / HTTPS)
برنامه از طریق Cloudflare Tunnel روی دامنه `fsfmonitor.ir` سرو می‌شود.
تانل در `cloudflared.log` لاگ می‌کند. راه‌اندازی تانل دائمی:
```bat
cloudflared tunnel run fsfmonitor
```
(نیازمند `tunnel.yml` با hostname `fsfmonitor.ir` → `http://127.0.0.1:8095`)
