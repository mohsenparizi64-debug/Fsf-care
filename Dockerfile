# Fsf-care — Dockerfile
# اپ فول‌استک فارسی (FastAPI + تک‌صفحه HTML) در یک ایمیج
# ساخت: docker build -t fsf-care .
# اجرا: docker run -p 8092:8092 -v fsf_data:/app/backend fsf-care

FROM python:3.11-slim

# باینری‌های سیستمی مورد نیاز OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# اول requirements رو کپی می‌کنیم تا لایه وابستگی‌ها کش بشه
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# کد برنامه (فایل‌های حساس در .gitignore هستند و کپی نمی‌شوند)
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# پوشه‌های داده در زمان اجرا ساخته می‌شوند
RUN mkdir -p backend/auth backend/projects backend/uploads

EXPOSE 8092

# uvicorn با host 0.0.0.0 برای دسترسی از بیرون کانتینر
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8092", "--app-dir", "backend"]
