@echo off
REM Buildup Tracker - server worker (blocking). Used by Task Scheduler.
REM قبل از اجرای سرور جدید، هر پروسه‌ای که روی پورت 8095 در حال گوش دادن است را می‌بندد.
cd /d "%~dp0"
set PY=C:\Users\Administrator\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe

REM پیدا کردن و بستن پروسه‌ی قدیمی روی پورت 8095
for /f "tokens=5" %%p in ('netstat -ano ^| findstr "0.0.0.0:8095" ^| findstr LISTENING') do (
  taskkill /F /PID %%p >nul 2>&1
)
timeout /t 3 >nul

"%PY%" -m uvicorn app.main:app --host 0.0.0.0 --port 8095
