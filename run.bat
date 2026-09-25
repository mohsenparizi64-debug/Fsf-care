@echo off
cd /d "%~dp0"
title fsfmonitor Launcher

echo ===================================================
echo   fsfmonitor Launcher
echo ===================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not added to PATH!
    pause
    exit /b 1
)

if not exist backend (
    echo ERROR: Folder backend not found!
    pause
    exit /b 1
)

if not exist frontend (
    echo ERROR: Folder frontend not found!
    pause
    exit /b 1
)

echo [1/2] Preparing Python Backend...
cd /d "%~dp0backend"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Installing Python dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo [2/2] Starting fsfmonitor on port 8095...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8095

pause
