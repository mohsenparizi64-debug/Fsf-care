@echo off
title fsfmonitor Server Launcher
color 0A
chcp 65001 > nul

echo ========================================================
echo   Starting fsfmonitor
echo   Backend + Frontend: Port 8095 (Host: 0.0.0.0)
echo ========================================================
echo.

echo [1/1] Launching FastAPI Server on Port 8095...
start "fsfmonitor Backend" cmd /k "cd /d %~dp0backend && (if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat) && python -m uvicorn app.main:app --host 0.0.0.0 --port 8095"

echo.
echo ========================================================
echo   SERVER LAUNCHED SUCCESSFULLY!
echo.
echo   Local Access (Inside VPS):
echo   - App: http://localhost:8095
echo   - API Health: http://localhost:8095/api/health
echo.
echo   Remote / Mobile Access:
echo   - Application: http://YOUR_SERVER_IP:8095
echo ========================================================
echo.
pause
