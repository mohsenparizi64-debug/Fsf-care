@echo off
REM fsfmonitor - watchdog: restart ONLY our server (port 8095) if health fails.
set LOG=%~dp0watchdog.log
curl -s -m 10 -o nul -w "%%{http_code}" http://127.0.0.1:8095/api/health > "%TEMP%\bt_health.txt" 2>nul
set /p CODE=<"%TEMP%\bt_health.txt"
if "%CODE%"=="200" (
  echo %DATE% %TIME% OK >> "%LOG%"
  exit /b 0
)
echo %DATE% %TIME% DOWN(code=%CODE%) restarting >> "%LOG%"
for /f "tokens=5" %%p in ('netstat -ano ^| findstr "0.0.0.0:8095" ^| findstr LISTENING') do taskkill /F /PID %%p >nul 2>&1
timeout /t 3 >nul
start "fsfmonitor Backend" /min cmd /c ""%~dp0server.bat""
echo %DATE% %TIME% restarted >> "%LOG%"
