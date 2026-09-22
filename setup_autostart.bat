@echo off
REM Register Buildup Tracker autostart tasks (run as Administrator).
set ROOT=%~dp0
set BACKEND=%ROOT%backend

schtasks /create /tn "BuildupBackend" /tr "\"%BACKEND%server.bat\"" /sc ONSTART /ru SYSTEM /rl HIGHEST /f
schtasks /create /tn "BuildupTunnel" /tr "\"C:\Users\Administrator\cloudflared.exe\" tunnel --url http://127.0.0.1:8095 --logfile \"%ROOT%cloudflared.log\"" /sc ONSTART /ru SYSTEM /rl HIGHEST /f
schtasks /create /tn "BuildupWatchdog" /tr "\"%BACKEND%watchdog.bat\"" /sc MINUTE /mo 5 /ru SYSTEM /rl HIGHEST /f

echo.
schtasks /query | findstr Buildup
pause
