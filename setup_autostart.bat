@echo off
REM Register fsfmonitor autostart tasks (run as Administrator).
set ROOT=%~dp0
set BACKEND=%ROOT%backend

schtasks /create /tn "fsfmonitorBackend" /tr "\"%BACKEND%server.bat\"" /sc ONSTART /ru SYSTEM /rl HIGHEST /f
schtasks /create /tn "fsfmonitorTunnel" /tr "\"C:\Users\Administrator\fsfmonitor\cloudflared.exe\" tunnel --config %ROOT%tunnel.yml run fsfmonitor" /sc ONSTART /ru SYSTEM /rl HIGHEST /f
schtasks /create /tn "fsfmonitorWatchdog" /tr "\"%BACKEND%watchdog.bat\"" /sc MINUTE /mo 5 /ru SYSTEM /rl HIGHEST /f

echo.
schtasks /query | findstr fsfmonitor
pause
