@echo off
cd /d "%~dp0"
title fsfmonitor Server (Task Scheduler worker)
REM fsfmonitor backend worker. Blocking. Started by fsfmonitorBackend task.
C:\Users\Administrator\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8095
