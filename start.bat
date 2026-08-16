@echo off
setlocal enabledelayedexpansion
title VoiceLab
color 0A
cd /d "%~dp0"

cls
echo  ==========================================
echo   VoiceLab  ^|  AI + TTS + Video Studio
echo  ==========================================
echo.

REM -- Make sure setup has actually been run ==================
if not exist "venv\Scripts\activate.bat" (
    color 0C
    echo  [X] VoiceLab hasn't been set up yet on this PC.
    echo      Please run setup.bat first ^(just double-click it^).
    echo.
    pause
    exit /b 1
)

REM -- Warn early if port 8080 is already taken ================
netstat -ano | findstr ":8080" | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    color 0E
    echo  [!] Port 8080 is already in use by something else.
    echo      If that's an earlier VoiceLab window, close it first.
    echo      Continuing anyway - it may fail to start below.
    echo.
)

echo  [1/3] Activating environment...
call venv\Scripts\activate.bat

echo  [2/3] Starting server...
start /b "" python server.py

echo  [3/3] Loading models ^(30-45 sec first time^)...

set c=0
:check
timeout /t 3 /nobreak >nul
set /a c+=1
powershell -NoProfile -Command "try{Invoke-WebRequest 'http://localhost:8080' -TimeoutSec 2 -UseBasicParsing -EA Stop;exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel%==0 goto ready
if %c% lss 25 goto check

color 0E
echo.
echo  [!] Taking longer than expected ^(slow first-time model
echo      download, or something went wrong^). Opening the browser
echo      anyway - if it doesn't load in a minute, check
echo      TROUBLESHOOTING.md - "Server doesn't start" section.
echo.

:ready
start "" http://localhost:8080

echo.
echo  ==========================================
echo   VoiceLab is LIVE at localhost:8080
echo   Close this window to STOP the server
echo  ==========================================
echo.
pause
