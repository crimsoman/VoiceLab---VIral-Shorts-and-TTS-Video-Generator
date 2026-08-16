@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title VoiceLab Setup
color 0B
cd /d "%~dp0"

echo.
echo  ==========================================
echo    VoiceLab  -  First-Time Setup
echo  ==========================================
echo.
echo  This will check your PC, set up a private Python
echo  environment inside this folder, and install everything
echo  VoiceLab needs. Nothing is installed system-wide.
echo.
pause

REM ══════════════════════════════════════════════════════════
REM STEP 1 — Find a usable Python (3.10 - 3.12)
REM ══════════════════════════════════════════════════════════
echo.
echo  [1/8] Checking for Python...
set "PY_CMD="

where py >nul 2>&1
if %errorlevel%==0 (
    for %%V in (3.12 3.11 3.10) do (
        if not defined PY_CMD (
            py -%%V --version >nul 2>&1
            if !errorlevel!==0 set "PY_CMD=py -%%V"
        )
    )
)
if not defined PY_CMD (
    where python >nul 2>&1
    if !errorlevel!==0 set "PY_CMD=python"
)

if not defined PY_CMD (
    color 0C
    echo.
    echo  [X] Python was not found on this PC.
    echo.
    echo      VoiceLab needs Python 3.10, 3.11, or 3.12.
    echo      1. Go to: https://www.python.org/downloads/
    echo      2. Download Python 3.11 ^(recommended^)
    echo      3. Run the installer
    echo         -^> IMPORTANT: tick "Add python.exe to PATH" during install
    echo      4. Restart this setup.bat afterwards
    echo.
    start "" https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Verify the version is actually in the supported range
%PY_CMD% -c "import sys; v=sys.version_info; sys.exit(0 if (v[0]==3 and 10<=v[1]<=12) else 1)"
if errorlevel 1 (
    color 0E
    echo.
    echo  [!] Found Python, but it's an unsupported version.
    echo      VoiceLab needs Python 3.10, 3.11, or 3.12.
    echo      Install one of those from https://www.python.org/downloads/
    echo      and run setup.bat again.
    echo.
    start "" https://www.python.org/downloads/
    pause
    exit /b 1
)
echo      OK - using: %PY_CMD%

REM ══════════════════════════════════════════════════════════
REM STEP 2 — Free disk space check (need ~8GB minimum)
REM ══════════════════════════════════════════════════════════
echo.
echo  [2/8] Checking free disk space...
set "FREE_GB=0"
for /f "usebackq delims=" %%s in (`powershell -NoProfile -Command "[math]::Round((Get-PSDrive (Get-Location).Drive.Name).Free / 1GB)"`) do set "FREE_GB=%%s"
echo      Free space on this drive: !FREE_GB! GB
if !FREE_GB! LSS 8 (
    color 0E
    echo.
    echo  [!] Less than 8GB free. VoiceLab + its AI models can use
    echo      several GB. Setup will continue, but you may run out
    echo      of space during first-time model downloads.
    echo.
    pause
)

REM ══════════════════════════════════════════════════════════
REM STEP 3 — Internet check
REM ══════════════════════════════════════════════════════════
echo.
echo  [3/8] Checking internet connection...
powershell -NoProfile -Command "try{Invoke-WebRequest -Uri https://pypi.org -TimeoutSec 6 -UseBasicParsing | Out-Null; exit 0}catch{exit 1}" >nul 2>&1
if errorlevel 1 (
    color 0C
    echo.
    echo  [X] No internet connection detected.
    echo      Setup needs internet to download packages the first time.
    echo      Connect to the internet and run setup.bat again.
    echo.
    pause
    exit /b 1
)
echo      OK - connected.

REM ══════════════════════════════════════════════════════════
REM STEP 4 — Port 8080 availability
REM ══════════════════════════════════════════════════════════
echo.
echo  [4/8] Checking if port 8080 is free...
netstat -ano | findstr ":8080" | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    color 0E
    echo.
    echo  [!] Something else is already using port 8080.
    echo      VoiceLab needs this port to run. Close whatever is using
    echo      it ^(another VoiceLab instance? Another local app?^), or
    echo      VoiceLab will fail to start later.
    echo.
    pause
) else (
    echo      OK - port 8080 is free.
)

REM ══════════════════════════════════════════════════════════
REM STEP 5 — Create/activate virtual environment
REM ══════════════════════════════════════════════════════════
echo.
echo  [5/8] Setting up private Python environment...
if not exist venv (
    %PY_CMD% -m venv venv
    if errorlevel 1 (
        color 0C
        echo.
        echo  [X] Failed to create the virtual environment.
        echo      See TROUBLESHOOTING.md - "venv creation fails" section.
        pause
        exit /b 1
    )
) else (
    echo      Environment already exists - reusing it.
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1

REM ══════════════════════════════════════════════════════════
REM STEP 6 — GPU scan, then install matching PyTorch build
REM ══════════════════════════════════════════════════════════
echo.
echo  [6/8] Scanning your PC for a GPU...
set "GPU_MODE=0"
set "GPU_NAME="
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    set "GPU_MODE=1"
    for /f "usebackq delims=" %%g in (`nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul`) do set "GPU_NAME=%%g"
    echo      NVIDIA GPU found: !GPU_NAME!
    echo      -^> Installing GPU-accelerated PyTorch. AI Image/Music/
    echo         Voice Clone features will run fast.
    pip install torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
) else (
    echo      No NVIDIA GPU detected - that's fine, VoiceLab runs on CPU too.
    echo      -^> Installing CPU-only PyTorch. Core features ^(voiceover,
    echo         captions, video export^) run at full speed. Heavier AI
    echo         features ^(AI Image/Music^) will just be slower.
    pip install torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
)
if errorlevel 1 (
    color 0C
    echo.
    echo  [X] PyTorch install failed. See TROUBLESHOOTING.md.
    pause
    exit /b 1
)

REM ══════════════════════════════════════════════════════════
REM STEP 7 — Install the rest of the app's requirements
REM ══════════════════════════════════════════════════════════
echo.
echo  [7/8] Installing VoiceLab's remaining requirements...
echo        ^(this can take a few minutes the first time^)
pip install -r requirements.txt
if errorlevel 1 (
    color 0C
    echo.
    echo  [X] Some packages failed to install.
    echo      See TROUBLESHOOTING.md - "pip install fails" section.
    pause
    exit /b 1
)
echo      OK - all core requirements installed.

REM ══════════════════════════════════════════════════════════
REM STEP 8 — Ollama check (used for AI script writing)
REM ══════════════════════════════════════════════════════════
echo.
echo  [8/8] Checking for Ollama ^(needed for AI script writing^)...
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    color 0E
    echo.
    echo      Ollama isn't installed. VoiceLab's AI script/caption
    echo      writing features need it. Everything else in the app
    echo      ^(voiceover, editing, export^) works fine without it.
    echo.
    set /p INSTALL_OLLAMA="      Open the Ollama download page now? (Y/N): "
    if /i "!INSTALL_OLLAMA!"=="Y" start "" https://ollama.com/download/windows
    echo      After installing it, just relaunch VoiceLab - it will
    echo      detect Ollama automatically. No need to rerun setup.
    color 0B
) else (
    echo      Ollama found.
    ollama list 2>nul | findstr /i "llama3.2" >nul 2>&1
    if errorlevel 1 (
        echo.
        set /p PULL_MODEL="      No AI model downloaded yet. Get a small starter model now, ~2GB (Y/N): "
        if /i "!PULL_MODEL!"=="Y" (
            echo      Downloading llama3.2:3b - this runs on almost any PC...
            ollama pull llama3.2:3b
        ) else (
            echo      No problem - pick any model later from the app's
            echo      Settings tab. It recommends models based on your PC.
        )
    ) else (
        echo      A local AI model is already installed.
    )
)

REM ══════════════════════════════════════════════════════════
REM Shortcuts
REM ══════════════════════════════════════════════════════════
echo.
echo  ==========================================
echo   Optional: quick-launch shortcuts
echo  ==========================================
set /p WANT_DESKTOP="  Add a Desktop shortcut? (Y/N): "
if /i "%WANT_DESKTOP%"=="Y" (
    powershell -NoProfile -Command "$s=(New-Object -COM WScript.Shell).CreateShortcut('%USERPROFILE%\Desktop\VoiceLab.lnk'); $s.TargetPath='%~dp0start.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=1; $s.Save()" >nul 2>&1
    echo      Added to Desktop.
)
set /p WANT_STARTMENU="  Add to Start Menu? (Y/N): "
if /i "%WANT_STARTMENU%"=="Y" (
    powershell -NoProfile -Command "$s=(New-Object -COM WScript.Shell).CreateShortcut([Environment]::GetFolderPath('StartMenu') + '\Programs\VoiceLab.lnk'); $s.TargetPath='%~dp0start.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=1; $s.Save()" >nul 2>&1
    echo      Added to Start Menu.
)
set /p WANT_STARTUP="  Auto-start VoiceLab when Windows starts up? (Y/N): "
if /i "%WANT_STARTUP%"=="Y" (
    powershell -NoProfile -Command "$s=(New-Object -COM WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Startup') + '\VoiceLab.lnk'); $s.TargetPath='%~dp0start.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.Save()" >nul 2>&1
    echo      VoiceLab will now start automatically on login.
)

color 0A
echo.
echo  ==========================================
echo   Setup complete!
echo  ==========================================
echo.
echo   Double-click start.bat ^(or your new shortcut^) any time
echo   to launch VoiceLab.
echo.
pause
