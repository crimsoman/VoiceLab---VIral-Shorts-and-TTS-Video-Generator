@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title VoiceLab Uninstall
color 0C
cd /d "%~dp0"

echo.
echo  ==========================================
echo   VoiceLab - Uninstall
echo  ==========================================
echo.
echo  This will remove:
echo   - The Python environment ^(venv folder^)
echo   - Any Desktop / Start Menu / Startup shortcuts
echo.
echo  This will NOT remove ^(unless you say yes below^):
echo   - Your exported videos/audio ^(exports folder^)
echo   - Downloaded AI models ^(hf_cache folder^)
echo   - Your project itself ^(server.py, index.html, etc.^)
echo.
set /p CONFIRM="  Continue? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo  Cancelled.
    pause
    exit /b 0
)

if exist venv (
    echo  Removing Python environment...
    rmdir /s /q venv
)

powershell -NoProfile -Command "Remove-Item -ErrorAction SilentlyContinue '%USERPROFILE%\Desktop\VoiceLab.lnk'" >nul 2>&1
powershell -NoProfile -Command "Remove-Item -ErrorAction SilentlyContinue ([Environment]::GetFolderPath('StartMenu') + '\Programs\VoiceLab.lnk')" >nul 2>&1
powershell -NoProfile -Command "Remove-Item -ErrorAction SilentlyContinue ([Environment]::GetFolderPath('Startup') + '\VoiceLab.lnk')" >nul 2>&1
echo  Removed shortcuts ^(if any existed^).

echo.
set /p WIPE="  Also delete your exported files and downloaded AI models? (Y/N): "
if /i "%WIPE%"=="Y" (
    if exist exports rmdir /s /q exports
    if exist hf_cache rmdir /s /q hf_cache
    if exist exports.db del /q exports.db
    if exist voicelab_settings.json del /q voicelab_settings.json
    echo  Deleted exports, cache, and settings.
)

echo.
echo  Done. To fully remove VoiceLab, you can now delete this
echo  whole folder manually.
echo.
pause
