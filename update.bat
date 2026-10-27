@echo off
rem Updates Edit Studio to the latest version and refreshes its packages.
cd /d "%~dp0"
if exist ".git" (
    git pull --ff-only
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-windows.ps1"
echo.
pause
