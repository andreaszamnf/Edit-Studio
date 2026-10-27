@echo off
rem Starts Edit Studio with a console window (useful to see messages if something goes wrong).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run install.bat first.
    pause
    exit /b 1
)
set PYTHONUTF8=1
chcp 65001 >nul
".venv\Scripts\python.exe" -m edit_studio %*
if errorlevel 1 pause
