@echo off
setlocal EnableExtensions

REM ============================================================
REM  Basketball Goal Detection Service - Auto Start (Windows)
REM  Portable: no hardcoded drive/path, auto-follow script dir
REM ============================================================

REM --- Script dir is also the project root (flat repo structure)
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_ROOT=%SCRIPT_DIR%"

REM --- Resolve Python: prefer project built-in env
set "PYTHON=%PROJECT_ROOT%\env\python.exe"
if not exist "%PYTHON%" set "PYTHON=%PROJECT_ROOT%\env\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo.
echo ============================================
echo  Basketball Goal Detection Service
echo ============================================
echo  Script : %SCRIPT_DIR%
echo  Python : %PYTHON%
echo  URL    : http://127.0.0.1:7871/
echo  Ctrl+C to stop
echo ============================================
echo.

REM --- Kill old process if port 7871 occupied
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":7871 " ^| findstr "LISTENING"') do (
    echo [Port 7871] Killing old PID=%%a
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 2 /nobreak >nul
)

REM --- Start service
cd /d "%SCRIPT_DIR%"
echo Starting service...
"%PYTHON%" demo_nicegui.py

echo.
echo Service stopped. Press any key to exit.
pause >nul
endlocal
