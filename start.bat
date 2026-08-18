@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  Basketball Goal Detection Service - Auto Start (Windows)
REM  Portable: no hardcoded drive/path, auto-follow script dir
REM  Log rotation: daily log file, auto-purge >7 days
REM ============================================================

REM --- Script dir is also the project root (flat repo structure)
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_ROOT=%SCRIPT_DIR%"

REM --- Resolve Python: prefer project built-in env
set "PYTHON=%PROJECT_ROOT%\env\python.exe"
if not exist "%PYTHON%" set "PYTHON=%PROJECT_ROOT%\env\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

REM --- Log dir + today's log filename (YYYYMMDD)
set "LOG_DIR=%PROJECT_ROOT%\cache\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "DATE_TAG=%%i"
set "LOG_FILE=%LOG_DIR%\server-%DATE_TAG%.log"

REM --- Purge log files older than 7 days
echo [Log] Purging logs older than 7 days ...
forfiles /p "%LOG_DIR%" /m "server-*.log" /d -7 /c "cmd /c del /q @path >nul 2>&1" >nul 2>&1

echo.
echo ============================================
echo  Basketball Goal Detection Service
echo ============================================
echo  Script : %SCRIPT_DIR%
echo  Python : %PYTHON%
echo  URL    : http://127.0.0.1:7871/
echo  Log    : %LOG_FILE%
echo  Ctrl+C to stop
echo ============================================
echo.

REM --- Force UTF-8 end-to-end (fix Chinese mojibake in console + log):
REM uv-managed Python defaults to UTF-8 output, but PowerShell decodes native
REM command output with the console codepage (GBK on zh-CN) -> garbled text.
REM Pin both sides to UTF-8 regardless of Python distribution.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul

REM --- Kill old process if port 7871 occupied
set "_KILLED=0"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":7871 " ^| findstr "LISTENING"') do (
    echo [Port 7871] Killing old PID=%%a
    taskkill /F /PID %%a >nul 2>&1
    set "_KILLED=1"
)
if "!_KILLED!"=="1" timeout /t 2 /nobreak >nul

REM --- Start service (stdout/stderr -> console + log file, PowerShell built-in Tee)
REM [Console]::OutputEncoding=UTF8 makes PowerShell decode python's UTF-8 output correctly
cd /d "%SCRIPT_DIR%"
echo Starting service...
powershell -NoProfile -Command "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; & '%PYTHON%' -u demo_nicegui.py 2>&1 | Tee-Object -FilePath '%LOG_FILE%' -Append"

echo.
echo Service stopped. Press any key to exit.
pause >nul
endlocal
