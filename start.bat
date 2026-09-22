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

REM --- Service port (override with BBALL_PORT env var; demo_nicegui.py reads the same)
if not defined BBALL_PORT set "BBALL_PORT=7871"

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
echo  URL    : http://127.0.0.1:%BBALL_PORT%/
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

REM --- Kill old process if service port occupied
set "_KILLED=0"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%BBALL_PORT% " ^| findstr "LISTENING"') do (
    echo [Port %BBALL_PORT%] Killing old PID=%%a
    taskkill /F /PID %%a >nul 2>&1
    set "_KILLED=1"
)
if "!_KILLED!"=="1" timeout /t 2 /nobreak >nul

REM --- Start service (stdout/stderr -> log file ONLY, no pipeline)
REM Do NOT funnel python's stdout through a PowerShell pipeline
REM (`& python ... | ForEach-Object { Write-Host $_ }`): that pipeline STALLS whenever
REM the console stops draining its output -- e.g. QuickEdit text selection in the
REM window, or the console being spawned by a parent that never reads it (automation /
REM background launch). python then blocks on its next stdout write, and because the
REM app logs from the event-loop thread the WHOLE SERVICE FREEZES: TCP still accepts
REM (HTTP never answers), CPU 0%, and not a single line reaches the log.
REM Writing straight to the log file via cmd redirection has no such coupling.
REM Tee-Object -FilePath is avoided too: PS 5.1 always writes UTF-16LE there, which
REM contradicts chcp 65001 and the app's own UTF-8 app.log.
REM NOTE: keep every REM/echo line in this file ASCII-only. cmd parses the batch file
REM in the *current* codepage, so a non-ASCII comment before `chcp 65001` is decoded
REM as GBK and executed as a command.
cd /d "%SCRIPT_DIR%"
echo Starting service...
echo   Live log: powershell -NoProfile "Get-Content '%LOG_FILE%' -Wait -Tail 20"
"%PYTHON%" -u demo_nicegui.py >> "%LOG_FILE%" 2>&1

echo.
echo Service stopped. Press any key to exit.
pause >nul
endlocal
