@echo off
set "PROJECT_DIR=E:\basketball-project\basketball-clipper"
cd /d "%PROJECT_DIR%"

echo ============================================
echo  Basketball Goal Detection Service
echo ============================================
echo  Browser: http://127.0.0.1:7871/
echo  Use incognito window to avoid SSE errors
echo  Press Ctrl+C to stop service
echo ============================================
echo.

REM Kill process on port 7871 if occupied
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":7871 " ^| findstr "LISTENING"') do (
    echo Killing old process PID=%%a on port 7871
    taskkill /F /PID %%a >nul 2>&1
)

echo Starting service...
E:\basketball-project\env\python.exe demo_nicegui.py
pause
