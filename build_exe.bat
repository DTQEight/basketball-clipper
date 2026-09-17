@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PYTHON=%SCRIPT_DIR%\env\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] 找不到 Python 环境: %PYTHON%
    echo 请先创建虚拟环境并安装依赖
    pause
    exit /b 1
)

echo ============================================
echo  Basketball Clipper - 打包 EXE
echo ============================================
echo.

REM --- 1. 确认 PyInstaller 已安装 ---
"%PYTHON%" -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [Setup] 安装 PyInstaller ...
    "%PYTHON%" -m pip install pyinstaller
)

REM --- 2. 清理旧产物 ---
if exist "%SCRIPT_DIR%\dist\basketball-clipper" (
    echo [Clean] 删除旧 dist ...
    rmdir /s /q "%SCRIPT_DIR%\dist\basketball-clipper"
)
if exist "%SCRIPT_DIR%\build" (
    echo [Clean] 删除旧 build ...
    rmdir /s /q "%SCRIPT_DIR%\build"
)

REM --- 3. 执行打包 ---
echo.
echo [Build] 开始打包（torch+CUDA 约 4.5GB，需 5-15 分钟）...
echo.
"%PYTHON%" -m PyInstaller "%SCRIPT_DIR%\build_exe.spec" --noconfirm

if errorlevel 1 (
    echo.
    echo [FAILED] 打包失败，请看上方错误信息
    pause
    exit /b 1
)

REM --- 4. 结果汇总 ---
echo.
echo ============================================
echo  打包完成！
echo ============================================
set "OUT=%SCRIPT_DIR%\dist\basketball-clipper"
echo  产物目录: %OUT%
echo  主程序: %OUT%\basketball-clipper.exe
echo.

REM 计算体积
for /f "delims=" %%a in ('powershell -NoProfile -Command "(Get-ChildItem '%OUT%' -Recurse -File | Measure-Object Length -Sum).Sum / 1GB"') do set "SIZE=%%a"
echo  总体积: !SIZE! GB
echo.
echo  使用方法:
echo    双击 basketball-clipper.exe
echo    浏览器自动打开 http://127.0.0.1:7871
echo.
echo  注意: 首次运行需加载 YOLO 模型 + CUDA，约 10-30 秒
pause
endlocal
