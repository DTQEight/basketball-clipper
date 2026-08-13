#!/bin/bash
# ============================================================
#  Basketball Goal Detection Service - Linux/macOS 一键启动
#  可移植版：不硬编码路径，跟随脚本位置自动定位
# ============================================================

set +e  # 即使脚本某步失败也不中断（方便看错误）

# 定位脚本目录：basketball-clipper/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 项目根：basketball-project/
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "============================================"
echo "  Basketball Goal Detection Service"
echo "============================================"
echo "  Script : $SCRIPT_DIR"
echo "  Project: $PROJECT_ROOT"
echo "  Browser: http://127.0.0.1:7871/"
echo "  Press Ctrl+C to stop"
echo "============================================"
echo ""

# 杀旧进程（端口占用时）
PID=$(lsof -ti:7871 2>/dev/null || true)
if [ -n "$PID" ]; then
    echo "[Port 7871] Killing old PID=$PID"
    kill -9 $PID 2>/dev/null || true
    sleep 2
fi

# 找 Python：优先项目内置环境 -> venv -> 系统 python
PYTHON=""
if [ -x "$PROJECT_ROOT/env/bin/python3" ]; then
    PYTHON="$PROJECT_ROOT/env/bin/python3"
elif [ -x "$PROJECT_ROOT/env/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/env/bin/python"
elif [ -x "$PROJECT_ROOT/venv/bin/python3" ]; then
    PYTHON="$PROJECT_ROOT/venv/bin/python3"
elif [ -x "$SCRIPT_DIR/venv/bin/python3" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python3"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

echo "  Python : $PYTHON"
echo ""

# 设置缓存根环境变量（可覆盖默认位置）
# export BBALL_CACHE_ROOT="$PROJECT_ROOT/cache"

cd "$SCRIPT_DIR"
echo "Starting service..."
"$PYTHON" demo_nicegui.py

echo ""
echo "Service stopped."
