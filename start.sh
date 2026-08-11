#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  Basketball Goal Detection Service"
echo "============================================"
echo "  Browser: http://127.0.0.1:7871/"
echo "  Use incognito window to avoid SSE errors"
echo "  Press Ctrl+C to stop service"
echo "============================================"
echo ""

# Kill process on port 7871 if occupied
PID=$(lsof -ti:7871 2>/dev/null || true)
if [ -n "$PID" ]; then
    echo "Killing old process PID=$PID on port 7871"
    kill -9 $PID 2>/dev/null || true
fi

# 激活虚拟环境（如果存在）
if [ -f "$SCRIPT_DIR/../venv/bin/activate" ]; then
    source "$SCRIPT_DIR/../venv/bin/activate"
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

echo "Starting service..."
python demo_nicegui.py