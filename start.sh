#!/bin/bash
# ============================================================
#  Basketball Goal Detection Service - Linux/macOS 一键启动
#  可移植版：不硬编码路径，跟随脚本位置自动定位
#  日志轮转：按日切分，自动清理超过 7 天的旧日志
# ============================================================

set +e  # 即使脚本某步失败也不中断（方便看错误）

# 定位脚本目录（同时也是项目根目录，扁平仓库结构）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

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

# 日志目录 + 按日命名，清理 >7 天旧日志
LOG_DIR="$PROJECT_ROOT/cache/logs"
mkdir -p "$LOG_DIR"
DATE_TAG="$(date +%Y%m%d)"
LOG_FILE="$LOG_DIR/server-${DATE_TAG}.log"
echo "[Log] Purging logs older than 7 days ..."
find "$LOG_DIR" -name "server-*.log" -type f -mtime +7 -delete 2>/dev/null || true

# 服务端口（可用环境变量 BBALL_PORT 覆盖，demo_nicegui.py 同源读取）
PORT="${BBALL_PORT:-7871}"

echo ""
echo "============================================"
echo "  Basketball Goal Detection Service"
echo "============================================"
echo "  Script : $SCRIPT_DIR"
echo "  Project: $PROJECT_ROOT"
echo "  Python : $PYTHON"
echo "  Browser: http://127.0.0.1:${PORT}/"
echo "  Log    : $LOG_FILE"
echo "  Press Ctrl+C to stop"
echo "============================================"
echo ""

# 杀旧进程（端口占用时）
KILLED=0
PID=$(lsof -ti:$PORT 2>/dev/null || true)
if [ -n "$PID" ]; then
    echo "[Port $PORT] Killing old PID=$PID"
    kill -9 $PID 2>/dev/null || true
    KILLED=1
fi
[ "$KILLED" -eq 1 ] && sleep 2

# 设置缓存根环境变量（可覆盖默认位置）
# export BBALL_CACHE_ROOT="$PROJECT_ROOT/cache"

cd "$SCRIPT_DIR"
export BBALL_PORT="$PORT"
echo "Starting service..."
# tee: 控制台 + 日志双写，-a 表示同日追加
"$PYTHON" -u demo_nicegui.py 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "Service stopped."
