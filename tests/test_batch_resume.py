# -*- coding: utf-8 -*-
"""批量任务进度恢复（batch_task_status）单元测试：字段结构与生命周期标记。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import state  # noqa: E402


def test_batch_task_status_schema():
    """全局进度 dict 必须含恢复 UI 所需的全部字段（防误删/改名破坏页面轮询）。"""
    required = {"running", "current", "index", "total", "pct",
                "message", "started_at", "cancel_requested"}
    assert required <= set(state.batch_task_status.keys())
    assert state.batch_task_status["running"] in (True, False)


def test_batch_task_status_update_lifecycle():
    """模拟 detection 写进度 → 页面读快照 → 结束清理 的完整生命周期。"""
    # 启动（detection 批量循环入口写法）
    state.batch_task_status.update({
        "running": True, "current": None, "index": 0, "total": 4,
        "pct": 0.0, "message": "批量识别启动...",
        "started_at": "12:00:00", "cancel_requested": False,
    })
    # 检测中（_cb 回调写法：整体百分比归一化）
    overall = (1 + 50 / 100.0) / 4 * 100.0
    state.batch_task_status.update({
        "running": True, "current": "2nd.mov", "index": 2, "total": 4,
        "pct": round(overall, 1), "message": "检测中 50%",
    })
    snap = dict(state.batch_task_status)  # 页面读快照（dict() 复制原子性同源）
    assert snap["running"] is True
    assert snap["index"] == 2 and snap["total"] == 4
    assert 37.0 < snap["pct"] < 38.0
    # 取消请求（跨页面可见）
    state.batch_task_status["cancel_requested"] = True
    assert state.batch_task_status["cancel_requested"] is True
    # 结束清理（detection 批量循环出口写法）
    state.batch_task_status.update({
        "running": False, "current": None, "pct": 100.0,
        "message": "已完成", "cancel_requested": False,
    })
    assert state.batch_task_status["running"] is False
    assert state.batch_task_status["pct"] == 100.0
