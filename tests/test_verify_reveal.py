# -*- coding: utf-8 -*-
"""验证进度占位（全部标完再显示片段）单元测试：状态时序契约。

UI 的 _refresh_result_cards 在 verify_status[vp].running=True 时渲染进度
占位、不渲染卡片；running 翻 False 后一次性出卡片。这里验证两个关键
时序保证：
  1. start_verify_thread 同步预置 running=True（否则检测完成回调先渲染
     无分数卡片，占位形同虚设）
  2. 模型不可用 / 异常路径必须清 running（否则占位永远等不到完成）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import goal_verifier as gv  # noqa: E402


def test_start_verify_thread_preinitializes_status(monkeypatch):
    """start_verify_thread 返回时 verify_status 必须已是 running=True。"""
    calls = []

    def _stub(*a, **k):
        calls.append(a)
        return (0, 0, 0)

    monkeypatch.setattr(gv, "verify_clips", _stub)
    vp = "x://fake/verify_preinit.mp4"
    gv.verify_status.pop(vp, None)
    gv.start_verify_thread(vp, [{"ts": 1.0}, {"ts": 2.0}], (0, 0, 10, 10))
    st = gv.verify_status.get(vp)
    assert st is not None, "状态未同步预置"
    assert st["running"] is True
    assert st["total"] == 2 and st["done"] == 0
    # worker 线程确实调用了 verify_clips（真实实现负责最终清 running）
    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        time.sleep(0.05)
    assert calls, "worker 未调用 verify_clips"


def test_verify_clips_unavailable_model_clears_running(monkeypatch):
    """模型不可用时 verify_clips 必须清 running（防占位永久卡死）。"""
    monkeypatch.setattr(gv, "_load_model", lambda: None)
    vp = "x://fake/verify_nomodel.mp4"
    gv.verify_status[vp] = {"done": 0, "total": 3, "running": True}
    ret = gv.verify_clips(vp, [{"ts": 1.0}], (0, 0, 10, 10))
    assert ret == (0, 0, 0)
    assert gv.verify_status[vp]["running"] is False
    gv.verify_status.pop(vp, None)
