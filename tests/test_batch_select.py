"""批量识别「单节勾选」：run_batch_detect(video_paths=...) 只跑勾选的视频。

覆盖三点：传子集只跑子集、不传时跑 state.batch_files 全部、空集合直接报错
（不能因为勾选为空就退化成"跑全量"，否则误触会把不打算重跑的视频覆盖掉）。
"""
import numpy as np
import pytest

import services.detection as detection
from services import state


def _info(_path):
    return {"total": 300, "fps": 30.0, "codec": "h264", "width": 320, "height": 240}


@pytest.fixture
def batch_env(monkeypatch):
    """4 个已标定视频的场景；run_detect 被替换为只记录"实际跑到哪个视频"。"""
    vids = [f"C:/fake/v{i}.mp4" for i in range(4)]
    called = []

    def _fake_run_detect(*_a, **_kw):
        called.append(state.video_state["path"])
        state.last_goals = [1.0, 2.0]
        return "ok", True

    snap = (state.batch_files, state.batch_calibs, state.batch_selected,
            state.batch_current_video)
    state.batch_files = list(vids)
    state.batch_calibs = {v: {"hoop": [0, 0, 10, 10], "baseline_idx": 0} for v in vids}
    state.batch_selected = set(vids)
    state.batch_current_video = None
    state.cancel_event.clear()
    monkeypatch.setattr(detection, "get_video_info", _info)
    monkeypatch.setattr(detection, "read_frame",
                        lambda *_a, **_kw: np.zeros((240, 320, 3), dtype=np.uint8))
    monkeypatch.setattr(detection, "run_detect", _fake_run_detect)
    try:
        yield vids, called
    finally:
        (state.batch_files, state.batch_calibs, state.batch_selected,
         state.batch_current_video) = snap
        state.batch_results.clear()
        state.cancel_event.clear()


def test_only_selected_videos_are_detected(batch_env):
    """勾选 2 个 → 只跑这 2 个，目录里其余视频不动。"""
    vids, called = batch_env
    status, ok = detection.run_batch_detect(0, 0, 0.2, 2.0,
                                            video_paths=[vids[1], vids[2]])
    assert ok is True
    assert called == [vids[1], vids[2]]
    assert "2/2" in status
    # 列表与标定保持完整：勾选只影响本轮跑哪些
    assert state.batch_files == vids
    assert set(state.batch_calibs) == set(vids)


def test_default_runs_all_batch_files(batch_env):
    """不传 video_paths → 行为与旧版一致，跑全部。"""
    vids, called = batch_env
    status, ok = detection.run_batch_detect(0, 0, 0.2, 2.0)
    assert ok is True
    assert called == vids
    assert "4/4" in status


def test_empty_selection_refuses_to_run(batch_env):
    """勾选为空 → 报错返回，不能退化成"跑全量"。"""
    _vids, called = batch_env
    status, ok = detection.run_batch_detect(0, 0, 0.2, 2.0, video_paths=[])
    assert ok is False
    assert called == []
    assert "请先加载文件夹" in status
