"""run_detect 解码 0 帧告警回归（B2）。

用空帧 stub 替换 VideoReader，验证"全程未解码到任何帧"时返回明确错误
而不是以"检测完成 / 0 进球"伪装成功收尾（旧实现会写一条空历史误导排查）。
"""
import numpy as np

import services.detection as detection
from services import state


class _EmptyReader:
    """模拟解码器：区间内一帧都解不出来（decode_errors>0）。"""

    def __init__(self, path):
        self.path = path
        self.decode_errors = 3          # 全部 packet 解码失败被容错跳过
        self.total = 1
        self.fps = 30.0

    def iter_frames(self, start=0, end=None, batch=1):
        return iter(())

    def close(self):
        pass


def _snapshot_state():
    return {
        "video_state": dict(state.video_state),
        "calib": dict(state.calib),
    }


def _restore_state(snap):
    state.video_state.clear()
    state.video_state.update(snap["video_state"])
    state.calib.clear()
    state.calib.update(snap["calib"])
    state.cancel_event.clear()


def test_run_detect_zero_decoded_frames_returns_error(monkeypatch):
    """B2: 0 帧解码 → 明确报错，不写历史不假装成功。"""
    snap = _snapshot_state()
    try:
        state.video_state.update(
            path="C:/fake/empty.mp4", total=5000, fps=30.0,
            current_frame=0, width=320, height=240, codec="h264")
        state.calib.update(
            hoop=[100, 100, 160, 140],
            baseline_frame=np.full((240, 320, 3), 100, dtype=np.uint8),
            baseline_idx=0, clicks=[])
        state.cancel_event.clear()

        monkeypatch.setattr(detection, "get_device", lambda: "cuda:0")
        monkeypatch.setattr(detection, "VideoReader", _EmptyReader)
        monkeypatch.setattr(detection, "get_ball_model",
                            lambda: (object(), "fake.pt"))
        monkeypatch.setattr(detection, "get_ball_class_ids",
                            lambda model, weights_path="": [0])

        status, ok = detection.run_detect(
            start_frame=0, end_frame=100, ball_conf=0.3, min_gap_sec=3.0,
            auto_threshold=False, task_token=0)
        assert ok is False
        assert "未解码到任何帧" in status
        # 不残留任何"看起来成功"的空结果
        assert state.last_goals == []
        assert state.last_goal_clips == []
    finally:
        _restore_state(snap)
