"""run_detect 解码 0 帧告警回归（B2）。

用空帧 stub 替换 VideoReader，验证"全程未解码到任何帧"时返回明确错误
而不是以"检测完成 / 0 进球"伪装成功收尾（旧实现会写一条空历史误导排查）。
"""
import os
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


class TestExportSingleClipHq:
    """export_single_clip_hq：卡片「导出」按集锦规格现切单球（hq 质量 + 时长）。

    验证数据源定位（全局/快照）、索引边界、源缺失报错，cut_clips 被桩替换
    避免真实 ffmpeg 调用。
    """
    def test_global_mode_uses_last_goal_clips(self, monkeypatch, tmp_path):
        import cutter.ffmpeg_cutter as fc
        snap = _snapshot_state()
        try:
            fake_src = tmp_path / "src.mp4"
            fake_src.write_bytes(b"x")
            state.video_state.update(
                path=str(fake_src), total=100, fps=30.0,
                current_frame=0, width=320, height=240, codec="h264")
            state.last_goals = [10.0]
            state.last_goal_clips = [{"ts": 10.0, "path": "preview.mp4",
                                      "idx": 0, "mark": None, "person": None}]

            calls = {}
            def _fake_cut(video_path, timestamps, pre_roll=5, post_roll=5,
                          min_gap=8, output_path=None, **kw):
                calls.update(video_path=video_path, timestamps=list(timestamps),
                             pre_roll=pre_roll, post_roll=post_roll,
                             min_gap=min_gap, output_path=output_path)
                out = output_path or "default.mp4"
                os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                open(out, "w").close()
                return out

            monkeypatch.setattr(fc, "cut_clips", _fake_cut)
            path, status = detection.export_single_clip_hq(0, pre_roll=5, post_roll=5)
            assert path is not None and os.path.exists(path)
            # 以集锦规格（非短预览 ±3s）调用：时长取 pre/post 参数，min_gap=0 不合并
            assert calls["video_path"] == str(fake_src)
            assert calls["timestamps"] == [10.0]
            assert calls["pre_roll"] == 5 and calls["post_roll"] == 5
            assert calls["min_gap"] == 0
            assert "已按集锦规格导出" in status
        finally:
            state.last_goals = []
            state.last_goal_clips = []
            _restore_state(snap)

    def test_snapshot_mode_uses_batch_results(self, monkeypatch, tmp_path):
        import cutter.ffmpeg_cutter as fc
        snap = _snapshot_state()
        try:
            fake_src = tmp_path / "batch.mp4"
            fake_src.write_bytes(b"x")
            state.batch_results[str(fake_src)] = {
                "goals": [20.0], "clips": [{"ts": 20.0, "path": "p.mp4",
                                            "idx": 0, "mark": None, "person": None}],
                "kept": set(), "finished_at": "x"}

            calls = {}
            def _fake_cut(video_path, timestamps, pre_roll=5, post_roll=5,
                          min_gap=8, output_path=None, **kw):
                calls.update(video_path=video_path, timestamps=list(timestamps))
                out = output_path or "default.mp4"
                os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                open(out, "w").close()
                return out
            monkeypatch.setattr(fc, "cut_clips", _fake_cut)
            path, status = detection.export_single_clip_hq(
                0, video_path=str(fake_src), pre_roll=3, post_roll=4)
            assert path is not None
            assert calls["video_path"] == str(fake_src)
            assert calls["timestamps"] == [20.0]
        finally:
            state.batch_results.clear()
            _restore_state(snap)

    def test_invalid_index_returns_error(self):
        snap = _snapshot_state()
        try:
            state.video_state.update(path="C:/fake/v.mp4", total=100, fps=30.0,
                                     current_frame=0, width=320, height=240,
                                     codec="h264")
            state.last_goal_clips = [{"ts": 10.0, "path": "p.mp4"}]
            path, status = detection.export_single_clip_hq(99)
            assert path is None
            assert "索引无效" in status
        finally:
            state.last_goal_clips = []
            _restore_state(snap)

    def test_no_video_loaded_returns_error(self):
        snap = _snapshot_state()
        try:
            state.video_state["path"] = None
            path, status = detection.export_single_clip_hq(0)
            assert path is None
            assert "请先加载视频" in status
        finally:
            _restore_state(snap)
