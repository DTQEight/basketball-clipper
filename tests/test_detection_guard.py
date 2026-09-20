"""run_detect 解码 0 帧告警回归（B2）。

用空帧 stub 替换 VideoReader，验证"全程未解码到任何帧"时返回明确错误
而不是以"检测完成 / 0 进球"伪装成功收尾（旧实现会写一条空历史误导排查）。
"""
import os
import numpy as np

import services.detection as detection
from services import state, goal_verifier


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


class TestAiVerifySwitch:
    """AI 识别总开关：关闭后四臂复核整段跳过（不产分数、不打自动标记）。

    关闭是"不再跑"，不是"清掉已有结果"——缓存分数与历史标签都不动，
    卡片上的 AI 徽标由 UI 在显示层屏蔽，重开即时恢复。
    """

    def test_mark_auto_noop_when_disabled(self):
        clips = [{"ts": 10.0, "path": "p.mp4", "idx": 0}]
        prev = goal_verifier.is_enabled()
        try:
            goal_verifier.set_enabled(False)
            assert goal_verifier.is_enabled() is False
            # 视频路径不存在：若真进了打分流程会去读视频，返回 0 且不落任何字段
            assert goal_verifier.mark_auto(
                clips, "C:/fake/none.mp4", [0, 0, 10, 10]) == 0
            assert "score" not in clips[0]
            assert "auto" not in clips[0]
            assert "mark" not in clips[0]
        finally:
            goal_verifier.set_enabled(prev)

    def test_disable_keeps_existing_scores_and_marks(self):
        clips = [{"ts": 10.0, "score": 0.9, "verify_score": 0.9, "auto": True,
                  "mark": "keep", "mark_source": "auto"}]
        prev = goal_verifier.is_enabled()
        try:
            goal_verifier.set_enabled(False)
            assert goal_verifier.mark_auto(
                clips, "C:/fake/none.mp4", [0, 0, 10, 10]) == 0
            assert clips[0]["score"] == 0.9
            assert clips[0]["mark"] == "keep"
            assert clips[0]["mark_source"] == "auto"
        finally:
            goal_verifier.set_enabled(prev)


class TestHistoryMissingScores:
    """加载历史前的「要不要补跑 AI 复核」判断：只查记录 + 片段缓存，不碰 GPU。

    返回 (need_ai, n_missing, n_total)。缓存未命中（片段要重新生成）时分数
    必然全缺；读取异常一律按「不问」处理，宁可少问一次也不能挡加载。
    """

    def teardown_method(self, method):
        state.clip_cache.clear()

    def _setup(self, monkeypatch, tmp_path, goals, cached_clips):
        video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"x")
        monkeypatch.setattr(state, "load_history",
                            lambda: [{"video": video, "goals": goals}])
        state.clip_cache.clear()
        if cached_clips is not None:
            state.clip_cache[state.clip_cache_key(video, goals)] = cached_clips
        return video

    def test_cache_miss_needs_ai(self, monkeypatch, tmp_path):
        video = self._setup(monkeypatch, tmp_path, [10.0, 20.0], None)
        assert detection.history_missing_scores(video) == (True, 2, 2)

    def test_all_scored_no_need(self, monkeypatch, tmp_path):
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"x")
        video = self._setup(monkeypatch, tmp_path, [10.0, 20.0], [
            {"ts": 10.0, "path": str(clip), "idx": 0, "score": 0.9},
            {"ts": 20.0, "path": str(clip), "idx": 1, "score": 0.1}])
        assert detection.history_missing_scores(video) == (False, 0, 2)

    def test_partial_missing_needs_ai(self, monkeypatch, tmp_path):
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"x")
        video = self._setup(monkeypatch, tmp_path, [10.0, 20.0], [
            {"ts": 10.0, "path": str(clip), "idx": 0},
            {"ts": 20.0, "path": str(clip), "idx": 1, "score": 0.1}])
        assert detection.history_missing_scores(video) == (True, 1, 2)

    def test_missing_clip_file_needs_ai(self, monkeypatch, tmp_path):
        """片段文件被清理掉 → 加载时会全部重新生成，分数同样是全缺。"""
        video = self._setup(monkeypatch, tmp_path, [10.0], [
            {"ts": 10.0, "path": str(tmp_path / "gone.mp4"), "idx": 0,
             "score": 0.9}])
        assert detection.history_missing_scores(video) == (True, 1, 1)

    def test_unknown_record_no_need(self, monkeypatch):
        monkeypatch.setattr(state, "load_history", lambda: [])
        assert detection.history_missing_scores("C:/nope.mp4") == (False, 0, 0)
