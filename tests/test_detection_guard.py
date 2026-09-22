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


class TestPartialDecodeGuard:
    """N1: 提前 EOF（只解出部分帧）不得当成"检测完成"。

    旧实现只查 processed==0：一旦解出过任何一帧就按成功收尾，写历史 + 删掉该
    视频全部断点 → 后半段进球永久缺失且无法续跑。判定 partial 时必须保留断点。
    """

    def test_partial_when_frame_no_stops_before_end(self):
        """主循环停止时帧号未推进到区间末尾 → 提前 EOF。"""
        assert detection._is_partial_decode(300, 900, False, 300) is True

    def test_complete_when_frame_no_reaches_end(self):
        """正常跑完（帧号到 end）不得误报——**即使解码计数远小于帧号区间**。

        实测 Y:\\...\\2026.08.31-2nd.mp4：容器时间基异常，解码 13403 帧而帧号跑到
        3.6 万。若按 processed < (end-start) 判定就会恒判"不完整"（连带不删断点）。
        """
        assert detection._is_partial_decode(900, 900, False, 134) is False

    def test_cancel_is_not_partial(self):
        """用户取消走取消分支（保存断点 + 丢弃结果），不是"解码不完整"。"""
        assert detection._is_partial_decode(300, 900, True, 300) is False

    def test_tail_boundary_within_tolerance(self):
        """末尾几帧的 pts 取整偏差不算截断（实测 43554 / 43556 → 不是截断）。"""
        assert detection._is_partial_decode(43554, 43556, False, 43554) is False
        # 差 10 帧仍在容差内（_PARTIAL_TOL_FRAMES）→ 不判截断
        assert detection._is_partial_decode(43546, 43556, False, 43546) is False

    def test_large_gap_is_partial(self):
        """缺口远超容差 → 真截断。"""
        assert detection._is_partial_decode(13000, 36203, False, 13000) is True

    def test_zero_frames_not_partial(self):
        """0 帧由 decode-fail 分支负责（那条不写历史），此处不重复判。"""
        assert detection._is_partial_decode(0, 900, False, 0) is False


class TestPersistMarksScope:
    """R1: _persist_marks 必须把 clips 的 ts 作为人工标签「覆盖范围」带给 state。

    漏掉 manual_scope 就退回全量替换：clips 外的旧 √/× 被当成"无标记"整批抹掉
    （预览切片失败、片段被 7 天清理后只重建一部分，或**重检测后 clips 未回填
    人工标记**——单视频 run_detect 路径就不回填）。用户上一轮标注静默消失。
    """

    def test_manual_scope_is_passed(self, monkeypatch):
        captured = {}

        def _fake(vp, **kw):
            captured.update(kw)
            return True

        monkeypatch.setattr(state, "update_history_labels", _fake)
        clips = [{"ts": 10.0, "path": "a.mp4", "idx": 0, "mark": "keep",
                  "mark_source": "manual"},
                 {"ts": 20.0, "path": "b.mp4", "idx": 1}]
        detection._persist_marks("/v.mp4", clips)
        assert captured["manual_scope"] == [10.0, 20.0]   # 范围=本次看得见的片段
        assert captured["kept_ts_list"] == [10.0]

    def test_write_manual_false_has_no_scope(self, monkeypatch):
        """重检测路径（write_manual=False）不碰人工标签，也不该传覆盖范围。"""
        captured = {}

        def _fake(vp, **kw):
            captured.update(kw)
            return True

        monkeypatch.setattr(state, "update_history_labels", _fake)
        detection._persist_marks(
            "/v.mp4",
            [{"ts": 10.0, "path": "a.mp4", "mark": "keep", "mark_source": "auto"}],
            write_manual=False)
        assert captured["manual_scope"] is None
        assert captured["kept_ts_list"] is None


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


class TestModelFingerprint:
    """口径指纹必须覆盖「会改变某臂分数」的全部输入。

    漏掉任何一个，换模型/换权重后缓存里的旧 score 不会失效，新旧口径的分数
    会混在一起出新阈值下的 √/×（静默错判）。2026.09.21 补上 A 臂模型与球检测权重。
    """

    @staticmethod
    def _isolate(monkeypatch, tmp_path, lgbm=None, weights_dir=None):
        """把指纹的输入收窄到临时路径，避免依赖真实权重与标定文件。"""
        monkeypatch.setattr(goal_verifier, "LGBM_MODEL", lgbm or (tmp_path / "no_lgbm.txt"))
        monkeypatch.setattr(goal_verifier, "BALL_WEIGHTS_DIR",
                            weights_dir or (tmp_path / "no_weights"))
        monkeypatch.setattr(goal_verifier, "_read_ensemble", lambda: 0.7)

    def test_stable_across_calls(self, monkeypatch, tmp_path):
        """同一批文件连算两次必须一致（否则分数会被反复误判为过期）。"""
        self._isolate(monkeypatch, tmp_path)
        assert goal_verifier.model_fingerprint() == goal_verifier.model_fingerprint()

    def test_a_arm_model_change_invalidates(self, monkeypatch, tmp_path):
        """重训 A 臂（换 model_lgbm.txt）→ 指纹必须变。"""
        p = tmp_path / "model_lgbm.txt"
        p.write_text("tree", encoding="utf-8")
        self._isolate(monkeypatch, tmp_path, lgbm=p)
        fp1 = goal_verifier.model_fingerprint()
        os.utime(p, ns=(10 ** 18, 10 ** 18))       # 模拟重训后落盘
        assert goal_verifier.model_fingerprint() != fp1

    def test_ball_weights_change_invalidates(self, monkeypatch, tmp_path):
        """换球检测权重 → A 臂特征分布变 → 指纹必须变。"""
        wd = tmp_path / "weights"
        wd.mkdir()
        w = wd / "basketball_ft.pt"
        w.write_bytes(b"0")
        self._isolate(monkeypatch, tmp_path, weights_dir=wd)
        fp1 = goal_verifier.model_fingerprint()
        os.utime(w, ns=(10 ** 18, 10 ** 18))
        assert goal_verifier.model_fingerprint() != fp1

    def test_missing_files_do_not_raise(self, monkeypatch, tmp_path):
        """权重/模型缺失（如 main 线上无 training/）不能抛异常，只降级哈希。"""
        self._isolate(monkeypatch, tmp_path)
        assert isinstance(goal_verifier.model_fingerprint(), str)
