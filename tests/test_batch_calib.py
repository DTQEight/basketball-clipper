"""批量标定落盘缓存（state 持久化 + detection 回填优先级）。

背景：批量面板的「保存标定」原本只写内存 state.batch_calibs，关掉应用就丢；
只有跑过检测的视频能靠历史记录回填，没跑过的只能重新逐个框。
本文件验证新增的 cache/batch_calibs.json 落盘，以及
「本次会话内存 > 落盘缓存 > 历史记录」三级回填优先级。
"""
import importlib
import json
import os

import pytest

from conftest import _ROOT  # noqa: F401  确保 sys.path 已注入（含缓存目录隔离）


@pytest.fixture
def state_mod(tmp_path, monkeypatch):
    """把 BBALL_CACHE_ROOT 指到临时目录并 reload state（隔离文件副作用）。"""
    monkeypatch.setenv("BBALL_CACHE_ROOT", str(tmp_path))
    from services import state
    return importlib.reload(state)


@pytest.fixture
def det():
    """detection 用 `from . import state` 持模块引用，state reload 后自动生效。"""
    from services import detection
    return detection


class TestBatchCalibPersist:
    def test_roundtrip(self, state_mod):
        assert state_mod.upsert_batch_calib("/v/a.mp4", (1, 2, 3, 4), 7) is True
        got = state_mod.load_batch_calibs()
        assert got["/v/a.mp4"] == {"hoop": (1, 2, 3, 4), "baseline_idx": 7}

    def test_upsert_keeps_other_videos(self, state_mod):
        state_mod.upsert_batch_calib("/v/a.mp4", (1, 2, 3, 4), 7)
        state_mod.upsert_batch_calib("/v/b.mp4", (5, 6, 7, 8), 9)
        assert set(state_mod.load_batch_calibs()) == {"/v/a.mp4", "/v/b.mp4"}

    def test_upsert_overwrites_same_video(self, state_mod):
        state_mod.upsert_batch_calib("/v/a.mp4", (1, 2, 3, 4), 7)
        state_mod.upsert_batch_calib("/v/a.mp4", (9, 9, 9, 9), 11)
        assert state_mod.load_batch_calibs()["/v/a.mp4"] == {
            "hoop": (9, 9, 9, 9), "baseline_idx": 11}

    def test_bad_hoop_not_written(self, state_mod):
        # 框不完整（只有 3 个数）不该落盘，否则下次回填出个坏标定
        assert state_mod.upsert_batch_calib("/v/a.mp4", (1, 2, 3), 7) is False
        assert state_mod.load_batch_calibs() == {}

    def test_missing_file_returns_empty(self, state_mod):
        assert state_mod.load_batch_calibs() == {}

    def test_corrupt_file_ignored(self, state_mod):
        os.makedirs(os.path.dirname(state_mod.BATCH_CALIB_FILE), exist_ok=True)
        with open(state_mod.BATCH_CALIB_FILE, "w", encoding="utf-8") as f:
            f.write("{ this is not json")
        assert state_mod.load_batch_calibs() == {}

    def test_corrupt_entry_skipped(self, state_mod):
        # 手改坏 / 旧版本残留：坏条目跳过，好条目仍要能读出来
        os.makedirs(os.path.dirname(state_mod.BATCH_CALIB_FILE), exist_ok=True)
        with open(state_mod.BATCH_CALIB_FILE, "w", encoding="utf-8") as f:
            json.dump({"good.mp4": {"hoop": [1, 2, 3, 4], "baseline_idx": 5},
                       "short.mp4": {"hoop": [1, 2]},
                       "notdict.mp4": "oops"}, f)
        assert set(state_mod.load_batch_calibs()) == {"good.mp4"}

    def test_hoop_normalized_to_int(self, state_mod):
        # 界面传来的坐标可能是 float/字符串，落盘后统一成 int 元组
        state_mod.upsert_batch_calib("/v/a.mp4", [1.0, 2.0, 3.0, 4.0], 7)
        got = state_mod.load_batch_calibs()["/v/a.mp4"]["hoop"]
        assert got == (1, 2, 3, 4)


class TestBackfillPriority:
    def test_from_disk_cache(self, state_mod, det):
        state_mod.batch_files = ["/v/a.mp4", "/v/b.mp4"]
        state_mod.batch_calibs = {}
        state_mod.upsert_batch_calib("/v/a.mp4", (1, 2, 3, 4), 7)
        assert det.backfill_batch_calibs() == 1
        assert state_mod.batch_calibs["/v/a.mp4"] == {"hoop": (1, 2, 3, 4), "baseline_idx": 7}
        assert "/v/b.mp4" not in state_mod.batch_calibs

    def test_history_fallback(self, state_mod, det):
        state_mod.add_history("/v/a.mp4", (9, 8, 7, 6), [1.0], baseline_idx=111)
        state_mod.batch_files = ["/v/a.mp4"]
        state_mod.batch_calibs = {}
        assert det.backfill_batch_calibs() == 1
        assert state_mod.batch_calibs["/v/a.mp4"] == {"hoop": (9, 8, 7, 6), "baseline_idx": 111}

    def test_disk_cache_wins_over_history(self, state_mod, det):
        # 同一视频两边都有：以落盘缓存为准（用户手动框的比历史里的旧值新）
        state_mod.add_history("/v/a.mp4", (9, 9, 9, 9), [1.0], baseline_idx=111)
        state_mod.upsert_batch_calib("/v/a.mp4", (1, 2, 3, 4), 7)
        state_mod.batch_files = ["/v/a.mp4"]
        state_mod.batch_calibs = {}
        assert det.backfill_batch_calibs() == 1
        assert state_mod.batch_calibs["/v/a.mp4"]["baseline_idx"] == 7

    def test_session_calib_not_overwritten(self, state_mod, det):
        # 本次会话刚框的（还没落盘的旧行为）不该被回填覆盖
        state_mod.upsert_batch_calib("/v/a.mp4", (1, 2, 3, 4), 7)
        state_mod.batch_files = ["/v/a.mp4"]
        state_mod.batch_calibs = {"/v/a.mp4": {"hoop": (5, 5, 5, 5), "baseline_idx": 3}}
        assert det.backfill_batch_calibs() == 0
        assert state_mod.batch_calibs["/v/a.mp4"]["hoop"] == (5, 5, 5, 5)

    def test_mixed_sources_counted(self, state_mod, det):
        state_mod.add_history("/v/b.mp4", (4, 4, 4, 4), [1.0], baseline_idx=2)
        state_mod.upsert_batch_calib("/v/a.mp4", (1, 2, 3, 4), 7)
        state_mod.batch_files = ["/v/a.mp4", "/v/b.mp4", "/v/c.mp4"]
        state_mod.batch_calibs = {}
        assert det.backfill_batch_calibs() == 2

    def test_renamed_video_skipped_without_error(self, state_mod, det):
        """视频被移动/改名：缓存里有旧路径，本次扫描没有它 → 静默跳过。"""
        state_mod.upsert_batch_calib("/v/old.mp4", (1, 2, 3, 4), 7)
        state_mod.batch_files = ["/v/new.mp4"]
        state_mod.batch_calibs = {}
        assert det.backfill_batch_calibs() == 0
        assert state_mod.batch_calibs == {}

    def test_no_batch_files_returns_zero(self, state_mod, det):
        state_mod.upsert_batch_calib("/v/a.mp4", (1, 2, 3, 4), 7)
        state_mod.batch_files = []
        state_mod.batch_calibs = {}
        assert det.backfill_batch_calibs() == 0

    def test_history_record_without_hoop_skipped(self, state_mod, det):
        state_mod.add_history("/v/a.mp4", None, [1.0], baseline_idx=2)
        state_mod.batch_files = ["/v/a.mp4"]
        state_mod.batch_calibs = {}
        assert det.backfill_batch_calibs() == 0


class TestCalibFromCheckpoint:
    """断点回填（第三来源）：检测被打断的视频 —— 没写历史、也没点过「保存标定」，
    断点 params 里的 hoop/baseline_idx 是唯一还留着的标定。"""

    def _save_cp(self, state_mod, vp, hoop, baseline_idx, cur=10):
        assert state_mod.save_checkpoint(
            vp, {"hoop": list(hoop), "baseline_idx": baseline_idx,
                 "fps": 30.0, "start_frame": 0, "end_frame": 100,
                 "ball_conf": 0.2, "min_gap_sec": 2.0},
            cur, {}) is True

    def test_recover_from_checkpoint(self, state_mod, det):
        self._save_cp(state_mod, "/v/a.mp4", (11, 22, 33, 44), 55)
        assert det.recover_calib_from_checkpoint("/v/a.mp4") == {
            "hoop": (11, 22, 33, 44), "baseline_idx": 55}

    def test_no_checkpoint_returns_none(self, state_mod, det):
        assert det.recover_calib_from_checkpoint("/v/a.mp4") is None

    def test_short_hoop_returns_none(self, state_mod, det):
        self._save_cp(state_mod, "/v/a.mp4", (11, 22, 33), 55)  # 只有 3 个数
        assert det.recover_calib_from_checkpoint("/v/a.mp4") is None

    def test_non_numeric_hoop_returns_none(self, state_mod, det):
        self._save_cp(state_mod, "/v/a.mp4", ("a", "b", "c", "d"), 55)
        assert det.recover_calib_from_checkpoint("/v/a.mp4") is None

    def test_backfill_uses_checkpoint(self, state_mod, det):
        self._save_cp(state_mod, "/v/a.mp4", (1, 2, 3, 4), 7)
        state_mod.batch_files = ["/v/a.mp4"]
        state_mod.batch_calibs = {}
        assert det.backfill_batch_calibs() == 1
        assert state_mod.batch_calibs["/v/a.mp4"] == {"hoop": (1, 2, 3, 4), "baseline_idx": 7}

    def test_history_wins_over_checkpoint(self, state_mod, det):
        # 历史与断点都有：以历史为准（跑完过的记录比中断时的断点更权威）
        self._save_cp(state_mod, "/v/a.mp4", (1, 2, 3, 4), 7)
        state_mod.add_history("/v/a.mp4", (9, 8, 7, 6), [1.0], baseline_idx=111)
        state_mod.batch_files = ["/v/a.mp4"]
        state_mod.batch_calibs = {}
        assert det.backfill_batch_calibs() == 1
        assert state_mod.batch_calibs["/v/a.mp4"]["hoop"] == (9, 8, 7, 6)

    def test_run_detect_fills_calib_from_checkpoint(self, state_mod, det, monkeypatch):
        """run_detect 入口：有断点无标定 → 不再拦「请先点击画面框住篮筐」。"""
        state_mod.video_state.update(path="/v/a.mp4", fps=30.0, total=1000)
        state_mod.calib["hoop"] = None
        state_mod.calib["baseline_frame"] = None
        state_mod.calib["baseline_idx"] = 0
        self._save_cp(state_mod, "/v/a.mp4", (11, 22, 33, 44), 55)
        monkeypatch.setattr(det, "read_frame", lambda *a, **k: object())
        monkeypatch.setattr(det, "get_device", lambda: "cpu")  # 在标定检查之后短路
        text, ok = det.run_detect(0, 100, 0.2, 2.0)
        assert ok is False
        assert "请先点击画面" not in text
        assert state_mod.calib["hoop"] == (11, 22, 33, 44)
        assert state_mod.calib["baseline_idx"] == 55

    def test_run_detect_still_requires_calib_without_checkpoint(self, state_mod, det):
        state_mod.video_state.update(path="/v/a.mp4", fps=30.0, total=1000)
        state_mod.calib["hoop"] = None
        state_mod.calib["baseline_frame"] = None
        text, ok = det.run_detect(0, 100, 0.2, 2.0)
        assert ok is False
        assert "请先点击画面" in text
