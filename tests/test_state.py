"""state 模块持久化测试（历史记录 / 片段缓存，隔离到 tmp 目录）。"""
import builtins
import glob
import importlib
import os
import sys
from pathlib import Path

import pytest

from conftest import _ROOT  # noqa: F401  确保 sys.path 已注入（含缓存目录隔离）


@pytest.fixture
def state_mod(tmp_path, monkeypatch):
    """把 BBALL_CACHE_ROOT 指到临时目录并 reload state 模块（隔离文件副作用）。"""
    monkeypatch.setenv("BBALL_CACHE_ROOT", str(tmp_path))
    from services import state
    return importlib.reload(state)


class TestHistory:
    def test_roundtrip_and_order(self, state_mod):
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [1.0, 2.0], baseline_idx=7)
        state_mod.add_history("/b.mp4", (5, 6, 7, 8), [3.0], baseline_idx=9,
                              ball_conf=0.3, proc_fps=114.5)
        records = state_mod.load_history()
        assert len(records) == 2
        assert records[0]["video"] == "/b.mp4"      # 最新在前
        assert records[1]["goals"] == [1.0, 2.0]
        assert records[0]["proc_fps"] == 114.5
        assert records[0]["ball_conf"] == 0.3

    def test_same_video_overwrites(self, state_mod):
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [1.0])
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [1.0, 5.0])
        records = state_mod.load_history()
        assert len(records) == 1
        assert records[0]["total"] == 2

    def test_unknown_field_raises(self, state_mod):
        with pytest.raises(TypeError):
            state_mod.add_history("/a.mp4", (1, 2, 3, 4), [], typo_field=1)

    def test_update_history_marks(self, state_mod):
        """verify 自动标记回写历史：按 ts 键存，basename 兜底匹配。"""
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [1.0, 5.0])
        clips = [{"ts": 1.0, "score": 0.9, "mark": "keep", "mark_source": "auto"},
                 {"ts": 5.0, "score": 0.05, "mark": "reject", "mark_source": "auto"}]
        assert state_mod.update_history_marks("/a.mp4", clips)
        rec = state_mod.load_history()[0]
        assert rec["clip_marks"]["1.0"]["mark"] == "keep"
        assert rec["clip_marks"]["5.0"]["score"] == 0.05
        # 路径迁移兜底：按文件名匹配同一条记录，增量合并不丢已有键
        assert state_mod.update_history_marks("/moved/a.mp4",
                                              [{"ts": 1.0, "score": 0.91,
                                                "mark": "keep", "mark_source": "auto"}])
        rec = state_mod.load_history()[0]
        assert rec["clip_marks"]["1.0"]["score"] == 0.91
        assert "5.0" in rec["clip_marks"]
        assert not state_mod.update_history_marks("/nope.mp4", clips)
        assert not state_mod.update_history_marks("/a.mp4", [{"ts": 1.0}])

    def test_corrupt_file_backed_up_not_wiped(self, state_mod):
        """损坏的历史文件必须先备份 .corrupt-*.bak，而不是被静默覆盖清空。"""
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [1.0])
        hist_file = state_mod.HISTORY_FILE
        assert os.path.exists(hist_file)
        with open(hist_file, "w", encoding="utf-8") as f:
            f.write("{ corrupted json !!!")
        assert state_mod.load_history() == []
        assert not os.path.exists(hist_file)                      # 原文件已被改名移走
        baks = glob.glob(hist_file + ".corrupt-*.bak")
        assert len(baks) == 1                                     # 备份在，内容可追溯
        with open(baks[0], encoding="utf-8") as f:
            assert "corrupted json" in f.read()

    def test_io_error_raises_and_keeps_file(self, state_mod, monkeypatch):
        """IO 瞬态错误（如 Windows 共享冲突）不得当作损坏改名（旧实现会误清空完好历史）。"""
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [1.0])
        hist_file = os.path.abspath(state_mod.HISTORY_FILE)
        real_open = builtins.open

        def _deny(p, *a, **k):
            if os.path.abspath(str(p)) == hist_file:
                raise PermissionError("locked by writer")
            return real_open(p, *a, **k)

        monkeypatch.setattr(builtins, "open", _deny)
        with pytest.raises(OSError):
            state_mod.load_history()
        monkeypatch.undo()
        # 文件原封不动，内容完好
        assert os.path.exists(hist_file)
        assert state_mod.load_history()[0]["video"] == "/a.mp4"
        assert not glob.glob(hist_file + ".corrupt-*")

    def test_atomic_write_no_tmp_leftover(self, state_mod):
        """原子写不留 .tmp-* 残留文件。"""
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [1.0])
        state_mod.add_history("/b.mp4", (1, 2, 3, 4), [2.0])
        leftovers = [f for f in os.listdir(state_mod.CACHE_ROOT) if ".tmp-" in f]
        assert leftovers == []

    def test_add_history_write_failure_returns_none(self, state_mod, monkeypatch):
        """写入失败时 add_history 返回 None 且旧记录不被破坏。"""
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [1.0])
        monkeypatch.setattr(state_mod.os, "replace",
                            lambda *a, **k: (_ for _ in ()).throw(PermissionError("busy")))
        assert state_mod.add_history("/b.mp4", (1, 2, 3, 4), [2.0]) is None
        monkeypatch.undo()
        records = state_mod.load_history()
        assert len(records) == 1 and records[0]["video"] == "/a.mp4"

    def test_no_truncation_unlimited(self, state_mod):
        """历史记录不限量保留（MAX_HISTORY_RECORDS=None，全部落盘）。"""
        n = 60  # 超出旧上限 50，验证不再截断
        for i in range(n):
            state_mod.add_history(f"/v{i}.mp4", (1, 2, 3, 4), [float(i)])
        records = state_mod.load_history()
        assert len(records) == n
        # 最近的在最前
        assert records[0]["video"] == f"/v{n - 1}.mp4"


class TestClipCache:
    def test_key_order_insensitive(self, state_mod):
        """写入方（sorted）与历史回读方（原序）必须命中同一 key。"""
        k1 = state_mod.clip_cache_key("/v.mp4", [3.0, 1.0, 2.0])
        k2 = state_mod.clip_cache_key("/v.mp4", [1.0, 2.0, 3.0])
        assert k1 == k2

    def test_put_and_evict(self, state_mod):
        clips = [{"ts": 1.0, "path": "/x.mp4", "idx": 0}]
        for i in range(state_mod.CLIP_CACHE_MAX_ENTRIES + 2):
            state_mod.put_clip_cache(("v", (float(i),)), clips)
        assert len(state_mod.clip_cache) == state_mod.CLIP_CACHE_MAX_ENTRIES
        assert ("v", (0.0,)) not in state_mod.clip_cache      # 最旧被驱逐
        assert ("v", (float(state_mod.CLIP_CACHE_MAX_ENTRIES + 1),)) in state_mod.clip_cache
        assert os.path.exists(state_mod.CLIP_CACHE_FILE)      # 已落盘

    def test_roundtrip_key_consistency(self, state_mod, tmp_path):
        """save 侧 round(3) 归一后落盘，load 侧读回的 key 应与写入方一致。"""
        # load_clip_cache 会过滤片段文件不存在的条目，必须用真实文件
        clip_file = tmp_path / "goal_0_1s.mp4"
        clip_file.write_bytes(b"x")
        key = state_mod.clip_cache_key("/v.mp4", [1.23456, 2.0])
        state_mod.put_clip_cache(key, [{"ts": 1.23456, "path": str(clip_file), "idx": 0}])
        state_mod.clip_cache.clear()
        state_mod.init_clip_cache()
        assert key in state_mod.clip_cache

    def test_evict_deletes_disk_clip(self, state_mod, tmp_path):
        """驱逐条目时同步删除其磁盘片段文件（豁免集锦成品）。"""
        clip_file = tmp_path / "goal_0_5s.mp4"
        clip_file.write_bytes(b"x")
        hl_file = tmp_path / "game-highlights.mp4"
        hl_file.write_bytes(b"x")
        for i in range(state_mod.CLIP_CACHE_MAX_ENTRIES + 1):
            state_mod.put_clip_cache(("v", (float(i),)),
                                     [{"ts": 1.0, "path": str(clip_file), "idx": 0}])
        assert not clip_file.exists()                          # 驱逐后文件被删
        # 集锦成品不应被任何驱逐路径删除（写入一个再触发驱逐验证豁免）
        state_mod.put_clip_cache(("h", (9.0,)),
                                 [{"ts": 1.0, "path": str(hl_file), "idx": 0}])
        for i in range(100, 100 + state_mod.CLIP_CACHE_MAX_ENTRIES + 1):
            state_mod.put_clip_cache(("v2", (float(i),)),
                                     [{"ts": 1.0, "path": str(hl_file), "idx": 0}])
        assert hl_file.exists()

    def test_marks_roundtrip(self, state_mod, tmp_path):
        """verify 分数/标记随缓存落盘并可读回（重启后免重新验证）。"""
        clip_file = tmp_path / "goal_1_0s.mp4"
        clip_file.write_bytes(b"x")
        key = state_mod.clip_cache_key("/v.mp4", [1.0])
        state_mod.put_clip_cache(key, [{"ts": 1.0, "path": str(clip_file), "idx": 0,
                                        "score": 0.83, "mark": "keep",
                                        "mark_source": "auto"}])
        state_mod.clip_cache.clear()
        state_mod.init_clip_cache()
        c = state_mod.clip_cache[key][0]
        assert c["score"] == 0.83
        assert c["mark"] == "keep"
        assert c["mark_source"] == "auto"

    def test_update_clip_cache_marks(self, state_mod, tmp_path):
        """回写按 ts 匹配、就地更新并落盘；无匹配视频返回 False。"""
        clip_file = tmp_path / "goal_2_0s.mp4"
        clip_file.write_bytes(b"x")
        key = state_mod.clip_cache_key("/v.mp4", [2.0, 5.0])
        state_mod.put_clip_cache(key, [
            {"ts": 2.0, "path": str(clip_file), "idx": 0},
            {"ts": 5.0, "path": str(clip_file), "idx": 1},
        ])
        marked = [{"ts": 2.0, "score": 0.9, "mark": "keep", "mark_source": "auto"},
                  {"ts": 5.0, "score": 0.05, "mark": "reject", "mark_source": "auto"}]
        assert state_mod.update_clip_cache_marks("/v.mp4", marked)
        cached = state_mod.clip_cache[key]
        assert cached[0]["mark"] == "keep" and cached[1]["mark"] == "reject"
        # 落盘：清内存重读仍在
        state_mod.clip_cache.clear()
        state_mod.init_clip_cache()
        assert state_mod.clip_cache[key][1]["score"] == 0.05
        assert not state_mod.update_clip_cache_marks("/other.mp4", marked)


class TestTaskLock:
    def test_acquire_release_with_token(self, state_mod):
        """token 机制：acquire 返回正整数；release 仅对匹配 token 生效。"""
        t1 = state_mod.try_acquire_task('detect')
        assert t1 > 0
        assert state_mod.current_task() == 'detect'
        assert state_mod.try_acquire_task('batch') == 0       # 互斥
        state_mod.release_task(t1 + 999)                       # 错误 token 不释放
        assert state_mod.current_task() == 'detect'
        state_mod.release_task(t1)
        assert state_mod.current_task() is None
        t2 = state_mod.try_acquire_task('batch')
        assert t2 > 0 and t2 != t1                             # token 单调递增
        state_mod.release_task()                               # 无 token 兼容旧语义
        assert state_mod.current_task() is None

    def test_unicode_decode_error_backed_up(self, state_mod):
        """GBK 编码的历史文件按损坏备份处理（不冒泡成检测失败）。"""
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [1.0])
        hist_file = state_mod.HISTORY_FILE
        with open(hist_file, "wb") as f:
            f.write('{"视频": "篮球"}'.encode('gbk'))          # 非 UTF-8 字节
        assert state_mod.load_history() == []
        assert glob.glob(hist_file + ".corrupt-*.bak")


class TestCancelEvent:
    def test_event_semantics(self, state_mod):
        state_mod.cancel_event.clear()
        assert not state_mod.cancel_event.is_set()
        state_mod.cancel_event.set()
        assert state_mod.cancel_event.is_set()
        state_mod.cancel_event.clear()


class TestCheckpoint:
    """检测断点（断点续识别）持久化测试。"""

    _PARAMS = {"start": 0, "end": 9000, "fps": 30.0,
               "hoop": [100, 200, 300, 400], "baseline_idx": 7,
               "ball_conf": 0.3, "min_gap_sec": 3.0,
               "diff_threshold": 15, "auto_threshold": True,
               "yolo_step": 2, "skip_yolo_no_motion": False,
               "min_circularity": 0.35, "min_in_hoop_frames": 2,
               "min_blob_area": 30, "search_margin": 80}

    def test_roundtrip_same_params(self, state_mod):
        state_mod.save_checkpoint("/a.mp4", 3000, [12.5, 40.0], self._PARAMS)
        cp = state_mod.load_checkpoint("/a.mp4", dict(self._PARAMS))
        assert cp is not None
        assert cp["frame"] == 3000
        assert cp["goals"] == [12.5, 40.0]

    def test_saved_extra_keys_do_not_break_match(self, state_mod):
        """写盘附带 auto_threshold_value 等扩展字段，比对只看调用方参数。"""
        params = {**self._PARAMS, "auto_threshold_value": 17}
        state_mod.save_checkpoint("/a.mp4", 100, [], params)
        assert state_mod.load_checkpoint("/a.mp4", dict(self._PARAMS)) is not None

    def test_param_mismatch_rejected(self, state_mod):
        state_mod.save_checkpoint("/a.mp4", 3000, [], self._PARAMS)
        changed = dict(self._PARAMS)
        changed["ball_conf"] = 0.5
        assert state_mod.load_checkpoint("/a.mp4", changed) is None
        changed2 = dict(self._PARAMS)
        changed2["hoop"] = [100, 200, 300, 401]
        assert state_mod.load_checkpoint("/a.mp4", changed2) is None

    def test_zero_frame_rejected(self, state_mod):
        state_mod.save_checkpoint("/a.mp4", 0, [], self._PARAMS)
        assert state_mod.load_checkpoint("/a.mp4", dict(self._PARAMS)) is None

    def test_other_video_no_checkpoint(self, state_mod):
        state_mod.save_checkpoint("/a.mp4", 100, [], self._PARAMS)
        assert state_mod.load_checkpoint("/b.mp4", dict(self._PARAMS)) is None
        assert not state_mod.has_checkpoint("/b.mp4")
        assert state_mod.has_checkpoint("/a.mp4")

    def test_clear_single_keeps_others(self, state_mod):
        state_mod.save_checkpoint("/a.mp4", 100, [], self._PARAMS)
        state_mod.save_checkpoint("/b.mp4", 200, [], self._PARAMS)
        state_mod.clear_checkpoint("/a.mp4")
        assert not state_mod.has_checkpoint("/a.mp4")
        assert state_mod.has_checkpoint("/b.mp4")
        state_mod.clear_checkpoint()                     # 全清
        assert not state_mod.has_checkpoint("/b.mp4")

    def test_corrupt_file_safe(self, state_mod):
        """断点文件损坏：读/判都安全返回空，写盘可自愈覆盖。"""
        with open(state_mod.CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            f.write("{ corrupted !!!")
        assert state_mod.load_checkpoint("/a.mp4", self._PARAMS) is None
        assert not state_mod.has_checkpoint("/a.mp4")
        state_mod.save_checkpoint("/a.mp4", 50, [], self._PARAMS)
        assert state_mod.load_checkpoint("/a.mp4", dict(self._PARAMS))["frame"] == 50
