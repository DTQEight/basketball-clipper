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

    def test_max_records_truncated(self, state_mod):
        """历史记录上限截断到 MAX_HISTORY_RECORDS。"""
        n = state_mod.MAX_HISTORY_RECORDS + 5
        for i in range(n):
            state_mod.add_history(f"/v{i}.mp4", (1, 2, 3, 4), [float(i)])
        assert len(state_mod.load_history()) == state_mod.MAX_HISTORY_RECORDS


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


class TestCancelEvent:
    def test_event_semantics(self, state_mod):
        state_mod.cancel_event.clear()
        assert not state_mod.cancel_event.is_set()
        state_mod.cancel_event.set()
        assert state_mod.cancel_event.is_set()
        state_mod.cancel_event.clear()
