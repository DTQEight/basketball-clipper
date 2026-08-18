"""state 模块持久化测试（历史记录 / 片段缓存，隔离到 tmp 目录）。"""
import glob
import importlib
import os
import sys
from pathlib import Path

import pytest

from conftest import _ROOT  # noqa: F401  确保 sys.path 已注入


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


class TestCancelEvent:
    def test_event_semantics(self, state_mod):
        state_mod.cancel_event.clear()
        assert not state_mod.cancel_event.is_set()
        state_mod.cancel_event.set()
        assert state_mod.cancel_event.is_set()
        state_mod.cancel_event.clear()
