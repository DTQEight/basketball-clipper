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
        hist_file = state_mod._history_file_for("/a.mp4")
        assert os.path.exists(hist_file)
        with open(hist_file, "w", encoding="utf-8") as f:
            f.write("{ corrupted json !!!")
        assert state_mod.load_history() == []
        assert not os.path.exists(hist_file)                      # 原文件已被改名移走
        baks = glob.glob(hist_file + ".corrupt-*.bak")
        assert len(baks) == 1                                     # 备份在，内容可追溯
        with open(baks[0], encoding="utf-8") as f:
            assert "corrupted json" in f.read()

    def test_io_error_skips_locked_file_not_wipe(self, state_mod, monkeypatch):
        """IO 瞬态错误（如 Windows 共享冲突）应跳过该文件，不影响其他视频的记录。"""
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [1.0])
        state_mod.add_history("/b.mp4", (1, 2, 3, 4), [2.0])
        locked_file = os.path.abspath(state_mod._history_file_for("/a.mp4"))
        real_open = builtins.open

        def _deny(p, *a, **k):
            if os.path.abspath(str(p)) == locked_file:
                raise PermissionError("locked by writer")
            return real_open(p, *a, **k)

        monkeypatch.setattr(builtins, "open", _deny)
        records = state_mod.load_history()
        monkeypatch.undo()
        # /a.mp4 被锁定跳过，/b.mp4 正常读取
        assert len(records) == 1
        assert records[0]["video"] == "/b.mp4"
        assert os.path.exists(locked_file)                          # 锁定文件原封不动
        assert not glob.glob(locked_file + ".corrupt-*")

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

    def test_no_max_records_limit(self, state_mod):
        """历史记录不再设上限：每个视频独立文件，可无限累积。"""
        n = 100  # 远超旧版 50 条上限
        for i in range(n):
            state_mod.add_history(f"/v{i}.mp4", (1, 2, 3, 4), [float(i)])
        assert len(state_mod.load_history()) == n

    def test_save_history_keeps_unlisted_files(self, state_mod):
        """H4: save_history 不得删除"不在传入列表中"的历史文件。

        旧实现的清理循环唯一实际触发场景是 load_history 因瞬态 IO 错误
        （Windows 共享冲突）跳过了某文件——此时调用方传来的列表缺这条
        记录，清理会把它永久删除（数据丢失）。
        """
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [1.0])
        state_mod.add_history("/b.mp4", (1, 2, 3, 4), [2.0])
        a_file = state_mod._history_file_for("/a.mp4")
        b_file = state_mod._history_file_for("/b.mp4")
        assert os.path.exists(a_file) and os.path.exists(b_file)
        # 模拟 /a.mp4 因 IO 错误没被读进列表：只带着 /b.mp4 的记录保存
        records = state_mod.load_history()
        records = [r for r in records if r.get("video") == "/b.mp4"]
        assert state_mod.save_history(records)
        assert os.path.exists(a_file)                     # 未列入的文件必须原封不动
        assert os.path.exists(b_file)
        assert len(state_mod.load_history()) == 2

    def test_update_history_labels_preserves_detect_time(self, state_mod):
        """L4: 打标记不得把记录的"检测时间"覆盖成"标记时间"。

        旧实现覆盖 time/timestamp，刚标记过的旧记录会浮到历史列表顶部，
        用户看到的"检测时间"实际是标记时间。
        """
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [10.0, 20.0])
        rec = state_mod.load_history()[0]
        old_time, old_ts = rec["time"], rec["timestamp"]
        assert state_mod.update_history_labels("/a.mp4", kept_ts_list=[10.0],
                                               deleted_ts_list=[20.0])
        rec2 = state_mod.load_history()[0]
        assert rec2["time"] == old_time                    # 检测时间不动
        assert rec2["timestamp"] == old_ts
        assert rec2["labels"]["label_time"]                # 标记时间单独记录
        assert rec2["labels"]["kept"] == [10.0]
        assert rec2["labels"]["deleted"] == [20.0]

    def test_add_history_remaps_labels_within_tolerance(self, state_mod):
        """M3: 改参数重跑后进球时间戳小幅偏移，旧标签按 ±0.5s 容差重映射到新 ts。

        直接沿用旧 ts 会在加载历史时精确匹配失败 → 偏移的球全部掉出默认集锦。
        """
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [10.0, 20.0, 30.0])
        assert state_mod.update_history_labels(
            "/a.mp4", kept_ts_list=[10.0, 30.0], deleted_ts_list=[20.0])
        # 重跑：进球时间戳整体偏移 +0.2s / +0.3s（在容差内）
        rec = state_mod.add_history("/a.mp4", (1, 2, 3, 4), [10.2, 20.3, 30.25])
        assert rec is not None
        lab = rec["labels"]
        assert lab["kept"] == [10.2, 30.25]                # 重映射到新 ts
        assert lab["deleted"] == [20.3]

    def test_add_history_discards_labels_on_low_match(self, state_mod):
        """M3: 新旧进球几乎对不上（匹配率 <50%）时丢弃旧标签。

        两次检测结果差异过大说明旧标签对新结果已无意义，保留只会误导。
        """
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [10.0, 20.0])
        assert state_mod.update_history_labels(
            "/a.mp4", kept_ts_list=[10.0], deleted_ts_list=[20.0])
        # 重跑结果完全不同（时间戳偏移远超容差）
        rec = state_mod.add_history("/a.mp4", (1, 2, 3, 4), [100.0, 200.0])
        assert rec is not None
        assert "labels" not in rec                         # 匹配率 0%，丢弃

    def test_add_history_partial_label_match_keeps_matched(self, state_mod):
        """M3: 部分匹配（≥50%）时保留已匹配的标签、丢弃未匹配的。"""
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [10.0, 20.0])
        assert state_mod.update_history_labels(
            "/a.mp4", kept_ts_list=[10.0, 20.0], deleted_ts_list=None)
        # 重跑：只有 10.0 附近的球还在（10.2），20.0 附近没了
        rec = state_mod.add_history("/a.mp4", (1, 2, 3, 4), [10.2, 200.0])
        assert rec is not None
        assert rec["labels"]["kept"] == [10.2]             # 匹配 1/2 = 50%，保留匹配项


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
        hist_file = state_mod._history_file_for("/a.mp4")
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
