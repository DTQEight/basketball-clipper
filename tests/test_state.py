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

    def test_person_labels_roundtrip(self, state_mod):
        """人物分类：update_history_labels 增量写入 → get_labels 读回（含清除）。"""
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [10.0, 20.0])
        assert state_mod.update_history_labels(
            "/a.mp4", kept_ts_list=None, deleted_ts_list=None,
            person_map={10.0: "小明", 20.0: "小红"})
        labels = state_mod.get_labels("/a.mp4")
        assert labels["persons"] == {10.0: "小明", 20.0: "小红"}
        # 清除一个人的分类，另一个人不受影响
        assert state_mod.update_history_labels(
            "/a.mp4", kept_ts_list=None, deleted_ts_list=None,
            person_map={10.0: ""})
        labels = state_mod.get_labels("/a.mp4")
        assert labels["persons"] == {20.0: "小红"}
        # √/× 标记更新不清人物分类（person_map=None 路径）
        assert state_mod.update_history_labels(
            "/a.mp4", kept_ts_list=[10.0], deleted_ts_list=None)
        assert state_mod.get_labels("/a.mp4")["persons"] == {20.0: "小红"}

    def test_persons_remap_on_redetect(self, state_mod):
        """改参数重跑后人物分类随 kept/deleted 一起按 ±0.5s 容差重映射。"""
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [10.0, 20.0])
        assert state_mod.update_history_labels(
            "/a.mp4", kept_ts_list=None, deleted_ts_list=None,
            person_map={10.0: "小明", 20.0: "小红"})
        rec = state_mod.add_history("/a.mp4", (1, 2, 3, 4), [10.2, 20.3])
        assert rec is not None
        # 存盘键为 str(round(ts,3))，重映射后指向新 ts
        assert rec["labels"]["persons"] == {"10.2": "小明", "20.3": "小红"}

    def test_export_goals_person_filter(self):
        """按人物导出：只导该人物片段，叠加 √/× 规则；无匹配返回空。"""
        from services import detection
        clips = [
            {"ts": 1.0, "person": "A"},
            {"ts": 2.0, "person": "B", "mark": "keep"},
            {"ts": 3.0, "person": "A", "mark": "reject"},
            {"ts": 4.0},                                   # 未分类
        ]
        # A 的池：ts1（无标记）+ ts3（×）→ × 排除，只导未标记的
        assert detection._export_goals(clips, [1.0, 2.0, 3.0, 4.0], "A") == [1.0]
        # B 的池有 √ → 只导 √
        assert detection._export_goals(clips, [1.0, 2.0, 3.0, 4.0], "B") == [2.0]
        # 无此人 → 空
        assert detection._export_goals(clips, [1.0, 2.0], "C") == []
        # 不筛人物 → 老规则：有 √ 只导 √
        assert detection._export_goals(clips, [1.0, 2.0, 3.0, 4.0]) == [2.0]
        # 仅已分类（哨兵）：任意已分类片段的池（A+B）→ 有 √ 只导 √
        assert detection._export_goals(
            clips, [1.0, 2.0, 3.0, 4.0],
            detection.PERSON_FILTER_CLASSIFIED) == [2.0]
        # 仅已分类、无任何标记 → 导出全部已分类的（未分类 ts4 不含）
        clips2 = [{"ts": 1.0, "person": "A"}, {"ts": 2.0, "person": "B"}, {"ts": 4.0}]
        assert detection._export_goals(
            clips2, [1.0, 2.0, 4.0],
            detection.PERSON_FILTER_CLASSIFIED) == [1.0, 2.0]
        # 没有任何已分类片段 → 空
        assert detection._export_goals(
            [{"ts": 1.0}], [1.0], detection.PERSON_FILTER_CLASSIFIED) == []

    def test_game_highlights_path_naming(self):
        """整场导出文件名：{文件夹名}-[人物]-highlights.mp4，与单节导出互不覆盖。"""
        from services import detection
        videos = [r"E:\games\2026.09.04\1st.mp4", r"E:\games\2026.09.04\2nd.mp4"]
        # 不筛人物 → {文件夹名}-highlights.mp4
        p = detection._game_highlights_path(videos, None)
        assert os.path.basename(p) == "2026.09.04-highlights.mp4"
        # 人物 → {文件夹名}-{人物}-highlights.mp4
        p = detection._game_highlights_path(videos, "小明")
        assert os.path.basename(p) == "2026.09.04-小明-highlights.mp4"
        # 哨兵「仅已分类」
        p = detection._game_highlights_path(videos, detection.PERSON_FILTER_CLASSIFIED)
        assert os.path.basename(p) == "2026.09.04-已分类-highlights.mp4"
        # 与单节导出不互相覆盖
        single = detection._person_highlights_path(videos[0], "小明")
        assert p != single and single != detection._game_highlights_path(videos, "小明")
        # 无公共目录（跨盘符）→ 回落首视频名-game
        p = detection._game_highlights_path([r"C:\a\1st.mp4", r"D:\b\2nd.mp4"], "小明")
        assert os.path.basename(p) == "1st-game-小明-highlights.mp4"

    def test_clips_from_record_rebuild(self):
        """从历史记录轻量重建片段视图：mark/person 不需要预览文件。"""
        from services import detection
        # goals 全精度 vs 标签键 round(3)：必须舍入匹配（31.2789 ↔ "31.279"）
        r = {"goals": [10.00008, 20.00004, 30.0],
             "labels": {"kept": [10.0], "deleted": [20.0],
                        "persons": {"30.0": "小明"}}}
        clips = detection._clips_from_record(r)
        assert [c["ts"] for c in clips] == [10.0, 20.0, 30.0]
        assert clips[0]["mark"] == "keep"
        assert clips[1]["mark"] == "reject"
        assert clips[2]["mark"] is None
        assert clips[2]["person"] == "小明"
        # 空 labels → 全部无标记
        clips = detection._clips_from_record({"goals": [5.0]})
        assert clips == [{"ts": 5.0, "mark": None, "person": None}]

    def test_fullgame_collects_sources_across_videos(self, state_mod, monkeypatch, tmp_path):
        """整场导出：跨视频收集同一人物片段，按文件名顺序传给剪辑器。"""
        from services import detection
        monkeypatch.setattr(detection, "state", state_mod)
        # 三个"视频"：1st 有小明+小红，2nd 只有小明，3rd 无记录（跳过）
        base = str(tmp_path / "2026.09.04")
        v1, v2, v3 = base + r"\1st.mp4", base + r"\2nd.mp4", base + r"\3rd.mp4"
        state_mod.add_history(v1, (1, 2, 3, 4), [10.0, 20.0])
        state_mod.update_history_labels(v1, None, None,
                                        person_map={10.0: "小明", 20.0: "小红"})
        state_mod.add_history(v2, (1, 2, 3, 4), [15.0])
        state_mod.update_history_labels(v2, None, None, person_map={15.0: "小明"})
        state_mod.batch_files = [v2, v3, v1]  # 乱序传入，函数内应排序

        captured = {}

        def _fake_cut(sources, *args, **kw):
            captured["sources"] = sources
            captured["output"] = kw.get("output_path")
            os.makedirs(os.path.dirname(kw["output_path"]), exist_ok=True)
            with open(kw["output_path"], "w") as f:  # 模拟生成产物
                f.write("x")
            return kw.get("output_path")

        monkeypatch.setattr(detection, "cut_clips", _fake_cut)
        path, msg = detection.generate_highlights_fullgame("小明", 5, 5, 8)
        assert path is not None
        # 按文件名排序：1st 在前，3rd 无记录被跳过
        assert [os.path.basename(v) for v, _ in captured["sources"]] == ["1st.mp4", "2nd.mp4"]
        assert captured["sources"][0][1] == [10.0]
        assert captured["sources"][1][1] == [15.0]
        # 输出路径带文件夹名 + 人物
        assert os.path.basename(captured["output"]) == "2026.09.04-小明-highlights.mp4"
        # 该人物整场无片段 → 明确报错
        path, msg = detection.generate_highlights_fullgame("不存在的人", 5, 5, 8)
        assert path is None and "没有可导出" in msg
        # 未扫描文件夹 → 提示
        state_mod.batch_files = []
        path, msg = detection.generate_highlights_fullgame("小明", 5, 5, 8)
        assert path is None and "扫描文件夹" in msg

    def test_fullgame_fallback_single_video_folder(self, state_mod, monkeypatch, tmp_path):
        """单视频模式回退：batch_files 空 + 已加载视频 → 扫描其所在文件夹。"""
        from services import detection
        monkeypatch.setattr(detection, "state", state_mod)
        base = tmp_path / "2026.09.05"
        base.mkdir()
        v1 = str(base / "1st.mp4")
        v2 = str(base / "2nd.mp4")
        # 磁盘上要有真实文件（scan_video_files 按扩展名扫描目录）
        for v in (v1, v2):
            with open(v, "wb") as f:
                f.write(b"x")
        state_mod.add_history(v1, (1, 2, 3, 4), [10.0])
        state_mod.update_history_labels(v1, None, None, person_map={10.0: "小明"})
        state_mod.add_history(v2, (1, 2, 3, 4), [20.0])
        state_mod.update_history_labels(v2, None, None, person_map={20.0: "小明"})
        # 单视频模式：batch_files 为空，只加载了 v2
        state_mod.batch_files = []
        state_mod.video_state.update(path=v2)

        captured = {}

        def _fake_cut(sources, *args, **kw):
            captured["sources"] = sources
            captured["output"] = kw.get("output_path")
            os.makedirs(os.path.dirname(kw["output_path"]), exist_ok=True)
            with open(kw["output_path"], "w") as f:
                f.write("x")
            return kw.get("output_path")

        monkeypatch.setattr(detection, "cut_clips", _fake_cut)
        path, msg = detection.generate_highlights_fullgame("小明", 5, 5, 8)
        assert path is not None
        # 文件夹内两个视频都被发现（自然顺序 1st 在前），各自贡献 1 个片段
        assert [os.path.basename(v) for v, _ in captured["sources"]] == ["1st.mp4", "2nd.mp4"]
        assert captured["sources"][0][1] == [10.0]
        assert captured["sources"][1][1] == [20.0]
        assert os.path.basename(captured["output"]) == "2026.09.05-小明-highlights.mp4"
        # batch_files 空 + 未加载视频 → 明确报错
        state_mod.video_state.update(path=None)
        path, msg = detection.generate_highlights_fullgame("小明", 5, 5, 8)
        assert path is None and "加载视频" in msg

    def test_person_highlights_path_naming(self):
        """按人物导出的文件名：{视频名}-{人物}-highlights.mp4，人名安全化。"""
        from services import detection
        # 正常人名：跟随人物命名
        p = detection._person_highlights_path("/v/2026.09.04-2nd.mp4", "小明")
        assert os.path.basename(p) == "2026.09.04-2nd-小明-highlights.mp4"
        # Windows 非法字符剔除、首尾空白剥离
        p = detection._person_highlights_path("/v/a.mp4", ' 小/明:*? ')
        assert os.path.basename(p) == "a-小明-highlights.mp4"
        # 全非法字符 → 回落 "person"
        p = detection._person_highlights_path("/v/a.mp4", '\\/:*?"<>|')
        assert os.path.basename(p) == "a-person-highlights.mp4"
        # 超长人名截断到 30 字符
        p = detection._person_highlights_path("/v/a.mp4", "很" * 50)
        assert os.path.basename(p) == f"a-{'很' * 30}-highlights.mp4"
        # 不同人物不互相覆盖
        p1 = detection._person_highlights_path("/v/a.mp4", "小明")
        p2 = detection._person_highlights_path("/v/a.mp4", "小红")
        assert p1 != p2
        # 哨兵「仅已分类」→ 文件名用 已分类（与全部/各人物导出互不覆盖）
        p3 = detection._person_highlights_path("/v/a.mp4", detection.PERSON_FILTER_CLASSIFIED)
        assert os.path.basename(p3) == "a-已分类-highlights.mp4"
        assert p3 not in (p1, p2)


    def test_get_record_single_video(self, state_mod):
        """get_record：单视频历史记录直读（无记录返回 None）。"""
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [1.0], baseline_idx=7)
        r = state_mod.get_record("/a.mp4")
        assert r is not None and r["video"] == "/a.mp4"
        assert list(r["hoop"]) == [1, 2, 3, 4]
        assert r["baseline_idx"] == 7
        assert state_mod.get_record("/nope.mp4") is None

    def test_backfill_batch_calibs_from_history(self, state_mod, monkeypatch):
        """跨会话复用标定：扫描后 batch_calibs 从历史回填，会话内标定优先。"""
        from services import detection
        monkeypatch.setattr(detection, "state", state_mod)
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [1.0], baseline_idx=7)
        state_mod.add_history("/b.mp4", (5, 6, 7, 8), [2.0], baseline_idx=9)
        state_mod.batch_files = ["/a.mp4", "/b.mp4", "/c.mp4"]  # /c 无历史记录
        state_mod.batch_calibs = {}
        n = detection.backfill_batch_calibs_from_history()
        assert n == 2
        assert state_mod.batch_calibs["/a.mp4"] == {"hoop": (1, 2, 3, 4), "baseline_idx": 7}
        assert state_mod.batch_calibs["/b.mp4"] == {"hoop": (5, 6, 7, 8), "baseline_idx": 9}
        assert "/c.mp4" not in state_mod.batch_calibs
        # 本次会话已保存的标定不被历史覆盖
        state_mod.batch_calibs["/a.mp4"] = {"hoop": (9, 9, 9, 9), "baseline_idx": 1}
        assert detection.backfill_batch_calibs_from_history() == 0
        assert state_mod.batch_calibs["/a.mp4"]["hoop"] == (9, 9, 9, 9)


    def test_person_roster_cross_video(self, state_mod):
        """全局人物名单：跨视频复用，登记/去重/最近使用优先/持久化。"""
        # 空名单
        assert state_mod.load_persons() == []
        # 登记三个人物（视频 A 里用过）
        assert state_mod.add_person("小明")
        assert state_mod.add_person("小红")
        assert state_mod.add_person("老王")
        # 最近使用在前
        assert state_mod.load_persons() == ["老王", "小红", "小明"]
        # 重复登记 → 去重并提到队首
        assert state_mod.add_person("小明")
        assert state_mod.load_persons() == ["小明", "老王", "小红"]
        # 空名/空白名不登记
        assert not state_mod.add_person("")
        assert not state_mod.add_person("   ")
        assert state_mod.load_persons() == ["小明", "老王", "小红"]
        # 持久化：模拟重启后重读（persons.json 在缓存根目录）
        assert os.path.exists(os.path.join(state_mod.CACHE_ROOT, "persons.json"))

    def test_set_person_registers_roster(self, state_mod, monkeypatch):
        """set_person 动作自动登记进全局名单（跨视频复用的入口）。"""
        from services import detection
        monkeypatch.setattr(detection, "state", state_mod)
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [10.0])
        state_mod.last_goal_clips.extend(
            [{"ts": 10.0, "path": "/c.mp4", "idx": 0}])
        state_mod.video_state.update(path="/a.mp4")
        detection.clip_action("set_person", 0, person="小明")
        assert state_mod.load_persons() == ["小明"]
        # 清除分类不登记空名
        detection.clip_action("set_person", 0, person="")
        assert state_mod.load_persons() == ["小明"]


    def test_harvest_persons_from_history(self, state_mod):
        """存量回填：旧分类（仅 labels.persons、无 persons.json）→ 全局名单。"""
        state_mod.add_history("/a.mp4", (1, 2, 3, 4), [10.0])
        state_mod.update_history_labels("/a.mp4", None, None, person_map={10.0: "智"})
        state_mod.add_history("/b.mp4", (1, 2, 3, 4), [20.0])
        state_mod.update_history_labels("/b.mp4", None, None,
                                        person_map={20.0: "浩", 21.0: "智"})
        # 名单为空 → 回填收录全部历史人物（跨视频去重）
        assert state_mod.load_persons() == []
        assert state_mod.harvest_persons_from_history() == 2
        assert set(state_mod.load_persons()) == {"智", "浩"}
        # 幂等：重复回填不增不减
        assert state_mod.harvest_persons_from_history() == 0
        assert set(state_mod.load_persons()) == {"智", "浩"}
        # 显式登记（最近使用）优先在前，回填新名字追加在后
        state_mod.add_person("小明")
        state_mod.add_history("/c.mp4", (1, 2, 3, 4), [30.0])
        state_mod.update_history_labels("/c.mp4", None, None, person_map={30.0: "诚"})
        assert state_mod.harvest_persons_from_history() == 1
        persons = state_mod.load_persons()
        assert persons[0] == "小明" and "诚" in persons and "智" in persons
        # 历史里没有任何人物分类 → 0 且不写盘
        monkey = state_mod.PERSONS_FILE + ".probe"
        state_mod.PERSONS_FILE = monkey
        orig_load = state_mod.load_history
        state_mod.load_history = lambda: []
        try:
            assert state_mod.harvest_persons_from_history() == 0
            assert not os.path.exists(monkey)
        finally:
            state_mod.load_history = orig_load


class TestMultiSourceCut:
    def test_cut_clips_multi_video_concat(self, tmp_path):
        """多源切片端到端：两个视频的片段合并成一个集锦，按源顺序拼接。"""
        cv2 = pytest.importorskip("cv2")
        import shutil
        from cutter import ffmpeg_cutter as fc
        if not (os.path.exists(fc._DEFAULT_FFMPEG)
                or shutil.which(fc._DEFAULT_FFMPEG)):
            pytest.skip("无可用 ffmpeg（当前解释器未装 imageio_ffmpeg 且 PATH 无 ffmpeg）")
        import numpy as np
        from cutter.ffmpeg_cutter import cut_clips

        def _make(path, n=150):
            """150 帧 @30fps = 5s 合成视频。"""
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
            if not w.isOpened():
                pytest.skip("cv2 无 mp4v 编码器")
            for i in range(n):
                f = np.full((240, 320, 3), 100, dtype=np.uint8)
                cv2.circle(f, (160, 120), 8 + (i % 5), (255, 255, 255), -1)
                w.write(f)
            w.release()
            return str(path)

        v1 = _make(tmp_path / "1st.mp4")
        v2 = _make(tmp_path / "2nd.mp4")
        out = str(tmp_path / "game.mp4")
        # 每源两个靠近的 ts（min_gap=8 合并为一段）→ 每源 1 段，共 2 段
        # v1: [0,4]+[2,6] → [0,6]；v2: [0,3]+[1,5] → [0,5] → 总时长 ≈ 11s
        result = cut_clips([(v1, [2.0, 4.0]), (v2, [1.0, 3.0])], None,
                           pre_roll=2, post_roll=2, min_gap=8, output_path=out)
        assert result == out and os.path.exists(out) and os.path.getsize(out) > 0
        from video_io import get_video_info
        info = get_video_info(out)
        assert 9.5 < info["total"] / info["fps"] < 12.5
        # 跨源不合并：min_gap=60 也只应在源内生效，两段分别 [3,5] 与 [0,1.2]
        out2 = str(tmp_path / "game2.mp4")
        result = cut_clips([(v1, [4.0]), (v2, [0.2])], None,
                           pre_roll=1, post_roll=1, min_gap=60, output_path=out2)
        assert result == out2
        info2 = get_video_info(out2)
        # 两段 ≈ 3.2s；若错误跨源合并会变成单段 [0,5] = 5s
        assert 2.0 < info2["total"] / info2["fps"] < 4.5

    def test_cut_clips_multi_empty(self):
        """多源空列表 / 空时间戳源 → 跳过或返回 None，不崩溃。"""
        import shutil
        from cutter import ffmpeg_cutter as fc
        from cutter.ffmpeg_cutter import cut_clips
        assert cut_clips([], None) is None  # 空列表在 ffmpeg 探测前早退
        if not (os.path.exists(fc._DEFAULT_FFMPEG)
                or shutil.which(fc._DEFAULT_FFMPEG)):
            pytest.skip("无可用 ffmpeg（当前解释器未装 imageio_ffmpeg 且 PATH 无 ffmpeg）")
        # 源列表里某个源时间戳为空 → 该源被跳过（整体仍可工作）
        assert cut_clips([("/nonexistent.mp4", [])], None,
                         output_path="/tmp/x.mp4") is None


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
