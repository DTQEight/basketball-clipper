"""手动插入进球（补漏检）的单元测试。

覆盖三块：① 时间文本解析；② 记录的「检测 ∪ 手动」合并；③ 增删与**重检测不丢**。
"""
import os

import pytest

from services import detection, state

V = "Y:/x/video.mp4"


# ==================== 时间文本解析 ====================

class TestParseTs:
    @pytest.mark.parametrize("text,expect", [
        ("256.5", 256.5),
        (" 256.5 ", 256.5),
        ("4:16.5", 256.5),          # 分:秒
        ("4:16", 256.0),
        ("1:04:16", 3856.0),        # 时:分:秒
        ("0:09.9", 9.9),
        ("4：16.5", 256.5),         # 全角冒号
        (0, 0.0),
        (256.5, 256.5),
    ])
    def test_valid(self, text, expect):
        assert detection.parse_ts(text) == pytest.approx(expect)

    @pytest.mark.parametrize("text", [
        None, "", "   ", "abc", "4:70", "1:99:00", "-5", "1:2:3:4", "::", "4::16",
    ])
    def test_invalid(self, text):
        assert detection.parse_ts(text) is None


# ==================== 记录合并 ====================

class TestRecordGoals:
    def test_union_sorted_deduped(self):
        rec = {"goals": [100.0, 200.0004], "manual_goals": [300.0, 100.0]}
        assert state.record_goals(rec) == [100.0, 200.0, 300.0]

    def test_only_manual(self):
        assert state.record_goals({"goals": [], "manual_goals": [42.5]}) == [42.5]

    def test_empty_and_junk(self):
        assert state.record_goals({}) == []
        assert state.record_goals(None) == []
        assert state.record_goals({"goals": ["bad", None], "manual_goals": [1.0]}) == [1.0]


# ==================== 增删（落盘） ====================

@pytest.fixture()
def hist(tmp_path, monkeypatch):
    """把历史目录指到临时目录，并写入一条基线记录（检测出 100s / 200s 两个球）。"""
    monkeypatch.setattr(state, "HISTORY_DIR", str(tmp_path / "history"))
    os.makedirs(state.HISTORY_DIR, exist_ok=True)
    state.save_history([{
        "video": V, "video_name": "video.mp4", "hoop": [0, 0, 10, 10],
        "goals": [100.0, 200.0], "labels": {"kept": [100.0]},
    }])
    return tmp_path


class TestAddRemove:
    def test_add_appends_manual_and_label(self, hist):
        status, ts, _ = state.add_manual_goal(V, 300.5, person="聪")
        assert (status, ts) == ("added", 300.5)
        rec = state.get_record(V)
        assert rec["manual_goals"] == [300.5]
        assert 300.5 in rec["labels"]["kept"]
        assert rec["labels"]["persons"][str(300.5)] == "聪"
        # 检测出的 goals 不被污染
        assert rec["goals"] == [100.0, 200.0]

    def test_add_near_existing_goal_is_deduped(self, hist):
        status, ts, msg = state.add_manual_goal(V, 100.4)
        assert status == "exists" and ts == pytest.approx(100.0)
        assert "未重复插入" in msg
        assert not state.get_record(V).get("manual_goals")

    def test_add_same_ts_twice_is_idempotent(self, hist):
        state.add_manual_goal(V, 300.0)
        status, ts, _ = state.add_manual_goal(V, 300.0)
        assert status == "exists" and ts == pytest.approx(300.0)
        assert state.get_record(V)["manual_goals"] == [300.0]

    def test_add_clears_previous_reject(self, hist):
        """之前判过 × 的时刻再插入 → 视为改判，反例要清掉（否则集锦会漏、训练集打架）。"""
        rec = state.get_record(V)
        rec["labels"]["deleted"] = [300.0]
        state.save_history([rec])
        state.add_manual_goal(V, 300.0)
        rec = state.get_record(V)
        assert 300.0 not in (rec["labels"].get("deleted") or [])
        assert 300.0 in rec["labels"]["kept"]

    def test_add_invalid_ts(self, hist):
        assert state.add_manual_goal(V, "abc")[0] == "error"
        assert state.add_manual_goal(V, -1)[0] == "error"

    def test_add_without_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state, "HISTORY_DIR", str(tmp_path / "h2"))
        os.makedirs(state.HISTORY_DIR, exist_ok=True)
        status, ts, msg = state.add_manual_goal("Y:/nope.mp4", 10.0)
        assert status == "error" and ts is None and "还没有历史记录" in msg

    def test_remove_manual(self, hist):
        state.add_manual_goal(V, 300.0, person="浩")
        ok, msg = state.remove_manual_goal(V, 300.0)
        assert ok and "已删除" in msg
        rec = state.get_record(V)
        assert not rec.get("manual_goals")
        assert 300.0 not in rec["labels"]["kept"]
        assert str(300.0) not in rec["labels"]["persons"]

    def test_remove_detected_goal_is_refused(self, hist):
        """检测出的进球不归「撤销手动」管（那是 √/× 的语义）。"""
        ok, msg = state.remove_manual_goal(V, 100.0)
        assert not ok and "不是手动插入" in msg
        assert state.get_record(V)["goals"] == [100.0, 200.0]


# ==================== 重检测不丢（最关键的回归） ====================

class TestSurvivesRedetect:
    def test_manual_goals_kept_after_redetect(self, hist):
        state.add_manual_goal(V, 300.0)
        new = state.add_history(V, (0, 0, 10, 10), [100.0, 200.0, 250.0])
        assert new["goals"] == [100.0, 200.0, 250.0]
        assert new.get("manual_goals") == [300.0]
        assert 300.0 in state.record_goals(new)

    def test_manual_goal_dropped_when_detection_covers_it(self, hist):
        """重检测把同一个球检出来了（相距 ≤1s）→ 手动条目让位给检测结果，避免重复卡片。"""
        state.add_manual_goal(V, 300.0)
        new = state.add_history(V, (0, 0, 10, 10), [100.0, 200.0, 300.4])
        assert not new.get("manual_goals")
        assert len(state.record_goals(new)) == len(set(new["goals"]))

    def test_manual_goals_survive_second_redetect(self, hist):
        """连续两次重检也不能把手动插入吃掉（幂等）。"""
        state.add_manual_goal(V, 300.0)
        state.add_history(V, (0, 0, 10, 10), [100.0, 200.0])
        state.add_history(V, (0, 0, 10, 10), [110.0, 210.0])
        assert state.get_record(V).get("manual_goals") == [300.0]
