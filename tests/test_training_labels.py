"""训练侧标签口径回归（R7 / N4）。

R7：extract_features 的断点续跑原先只按 event_id 跳过，features.jsonl 的 label
冻结在首次提取时——用户改标 + 重跑 export 之后再跑本脚本不会更新标签，train_lgbm
就用旧标签训练。修复后改标走 _relabel_features 原地改写（特征不变，不重算）。
"""
import importlib
import json
import sys
import types

import pytest


@pytest.fixture
def ef_mod(monkeypatch):
    """加载 training/extract_features.py，绕过 app(torch) / video_io(av) 重依赖。"""
    fake_app = types.ModuleType("app")
    fake_app.get_ball_model = lambda: (None, "")
    fake_app.get_ball_class_ids = lambda m, w="": []
    fake_app.get_device = lambda: "cpu"
    monkeypatch.setitem(sys.modules, "app", fake_app)
    fake_io = types.ModuleType("video_io")
    fake_io.VideoReader = object
    fake_io.get_video_info = lambda p: {}
    monkeypatch.setitem(sys.modules, "video_io", fake_io)
    monkeypatch.delitem(sys.modules, "training.extract_features", raising=False)
    return importlib.import_module("training.extract_features")


def _write(path, rows, extra_lines=()):
    lines = [json.dumps(r, ensure_ascii=False) for r in rows] + list(extra_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


class TestRelabelFeatures:
    def test_rewrites_only_changed_label(self, ef_mod, tmp_path):
        """改标：只动 label 字段，特征值原样保留（不重算特征）。"""
        p = tmp_path / "features.jsonl"
        _write(p, [{"event_id": "a", "label": 1, "f0": 0.5, "f1": 0.7},
                   {"event_id": "b", "label": 0, "f0": 0.2, "f1": 0.1}])
        assert ef_mod._relabel_features(p, {"a": 0}) == 1
        rows = _read(p)
        assert rows[0]["label"] == 0
        assert rows[0]["f0"] == 0.5 and rows[0]["f1"] == 0.7   # 特征不动
        assert rows[1]["label"] == 0                          # 未指定的事件不动

    def test_no_change_writes_nothing(self, ef_mod, tmp_path):
        """标签本来就一致 → 返回 0 且不触碰文件（避免无谓 IO/时间戳变化）。"""
        p = tmp_path / "features.jsonl"
        _write(p, [{"event_id": "a", "label": 1}])
        before = p.stat().st_mtime_ns
        assert ef_mod._relabel_features(p, {"a": 1}) == 0
        assert p.stat().st_mtime_ns == before
        assert not (tmp_path / "features.jsonl.tmp").exists()

    def test_corrupt_line_preserved(self, ef_mod, tmp_path):
        """坏行必须原样保留——静默丢弃会永久损失已抽特征。"""
        p = tmp_path / "features.jsonl"
        _write(p, [{"event_id": "a", "label": 1}], extra_lines=["{ not json"])
        assert ef_mod._relabel_features(p, {"a": 0}) == 1
        text = p.read_text(encoding="utf-8")
        assert "{ not json" in text                       # 坏行原样在
        good = [l for l in text.splitlines() if l.strip() and l.strip() != "{ not json"]
        assert json.loads(good[0])["label"] == 0          # 好行已改写

    def test_atomic_replace_leaves_no_tmp(self, ef_mod, tmp_path):
        p = tmp_path / "features.jsonl"
        _write(p, [{"event_id": "a", "label": 1}])
        ef_mod._relabel_features(p, {"a": 0})
        assert not (tmp_path / "features.jsonl.tmp").exists()
        assert _read(p)[0]["label"] == 0

    def test_unknown_event_id_ignored(self, ef_mod, tmp_path):
        p = tmp_path / "features.jsonl"
        _write(p, [{"event_id": "a", "label": 1}])
        assert ef_mod._relabel_features(p, {"zzz": 0}) == 0
        assert _read(p)[0]["label"] == 1
