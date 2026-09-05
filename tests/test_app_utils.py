"""app.get_ball_class_ids 与 video_utils.scan_video_files 单元测试。"""
import sys
import types

import pytest

from services import video_utils
from app import get_ball_class_ids, get_model, _model_cache, _model_fail_cache


class _FakeModel:
    def __init__(self, names):
        self.names = names


class TestBallClassIds:
    def test_custom_basketball_weights(self):
        """自定义微调权重 names={0:'basketball'} → [0]（保持原行为）。"""
        assert get_ball_class_ids(_FakeModel({0: "basketball"})) == [0]

    def test_coco_fallback_weights(self):
        """COCO 回退权重：类 0 是 person，sports ball 是 32 → 必须反查出 [32]。"""
        names = {0: "person", 32: "sports ball", 39: "bottle"}
        assert get_ball_class_ids(_FakeModel(names)) == [32]

    def test_multiple_ball_classes_sorted(self):
        names = {5: "soccer ball", 2: "basketball", 9: "car"}
        assert get_ball_class_ids(_FakeModel(names)) == [2, 5]

    def test_empty_names_fallback_empty(self):
        """names 异常 → 返回 []（调用方应拒绝检测，而不是把 person 当球确认）。"""
        assert get_ball_class_ids(_FakeModel({})) == []
        assert get_ball_class_ids(_FakeModel(None)) == []

    def test_baseball_bat_not_matched(self):
        """精确匹配：'baseball bat'/'baseball glove' 含子串 ball 但不是球类。"""
        names = {0: "person", 39: "baseball bat", 38: "baseball glove", 32: "sports ball"}
        assert get_ball_class_ids(_FakeModel(names)) == [32]


class TestNaturalSort:
    def test_scan_video_files_natural_order(self, tmp_path):
        """自然排序：a1 < a2 < a10（字典序会把 a10 排在 a2 前）。"""
        for name in ["a10.mp4", "b3.avi", "a2.mp4", "a1.mp4", "readme.txt", "sub"]:
            p = tmp_path / name
            if name == "sub":
                p.mkdir()
            else:
                p.write_text("x")
        files = video_utils.scan_video_files(str(tmp_path))
        basenames = [f.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for f in files]
        assert basenames == ["a1.mp4", "a2.mp4", "a10.mp4", "b3.avi"]

    def test_scan_invalid_folder(self):
        assert video_utils.scan_video_files("/nonexistent-folder-xyz") == []
        assert video_utils.scan_video_files("") == []


def _fake_ultralytics(YOLO_cls):
    """向 sys.modules 注入假 ultralytics 模块（app.get_model 内 import）。"""
    mod = types.ModuleType("ultralytics")
    mod.YOLO = YOLO_cls
    return mod


class TestGetModel:
    """B4+B5: 模型加载/设备迁移失败显式报错 + 失败负缓存。"""

    def test_load_failure_raises_and_negative_cached(self, monkeypatch):
        """加载失败抛带权重的明确 RuntimeError；第二次调用不再重复加载（负缓存）。"""
        import app
        calls = {"n": 0}

        class _FakeYOLO:
            def __init__(self, weights):
                calls["n"] += 1
                raise RuntimeError("weights corrupt")

        monkeypatch.setitem(sys.modules, "ultralytics", _fake_ultralytics(_FakeYOLO))
        _model_cache.clear()
        _model_fail_cache.clear()
        try:
            with pytest.raises(RuntimeError) as ei:
                get_model("bad.pt")
            assert "YOLO 模型加载失败" in str(ei.value)
            assert "bad.pt" in str(ei.value)
            with pytest.raises(RuntimeError):
                get_model("bad.pt")
            assert calls["n"] == 1          # 负缓存命中，未再次慢加载
        finally:
            _model_cache.clear()
            _model_fail_cache.clear()

    def test_device_migration_failure_raises(self, monkeypatch):
        """B4: m.to(device) 失败必须抛错，不能只 log warning 后让推理端每次失败。"""
        import app
        calls = {"n": 0}

        class _FakeYOLO:
            def __init__(self, weights):
                calls["n"] += 1

            def to(self, dev):
                raise RuntimeError("cuda oom")

        monkeypatch.setitem(sys.modules, "ultralytics", _fake_ultralytics(_FakeYOLO))
        monkeypatch.setattr(app, "get_device", lambda: "cuda:0")
        _model_cache.clear()
        _model_fail_cache.clear()
        try:
            with pytest.raises(RuntimeError) as ei:
                get_model("mig.pt")
            assert "YOLO 模型加载失败" in str(ei.value)
            # 失败未进正缓存：清负缓存后重试仍会重新尝试（失败可被恢复）
            assert "mig.pt" not in _model_cache
        finally:
            _model_cache.clear()
            _model_fail_cache.clear()
