# -*- coding: utf-8 -*-
"""篮筐跟踪（支架被撞场景）测试：HoopTracker / GoalDetector.set_hoop / _hoop_at。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import _ROOT  # noqa: F401  确保 sys.path 已注入

from services.hoop_tracker import HoopTracker

_HOOP = (100, 60, 160, 120)   # 60x60
_FH, _FW = 272, 480


def _make_frame(texture, x, y):
    """深灰底 + 在 (x, y) 放置纹理块（唯一可识别结构）。"""
    f = np.full((_FH, _FW, 3), 60, dtype=np.uint8)
    th, tw = texture.shape[:2]
    f[y:y + th, x:x + tw] = texture
    return f


def _texture(seed=7, size=60):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return t


class TestHoopTracker:
    def test_no_move_returns_none(self):
        tex = _texture()
        tr = HoopTracker(_make_frame(tex, 100, 60), _HOOP)
        # 同位置（纹理完全一致）→ 不报移位
        assert tr.update(_make_frame(tex, 100, 60)) is None
        assert tr.n_moves == 0

    def test_move_detected_and_followed(self):
        tex = _texture()
        tr = HoopTracker(_make_frame(tex, 100, 60), _HOOP)
        # 纹理整体平移 (40, 25)：画面里篮筐移位
        mv = tr.update(_make_frame(tex, 140, 85))
        assert mv is not None, "移位未被检出"
        assert mv["shift_px"] >= 40
        nx1, ny1, nx2, ny2 = mv["hoop"]
        assert abs(nx1 - 140) <= 2 and abs(ny1 - 85) <= 2, mv["hoop"]
        assert abs(nx2 - 200) <= 2 and abs(ny2 - 145) <= 2, mv["hoop"]
        assert tr.n_moves == 1
        # 跟踪器已跟上：新位置再查 → 无移位
        assert tr.update(_make_frame(tex, 140, 85)) is None

    def test_two_consecutive_moves(self):
        tex = _texture()
        tr = HoopTracker(_make_frame(tex, 100, 60), _HOOP)
        mv1 = tr.update(_make_frame(tex, 150, 100))
        mv2 = tr.update(_make_frame(tex, 90, 130))
        assert mv1 is not None and mv2 is not None
        assert abs(mv2["hoop"][0] - 90) <= 2 and abs(mv2["hoop"][1] - 130) <= 2
        assert tr.n_moves == 2

    def test_lost_on_blank_frame(self):
        tex = _texture()
        tr = HoopTracker(_make_frame(tex, 100, 60), _HOOP)
        # 空白帧（纹理消失 = 遮挡/变焦）：不误报移位，保持原位置
        blank = np.full((_FH, _FW, 3), 60, dtype=np.uint8)
        assert tr.update(blank) is None
        assert tr.lost_streak >= 1
        assert tr.box == (100.0, 60.0, 160.0, 120.0)

    def test_small_jitter_not_a_move(self):
        """亚阈值抖动（< move_thr）不算移位，模板继续用旧位置。"""
        tex = _texture()
        tr = HoopTracker(_make_frame(tex, 100, 60), _HOOP)
        assert tr.update(_make_frame(tex, 104, 62)) is None
        assert tr.n_moves == 0


class TestGoalDetectorSetHoop:
    def test_set_hoop_updates_all_derived(self):
        from tracker import GoalDetector
        base = _make_frame(_texture(), 100, 60)
        det = GoalDetector(_HOOP, baseline_frame=base, fps=30.0,
                           search_margin=80)
        # 制造进行中的状态（模拟检测到一半）
        det.blob_above_hoop = True
        det.blob_in_hoop = True
        det.blob_in_hoop_frames = 3
        det.last_blob_box = (110, 70, 130, 90)

        new_hoop = (200, 150, 260, 210)
        det.set_hoop(new_hoop, frame=_make_frame(_texture(seed=9), 200, 150),
                     frame_idx=1234)

        assert (det.hoop_x1, det.hoop_y1, det.hoop_x2, det.hoop_y2) == (200, 150, 260, 210)
        assert det.hoop_cx == 230 and det.hoop_cy == 180
        assert det.hoop_w == 60 and det.hoop_h == 60
        assert det.search_x1 == 120 and det.search_y1 == 70   # margin=80
        # 状态机复位
        assert det.blob_above_hoop is False
        assert det.blob_in_hoop is False
        assert det.blob_in_hoop_frames == 0
        assert det.last_blob_box is None
        # 基准帧已重置（对应新位置 ROI，非空）
        assert det.baseline_gray is not None and det.baseline_gray.size > 0
        assert det.last_baseline_frame_idx == 1234

    def test_baseline_matches_new_roi(self):
        """重置后的基准帧尺寸应等于新搜索区 ROI（错位会导致整段 diff 失效）。"""
        from tracker import GoalDetector
        det = GoalDetector(_HOOP, baseline_frame=_make_frame(_texture(), 100, 60),
                           fps=30.0, search_margin=80)
        frame2 = _make_frame(_texture(seed=9), 200, 150)
        det.set_hoop((200, 150, 260, 210), frame=frame2, frame_idx=100)
        expect = det._roi_gray(frame2)
        assert det.baseline_gray.shape == expect.shape


class TestHoopAt:
    """按事件时间取篮筐坐标（验证阶段用）。"""

    def test_lookup(self):
        from services.goal_verifier import _hoop_at
        track = [
            {"frame": 300, "ts": 10.0, "hoop": [1, 2, 3, 4]},
            {"frame": 1500, "ts": 50.0, "hoop": [5, 6, 7, 8]},
        ]
        default = (9, 9, 9, 9)
        assert _hoop_at(track, 5.0, default) == default        # 移位前
        assert _hoop_at(track, 10.0, default) == (1, 2, 3, 4)  # 恰好移位时点
        assert _hoop_at(track, 30.0, default) == (1, 2, 3, 4)  # 两次移位之间
        assert _hoop_at(track, 60.0, default) == (5, 6, 7, 8)  # 第二次移位后
        assert _hoop_at(None, 60.0, default) == default        # 无轨迹


class TestHistoryHoopTrack:
    def test_add_history_persists_track(self, tmp_path, monkeypatch):
        """hoop_track 走特殊嵌套字段持久化（未知字段路径会抛 TypeError）。"""
        import importlib
        monkeypatch.setenv("BBALL_CACHE_ROOT", str(tmp_path))
        from services import state
        state = importlib.reload(state)
        state.add_history("/a.mp4", (1, 2, 3, 4), [10.0, 60.0],
                          hoop_track=[{"frame": 300, "ts": 10.02, "hoop": [1, 2, 3, 4]},
                                      {"frame": 1500, "ts": 50.0, "hoop": [5, 6, 7, 8]}])
        rec = state.load_history()[0]
        assert rec["hoop_track"][0] == {"frame": 300, "ts": 10.02, "hoop": [1, 2, 3, 4]}
        assert rec["hoop_track"][1]["hoop"] == [5, 6, 7, 8]

    def test_add_history_without_track(self, tmp_path, monkeypatch):
        import importlib
        monkeypatch.setenv("BBALL_CACHE_ROOT", str(tmp_path))
        from services import state
        state = importlib.reload(state)
        state.add_history("/b.mp4", (1, 2, 3, 4), [1.0])
        assert "hoop_track" not in state.load_history()[0]
