# -*- coding: utf-8 -*-
"""裁剪推理：输入框必须 ⊇ 筐接受框，坐标映射必须可逆。

背景：整帧 1920×1080 letterbox 到 1280 时球只剩 ~13-20px，单次推理 ~28ms。
改为只推「筐接受框 + 20% 外扩」（约 430×500）并推 512，球保持 ~20-31px。
唯一的风险是输入范围没盖住接受框 → 球在筐附近却不在输入里 → 静默漏球，
所以把「⊇ 接受框」写成断言。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import detection      # noqa: E402
from tracker import GoalDetector    # noqa: E402

H, W = 1080, 1920
FRAME_SHAPE = (H, W, 3)


def _det(hoop):
    return GoalDetector(hoop_box=hoop, loose_mode=True, yolo_confirm=True,
                        diff_threshold=25, min_blob_area=30, search_margin=80,
                        rolling_baseline_sec=0, auto_threshold=False, fps=30.0)


class TestYoloInputBox:
    @pytest.mark.parametrize('hoop', [
        (421, 246, 510, 350),      # 中景
        (1364, 210, 1470, 322),    # 右侧
        (20, 30, 120, 140),        # 贴左上角（会被 clamp）
        (1830, 950, 1915, 1070),   # 贴右下角（会被 clamp）
        (289, 75, 416, 239),       # 大框（08.15-1st）
    ])
    def test_crop_contains_accept_box(self, hoop):
        d = _det(hoop)
        ax1, ay1, ax2, ay2 = d.yolo_accept_box()
        box = detection._yolo_input_box(d.yolo_accept_box(), FRAME_SHAPE)
        assert box is not None
        x1, y1, x2, y2 = box
        # 接受框与画面求交后必须被完全包含（画面外的部分本来就不可能检到）
        ix1, iy1 = max(0, ax1), max(0, ay1)
        ix2, iy2 = min(W, ax2), min(H, ay2)
        assert x1 <= ix1 and y1 <= iy1 and x2 >= ix2 and y2 >= iy2, \
            '裁剪框 %s 未覆盖接受框交 %s' % (box, (ix1, iy1, ix2, iy2))

    def test_crop_inside_frame(self):
        for hoop in [(421, 246, 510, 350), (20, 30, 120, 140), (1830, 950, 1915, 1070)]:
            x1, y1, x2, y2 = detection._yolo_input_box(_det(hoop).yolo_accept_box(),
                                                       FRAME_SHAPE)
            assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H

    def test_margin_zero_means_full_frame(self):
        d = _det((421, 246, 510, 350))
        assert detection._yolo_input_box(d.yolo_accept_box(), FRAME_SHAPE,
                                         margin=0) is None
        assert detection._yolo_input_box(d.yolo_accept_box(), FRAME_SHAPE,
                                         margin=-1) is None

    def test_degenerate_hoop_returns_none(self):
        assert detection._yolo_input_box((100, 100, 100, 100), FRAME_SHAPE) is None
        assert detection._yolo_input_box((200, 100, 100, 100), FRAME_SHAPE) is None


class TestYoloInput:
    def test_crop_is_small_and_offsets_match(self):
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        d = _det((421, 246, 510, 350))
        crop, ox, oy, imgsz = detection._yolo_input(frame, d)
        x1, y1, x2, y2 = detection._yolo_input_box(d.yolo_accept_box(), FRAME_SHAPE)
        assert (ox, oy) == (x1, y1)
        assert crop.shape == (y2 - y1, x2 - x1, 3)
        assert imgsz == detection.YOLO_IMGSZ_CROP
        # 裁剪面积应显著小于整帧（本方案省时的来源）
        assert crop.size < frame.size * 0.2

    def test_full_frame_when_crop_disabled(self, monkeypatch):
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        d = _det((421, 246, 510, 350))
        monkeypatch.setattr(detection, 'YOLO_CROP_MARGIN', 0)
        crop, ox, oy, imgsz = detection._yolo_input(frame, d)
        assert crop is frame and (ox, oy) == (0, 0)
        assert imgsz == detection.YOLO_IMGSZ_FULL

    @pytest.mark.parametrize('hoop', [
        (421, 246, 510, 350), (1364, 210, 1470, 322), (298, 152, 413, 280),
        (250, 308, 342, 424), (1305, 272, 1402, 367),
    ])
    def test_scale_does_not_shrink_ball(self, hoop):
        """imgsz/裁剪长边 ≥ 0.9：球在输入里不被缩小。

        这是「裁剪不牺牲小目标」的量化依据——旧注释记载 960 太小（球 12-17px）
        才提到 1280，所以裁剪路径的缩放必须优于整帧 1280 路径（1280/1920 = 0.667）。
        """
        d = _det(hoop)
        x1, y1, x2, y2 = detection._yolo_input_box(d.yolo_accept_box(), FRAME_SHAPE)
        scale = detection.YOLO_IMGSZ_CROP / max(x2 - x1, y2 - y1)
        full_scale = detection.YOLO_IMGSZ_FULL / max(FRAME_SHAPE[:2])
        assert scale >= 0.9, '缩放 %.2f 会把球缩小' % scale
        assert scale > full_scale, '裁剪路径 %.2f 不优于整帧路径 %.2f' % (scale, full_scale)

    def test_crop_writes_through_to_same_pixels(self):
        """裁剪只是切片：crop 中的球像素位置 + 偏移 = 整帧中的位置。"""
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        frame[300, 460] = 255
        d = _det((421, 246, 510, 350))
        crop, ox, oy, _ = detection._yolo_input(frame, d)
        ys, xs = np.nonzero(crop[:, :, 0])
        assert (xs[0] + ox, ys[0] + oy) == (460, 300)


class TestShiftBallsToFrame:
    def test_shift(self):
        balls = [(10.0, 20.0, 5.0, 15.0, 15.0, 25.0, 0.9)]
        out = detection._shift_balls_to_frame(balls, 400, 200)
        assert out == [(410.0, 220.0, 405.0, 215.0, 415.0, 225.0, 0.9)]

    def test_identity_when_no_offset(self):
        balls = [(10.0, 20.0, 5.0, 15.0, 15.0, 25.0, 0.9)]
        assert detection._shift_balls_to_frame(balls, 0, 0) is balls

    def test_empty(self):
        assert detection._shift_balls_to_frame([], 10, 10) == []
        assert detection._shift_balls_to_frame(None, 10, 10) is None


class TestAcceptBoxGeometryUnchanged:
    """_check_yolo_near_hoop 抽出的几何必须与原实现逐值一致。"""

    def test_values(self):
        for hoop in [(421, 246, 510, 350), (1364, 210, 1470, 322)]:
            x1, y1, x2, y2 = hoop
            w, h = x2 - x1, y2 - y1
            assert _det(hoop).yolo_accept_box() == pytest.approx(
                (x1 - w * 1.5, y1 - h * 1.0, x2 + w * 1.5, y2 + h * 2.0))
