# -*- coding: utf-8 -*-
"""YOLO 推理输入：输入框必须 ⊇ 筐接受框，三种模式的画布/偏移/坐标映射必须正确。

三种模式（`detection.YOLO_INPUT_MODE`）：
  · 'crop' 真裁剪（**默认**）：切出输入框 → 偏移为框左上角，省时（单次 ~22ms）
  · 'mask' 抹黑画布（**两段式的备用档** `YOLO_PROBE_MODE`）：整帧画布不动，只把
    输入框外抹黑 → 偏移恒为 0，几何与训练分布一致（单次 ~45ms）
  · 'full' 整帧（旧行为）：原样送整帧

2026.09.21 实测（4 场 / 80 个真球 / 518 个真硬帧）推翻了「裁剪不牺牲小目标」：
真裁剪@640 在硬帧上命中 0%（球被放大到 ~22px，但画布变成竖版小图 488×583，
与训练的 16:9 分布不符）；抹黑画布@1280 命中 93%、正常帧仅损 6%。
但库里 5 场 A/B 显示全局换 mask 只换来 +1 个真球、却要 +14.5% 检测耗时，故最终采用
**两段式**：默认裁剪，仅在轨迹即将被 YOLO 硬否决时用备用档补检一次
（见 tests/test_tracker.py 的 TestYoloProbe）。

唯一的结构性风险仍是「输入范围没盖住接受框 → 球在筐附近却不在输入里 → 静默漏球」，
所以把「⊇ 接受框」写成断言，并对 mask 模式断言「框外确实被抹黑、框内像素原样保留」。
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
        crop, ox, oy, imgsz = detection._yolo_input(frame, d, mode='crop')
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
        """'crop' 模式的缩放 ≥ 0.9（球不被缩小）——这是对照档的性质，不是线上档。

        注意：缩放不缩小 ≠ 检出更好。2026.09.21 实测证明裁剪档在真球帧上命中 0%
        （几何换成竖版小图），线上已改用 'mask'；此断言只用于守住对照档的语义。
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
        crop, ox, oy, _ = detection._yolo_input(frame, d, mode='crop')
        ys, xs = np.nonzero(crop[:, :, 0])
        assert (xs[0] + ox, ys[0] + oy) == (460, 300)


class TestYoloInputMask:
    """抹黑画布（两段式的备用档）：画布尺寸不变、框外全黑、框内像素原样、偏移为 0。"""

    def test_default_mode_is_crop_and_probe_is_mask(self):
        """默认档是裁剪（快），备用档是抹黑画布——两者不同才有兜底的意义。"""
        assert detection.YOLO_INPUT_MODE == 'crop', '默认档必须是裁剪（省时）'
        assert detection.YOLO_PROBE_MODE == 'mask', '兜底档必须是抹黑画布'
        assert detection.YOLO_PROBE_MODE != detection.YOLO_INPUT_MODE, \
            '两档相同时补检必然得出同样的"无球"结论，兜底无意义（run_detect 会跳过装配）'

    @pytest.mark.parametrize('hoop', [
        (421, 246, 510, 350), (1364, 210, 1470, 322), (20, 30, 120, 140),
        (1830, 950, 1915, 1070),
    ])
    def test_canvas_size_and_offsets(self, hoop):
        frame = np.full(FRAME_SHAPE, 255, dtype=np.uint8)
        d = _det(hoop)
        out, ox, oy, imgsz = detection._yolo_input(frame, d, mode='mask')
        assert out.shape == frame.shape, '画布必须保持整帧尺寸（几何与训练一致）'
        assert (ox, oy) == (0, 0), '偏移必须为 0，否则坐标换算会错'
        assert imgsz == detection.YOLO_IMGSZ_MASK
        assert imgsz >= detection.YOLO_IMGSZ_CROP, '抹黑档必须用更高的输入分辨率'

    @pytest.mark.parametrize('hoop', [
        (421, 246, 510, 350), (1364, 210, 1470, 322), (20, 30, 120, 140),
        (1830, 950, 1915, 1070),
    ])
    def test_window_kept_and_outside_blacked(self, hoop):
        """框内保留原像素、框外全黑；且亮区边界就是输入框（= 接受框被完整覆盖）。"""
        frame = np.full(FRAME_SHAPE, 255, dtype=np.uint8)
        d = _det(hoop)
        out, _ox, _oy, _ = detection._yolo_input(frame, d, mode='mask')
        x1, y1, x2, y2 = detection._yolo_input_box(d.yolo_accept_box(), FRAME_SHAPE)
        ys, xs = np.nonzero(out[:, :, 0])
        assert (xs.min(), ys.min(), xs.max(), ys.max()) == (x1, y1, x2 - 1, y2 - 1)
        # 框内像素原样（没有被改动/缩放）
        assert out[y1:y2, x1:x2].min() == 255
        # 接受框的实际范围仍在亮区内（框 ⊇ 接受框）
        ax1, ay1, ax2, ay2 = d.yolo_accept_box()
        assert x1 <= max(0, ax1) and y1 <= max(0, ay1)
        assert x2 >= min(W, ax2) and y2 >= min(H, ay2)

    def test_ball_pixel_inside_window_survives(self):
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        frame[300, 460] = 255          # 球（在窗口内）
        frame[100, 100] = 255          # 窗口外的干扰
        d = _det((421, 246, 510, 350))
        out, _ox, _oy, _ = detection._yolo_input(frame, d, mode='mask')
        assert out[300, 460].tolist() == [255, 255, 255]
        assert out[100, 100].tolist() == [0, 0, 0]

    def test_margin_zero_returns_frame_identity(self, monkeypatch):
        """margin<=0 仍表示「不做任何处理」，与 mask 档无关。"""
        frame = np.full(FRAME_SHAPE, 255, dtype=np.uint8)
        d = _det((421, 246, 510, 350))
        monkeypatch.setattr(detection, 'YOLO_CROP_MARGIN', 0)
        out, ox, oy, imgsz = detection._yolo_input(frame, d)
        assert out is frame and (ox, oy) == (0, 0)
        assert imgsz == detection.YOLO_IMGSZ_FULL

    def test_mode_full_returns_identity_without_masking(self):
        frame = np.full(FRAME_SHAPE, 255, dtype=np.uint8)
        d = _det((421, 246, 510, 350))
        out, ox, oy, imgsz = detection._yolo_input(frame, d, mode='full')
        assert out is frame and (ox, oy) == (0, 0)
        assert imgsz == detection.YOLO_IMGSZ_FULL
        assert out.min() == 255, 'full 档不能抹黑任何像素'

    def test_mode_argument_overrides_module_constant(self):
        """显式 mode 必须覆盖模块常量（否则 A/B 对照会静默失效）。"""
        frame = np.full(FRAME_SHAPE, 255, dtype=np.uint8)
        d = _det((421, 246, 510, 350))
        crop, ox, oy, imgsz = detection._yolo_input(frame, d, mode='crop')
        x1, y1, _, _ = detection._yolo_input_box(d.yolo_accept_box(), FRAME_SHAPE)
        assert crop.shape[0] < H and (ox, oy) == (x1, y1)
        assert imgsz == detection.YOLO_IMGSZ_CROP


class TestProbeConfFilter:
    """补检证据的置信度门槛：低分证据不进 tracker 历史（防状态机时序被扰动）。"""

    BALLS = [(1.0, 2.0, 0, 0, 0, 0, 0.9), (3.0, 4.0, 0, 0, 0, 0, 0.62),
             (5.0, 6.0, 0, 0, 0, 0, 0.49), (7.0, 8.0, 0, 0, 0, 0, 0.20)]

    def test_gate_is_meaningful(self):
        assert 0.2 < detection.YOLO_PROBE_CONF <= 1.0, \
            '门槛必须高于默认 ball_conf（0.2），否则等于不过滤'

    def test_keeps_high_conf_only(self):
        kept = detection._filter_probe_balls(self.BALLS)
        assert [round(b[6], 2) for b in kept] == [0.9, 0.62]

    def test_all_below_returns_empty(self):
        assert detection._filter_probe_balls(self.BALLS[2:]) == []

    def test_empty_and_none(self):
        assert detection._filter_probe_balls([]) == []
        assert detection._filter_probe_balls(None) == []

    def test_threshold_from_module_constant_and_param(self, monkeypatch):
        monkeypatch.setattr(detection, 'YOLO_PROBE_CONF', 0.8)
        assert [round(b[6], 2) for b in detection._filter_probe_balls(self.BALLS)] == [0.9]
        # 显式传参覆盖模块常量（A/B 调门槛时必须生效）
        assert len(detection._filter_probe_balls(self.BALLS, 0.2)) == 4


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
