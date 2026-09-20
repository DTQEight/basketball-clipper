# -*- coding: utf-8 -*-
"""YOLO 球框收集 / 写入历史的回归测试。

背景（2026.08.15-1st 7:35~10:32 三分钟无候选的逐帧归因）：
旧实现 `detection.py` 用 `best = int(np.argmax(confs))` 只保留该帧**最高分**的
一个球框。该画面上有一个远端稳定误报（约 (1738,553)，置信度常年 0.5~0.69），
在真球穿过筐口的瞬间以微弱优势胜过筐边真球（0.68 vs 0.47、0.62 vs 0.55），
于是 ball_pos_history 里只有远端那个点 → _check_yolo_near_hoop 连续判否 →
斑块轨迹完整（上沿→下沿）的真进球被 YOLO 硬否决。

修复：保留全部球框，由 tracker.feed 逐个写入历史——`_check_yolo_near_hoop`
关心的是「筐邻域有没有球」，任何一个检出的球落在邻域内都应算数。
"""
import numpy as np
import pytest

from services import detection
from tracker import GoalDetector

FPS = 30
HOOP = (130, 80, 170, 120)      # w=40 h=40 -> 接受范围 x∈[70,230] y∈[40,280]
CX = 150

FAR_FP = (1737.0, 553.0, 1730.0, 546.0, 1744.0, 560.0, 0.68)   # 远端稳定误报（分更高）
NEAR_BALL = (150.0, 110.0, 145.0, 105.0, 155.0, 115.0, 0.47)   # 筐边真球（分更低）


class _Tensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(conf)

    def __len__(self):
        return len(self.conf.numpy())


class _Res:
    """最小 YOLO 结果桩（只需要 boxes.xyxy / boxes.conf）。"""

    def __init__(self, boxes=()):
        if not boxes:
            self.boxes = None
            return
        xyxy = [[b[0] - b[2], b[1] - b[3], b[0] + b[2], b[1] + b[3]]
                for b in boxes]                      # (cx, cy, hw, hh) -> xyxy
        self.boxes = _Boxes(xyxy, [b[4] for b in boxes])


def _base_frame():
    return np.full((240, 320, 3), 100, dtype=np.uint8)


def _det():
    return GoalDetector(hoop_box=HOOP, baseline_frame=_base_frame(),
                        min_gap_sec=3.0, diff_threshold=25, min_blob_area=20,
                        search_margin=60, loose_mode=True, yolo_confirm=True,
                        rolling_baseline_sec=0, min_circularity=0.35,
                        min_in_hoop_frames=2, auto_threshold=False)


# ===== detection._ball_boxes_from_result =====

def test_keeps_every_box_sorted_by_conf():
    """全部框都要保留，且按置信度降序（第一个 = 主球，供跳帧复用读取）。"""
    res = _Res(boxes=[(400.0, 80.0, 10.0, 10.0, 0.30),
                      (1730.0, 553.0, 7.0, 7.0, 0.62)])
    out = detection._ball_boxes_from_result(res)
    assert len(out) == 2
    assert out[0][6] == pytest.approx(0.62)
    assert (out[0][0], out[0][1]) == (pytest.approx(1730.0), pytest.approx(553.0))
    assert (out[1][0], out[1][1]) == (pytest.approx(400.0), pytest.approx(80.0))


def test_no_boxes_returns_empty():
    assert detection._ball_boxes_from_result(_Res()) == []


# ===== tracker.feed 逐个写入历史 =====

def test_feed_writes_all_ball_positions():
    d = _det()
    d.feed([FAR_FP, NEAR_BALL], 0, FPS, frame=_base_frame())
    assert len(d.ball_pos_history) == 2
    assert {h[0] for h in d.ball_pos_history} == {0}


def test_feed_none_writes_nothing():
    d = _det()
    d.feed(None, 0, FPS, frame=_base_frame())
    assert len(d.ball_pos_history) == 0


# ===== 根因回归：筐边真球不被远端误报挤掉 =====

def test_near_ball_not_stolen_by_far_false_positive():
    """08.15-1st 8:14 回归：远端误报分更高时，筐边真球仍必须能确认。"""
    d = _det()
    d.feed([FAR_FP, NEAR_BALL], 0, FPS, frame=_base_frame())
    ok, status = d._check_yolo_near_hoop()
    assert ok is True, '筐边有球就该确认'
    assert status == 'confirmed'


def test_only_far_false_positive_still_rejected():
    """对照：只有远端误报（旧实现丢掉了筐边真球后的实际历史）→ 仍应否决。

    说明接受范围没有被放宽——远端那个点本来就在邻域之外。
    """
    d = _det()
    d.feed([FAR_FP], 0, FPS, frame=_base_frame())
    ok, status = d._check_yolo_near_hoop()
    assert ok is False
    assert status == 'rejected'
