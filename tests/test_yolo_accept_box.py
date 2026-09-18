# -*- coding: utf-8 -*-
"""YOLO 硬否决接受范围的测试。

背景：实测有真实进球因球心落在「篮筐框 ± 1 倍自身尺寸」之外而被否决，
导致漏检（见 2026.09.01-1st 的 5:43 / 9:20）。现改为：
    横向 1.5 倍、向上 1.0 倍、向下 2.0 倍（向下额外放宽——进球是单向
    过程，球必定穿过筐口落到下方）。

这些用例用真实观测到的球心坐标做回归保护。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker import GoalDetector  # noqa: E402

HOOP = (1360, 338, 1426, 424)   # w=66, h=86  ->  x∈[1261,1525]  y∈[252,596]


def _det(yolo_confirm=True):
    return GoalDetector(hoop_box=HOOP, yolo_confirm=yolo_confirm,
                        loose_mode=True)


def _check(ball_xy, yolo_confirm=True):
    d = _det(yolo_confirm)
    if ball_xy is not None:
        d.ball_pos_history.append((0, ball_xy[0], ball_xy[1]))
    return d._check_yolo_near_hoop()


@pytest.mark.parametrize('ball,label', [
    ((1390, 380), '筐框正中心'),
    ((1295.7, 519.7), '5:43 实测球心：筐左下，原 1.0 倍外扩时被拒'),
    ((1267.5, 520.8), '9:20 实测球心：更靠左，原 x/y 双超界'),
    ((1360, 590), '筐下 1.9 倍筐高'),
    ((1400, 260), '筐上 0.9 倍筐高'),
    ((1265, 300), '筐左 1.44 倍筐宽'),
])
def test_accepts_ball_in_enlarged_box(ball, label):
    ok, status = _check(ball)
    assert ok is True, '应通过：%s' % label
    assert status == 'confirmed'


@pytest.mark.parametrize('ball,label', [
    ((1240, 520), '筐左 1.8 倍筐宽，超出接受范围'),
    ((1300, 620), '筐下 2.3 倍筐高，超出接受范围'),
    ((1300, 200), '筐上 1.6 倍筐高，超出接受范围'),
    ((1600, 400), '筐右 2.6 倍筐宽，超出接受范围'),
])
def test_rejects_ball_outside_box(ball, label):
    ok, status = _check(ball)
    assert ok is False, '应否决：%s' % label
    assert status == 'rejected'


def test_no_ball_rejected():
    ok, status = _check(None)
    assert ok is False and status == 'rejected'


def test_yolo_confirm_off_always_passes():
    """yolo_confirm=False 时不做球确认，永远放行（纯 diff 模式）。"""
    ok, status = _check(None, yolo_confirm=False)
    assert ok is True and status == 'skipped'


def test_downward_margin_wider_than_upward():
    """向下必须比向上宽松：球穿过筐口后必然落到筐下方。"""
    d = _det()
    assert d.hoop_h > 0
    below = 424 + int(d.hoop_h * 2.0) - 1     # 向下边界内侧
    above = 338 - int(d.hoop_h * 1.0) + 1     # 向上边界内侧
    assert _check((1400, below))[0] is True
    assert _check((1400, above))[0] is True
    # 向下 2.0 倍内的点应通过，而向上 2.0 倍处的点应被拒
    assert _check((1400, 424 + int(d.hoop_h * 1.9)))[0] is True
    assert _check((1400, 338 - int(d.hoop_h * 1.5)))[0] is False
