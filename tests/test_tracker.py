"""GoalDetector 合成帧单元测试（纯 CPU，不依赖 YOLO/视频文件）。

用合成灰底 + 白色圆形/条形斑块驱动检测器，覆盖：
- loose 模式 + YOLO 确认进球路径
- YOLO 硬否决（有运动斑块但无球位置 → 拒绝）
- 冷却期（min_gap_sec 内重复进球被拒）
- 自适应阈值预热（30s 样本计数完成 → median(P95)+8 clamp[8,50]）
- 圆形度过滤（长条形斑块被拒）
- has_motion_near_hoop 条件跳过判定
- deque 历史容器语义（本轮重构回归防护）
"""
import cv2
import numpy as np
import pytest

from tracker import GoalDetector, WARMUP_TARGET_SEC

FPS = 30
HOOP = (130, 80, 170, 120)   # x1, y1, x2, y2（40x40 篮筐）
CX = 150                      # 篮筐中心 x


def _base_frame():
    """无球基准帧：均匀灰底。"""
    return np.full((240, 320, 3), 100, dtype=np.uint8)


def _frame_with_ball(cy, radius=6):
    f = _base_frame()
    cv2.circle(f, (CX, int(cy)), radius, (255, 255, 255), -1)
    return f


def _frame_with_bar(cy, w=56, h=4):
    """细长条形斑块：模糊+形态学处理后圆形度仍 ~0.2，低于 0.35 阈值。"""
    f = _base_frame()
    cv2.rectangle(f, (CX - w // 2, int(cy) - h // 2),
                  (CX + w // 2, int(cy) + h // 2), (255, 255, 255), -1)
    return f


def _ball_pos(cy, radius=6):
    return (float(CX), float(cy), float(CX - radius), float(cy - radius),
            float(CX + radius), float(cy + radius), 0.9)


def _detector(**kw):
    """宽松模式 + YOLO 确认 + 固定阈值 + 关闭滚动基准（基准帧全程稳定）。"""
    params = dict(hoop_box=HOOP, baseline_frame=_base_frame(),
                  min_gap_sec=3.0, diff_threshold=25, min_blob_area=20,
                  search_margin=60, loose_mode=True, yolo_confirm=True,
                  rolling_baseline_sec=0, min_circularity=0.35,
                  min_in_hoop_frames=2, auto_threshold=False)
    params.update(kw)
    return GoalDetector(**params)


def _feed_seq(det, cys, start_idx=0, with_yolo=True):
    """按 cy 序列喂帧，返回每个进球时间戳的列表。"""
    goals = []
    for i, cy in enumerate(cys):
        f = _frame_with_ball(cy)
        bp = _ball_pos(cy) if with_yolo else None
        g = det.feed(bp, start_idx + i, FPS, frame=f)
        if g is not None:
            goals.append(g)
    return goals


class TestGoalPaths:
    def test_loose_goal_with_yolo_confirm(self):
        """球经过筐内 2 帧且 YOLO 有球 → 注册 1 个进球。"""
        det = _detector()
        # cy: 上方2帧 → 筐内3帧（第2帧即触发）→ 下方2帧
        goals = _feed_seq(det, [40, 40, 100, 100, 100, 140, 140])
        assert len(det.goals) == 1
        assert det.goals == goals
        assert det.goals[0] == pytest.approx(3 / FPS)  # 第2个筐内帧（idx=3）
        assert det.diag["yolo_confirmed"] >= 1

    def test_yolo_reject_no_ball_pos(self):
        """有运动斑块但 YOLO 始终无球 → loose 触发被硬否决，0 进球。"""
        det = _detector()
        _feed_seq(det, [40, 40, 100, 100, 100, 140, 140], with_yolo=False)
        assert det.goals == []
        assert det.diag["yolo_rejected"] >= 1

    def test_cooldown_blocks_second_goal(self):
        """冷却期内重复进球被拒，超过 min_gap_sec 后可再次注册。"""
        det = _detector(min_gap_sec=3.0)
        _feed_seq(det, [40, 40, 100, 100])                      # 进球 @ idx 3
        assert len(det.goals) == 1
        # 立即再来一次（仍在冷却期）→ 被拒
        _feed_seq(det, [100, 100, 100], start_idx=5)
        assert len(det.goals) == 1
        assert det.diag["reject_cooldown"] >= 1
        # 3 秒后（帧号跳到 100）→ 第二个进球
        _feed_seq(det, [100, 100, 100], start_idx=100)
        assert len(det.goals) == 2
        assert det.goals[1] == pytest.approx(101 / FPS)  # idx=101 第2个筐内帧


class TestShapeFilter:
    def test_bar_blob_rejected_by_circularity(self):
        """长条形斑块（圆形度 ~0.3 < 0.35）被形状过滤，不产生进球。"""
        det = _detector()
        for i, cy in enumerate([60, 80, 100, 100, 100, 120]):
            g = det.feed(_ball_pos(cy), i, FPS, frame=_frame_with_bar(cy))
            assert g is None
        assert det.diag["reject_shape"] >= 1
        assert det.goals == []


class TestAutoThreshold:
    def test_warmup_completes_and_clamps(self):
        """30s 静止画面预热：阈值 = median(P95=0)+8 = 8（下限 clamp）。"""
        det = _detector(auto_threshold=True, diff_threshold=15)
        n_need = int(WARMUP_TARGET_SEC * FPS)
        base = _base_frame()
        for i in range(n_need + 5):
            det.feed(None, i, FPS, frame=base.copy())
        assert det._warmup_done
        assert det._auto_threshold_value == 8
        assert det._warmup_sample_count == n_need


class TestConditionalSkip:
    def test_has_motion_near_hoop(self):
        """无运动 → False（可跳过 YOLO）；大球经过 → True。"""
        det = _detector()
        assert det.has_motion_near_hoop(_base_frame()) is False
        assert det.has_motion_near_hoop(_frame_with_ball(100, radius=12)) is True


class TestDequeHistory:
    def test_blob_history_bounded_and_reset(self):
        """blob_history 为有界 deque，注册进球后清空。"""
        det = _detector()
        _feed_seq(det, [40, 40, 100, 100])
        assert len(det.goals) == 1
        assert len(det.blob_history) == 0          # _register_goal 清空
        # 连续喂 40 帧上方斑块，deque(maxlen=30) 自动淘汰
        _feed_seq(det, [40] * 40, start_idx=200)
        assert len(det.blob_history) == 30

    def test_ball_pos_history_window(self):
        """ball_pos_history 只保留最近 yolo_window_frames 帧。"""
        det = _detector()
        for i in range(50):
            det.feed(_ball_pos(40), i, FPS, frame=_frame_with_ball(40))
        cutoff = 49 - det.yolo_window_frames
        assert det.ball_pos_history[0][0] >= cutoff
