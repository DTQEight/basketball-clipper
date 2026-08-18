"""GoalDetector 合成帧单元测试（纯 CPU，不依赖 YOLO/视频文件）。

用合成灰底 + 白色圆形/条形斑块驱动检测器，覆盖：
- loose 模式 + YOLO 确认进球路径
- strict visual 路径（上方→下方穿越 + 下行趋势）与 side 路径（无上方状态）
- YOLO 硬否决（有运动斑块但无球位置 → 拒绝）
- 冷却期（min_gap_sec 内重复进球被拒 + 冷却早退重置进框计数）
- 自适应阈值预热（clamp 上下限 / 噪声偏移 / 预热期不判进球）
- 圆形度过滤（长条形斑块被拒）
- has_motion_near_hoop 条件跳过判定 + ROI 复用参数
- fps 换算的时间窗口（60fps 窗口翻倍）
- 滚动基准帧（时间触发 / 斑块持续触发）
- deque 历史容器语义（本轮重构回归防护）
- 合成小视频端到端回归（VideoReader + GoalDetector 集成）
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


class TestStrictVisualPath:
    """strict 模式（loose_mode=False）：上方→下方穿越判定。"""

    def test_visual_goal_above_to_below(self):
        """斑块从上方经筐内到下方（趋势向下）→ visual 路径注册 1 球。"""
        det = _detector(loose_mode=False, yolo_confirm=False)
        goals = _feed_seq(det, [40, 40, 100, 100, 140, 140], with_yolo=False)
        assert len(det.goals) == 1
        assert det.goals == goals
        # 触发帧 = 第一个下方帧（idx=4）
        assert det.goals[0] == pytest.approx(4 / FPS)

    def test_side_goal_without_above_state(self):
        """斑块未到过上方、从筐内直接到下方 → side 路径。"""
        det = _detector(loose_mode=False, yolo_confirm=False)
        _feed_seq(det, [100, 100, 140, 140], with_yolo=False)
        assert len(det.goals) == 1
        assert det.diag["side_goal"] == 1

    def test_downward_trend_unit(self):
        """趋势检查单元钉住：尾 4 帧 y 差 ≤5 判 False（把 >5 改反此测试失败）。"""
        det = _detector(loose_mode=False, yolo_confirm=False)
        det.blob_history.extend([(0, 10.0), (1, 10.5), (2, 11.0), (3, 11.2)])
        assert det._check_downward_trend() is False
        det.blob_history.clear()
        det.blob_history.extend([(0, 10.0), (1, 20.0), (2, 20.0), (3, 30.0)])
        assert det._check_downward_trend() is True


class TestCooldownReset:
    def test_cooldown_early_return_resets_in_hoop_counter(self):
        """冷却早退必须重置进框计数（防冷却后第一帧残留计数直接触发 loose 误报）。"""
        det = _detector()
        _feed_seq(det, [40, 40, 100, 100])          # 进球 @ idx 3
        assert len(det.goals) == 1
        # 冷却期内（idx=4，gap=1/30s < 3s）喂筐内帧 → 早退分支
        det.feed(_ball_pos(100), 4, FPS, frame=_frame_with_ball(100))
        assert det.diag["reject_cooldown"] >= 1
        assert det.blob_in_hoop_frames == 0         # 修复点：不留残留计数


class TestFpsScaling:
    def test_timeouts_scale_with_fps(self):
        """按秒定义的时间窗口必须随 fps 换算（30fps 基准 45/10 帧）。"""
        det30 = GoalDetector(HOOP, min_gap_sec=3.0, diff_threshold=25,
                             rolling_baseline_sec=0, auto_threshold=False, fps=30.0)
        assert det30.above_timeout_frames == 45     # 1.5s @30fps
        assert det30.yolo_window_frames == 10       # 0.34s @30fps
        det60 = GoalDetector(HOOP, min_gap_sec=3.0, diff_threshold=25,
                             rolling_baseline_sec=0, auto_threshold=False, fps=60.0)
        assert det60.above_timeout_frames == 90     # 1.5s @60fps（旧实现仍是 45）
        assert det60.yolo_window_frames == 20

    def test_ball_frame_recorded_with_original_index(self):
        """跳帧复用时 ball_pos 用原始检测帧号记录（时间窗口语义保持）。"""
        det = _detector()
        det.feed(_ball_pos(100), frame_idx=10, fps=FPS,
                 frame=_frame_with_ball(100), ball_frame=8)
        assert det.ball_pos_history[-1][0] == 8


class TestWarmupExtended:
    def test_no_goal_registered_during_warmup(self):
        """预热期内不判定进球（把 return None 删掉此测试失败）。"""
        det = _detector(auto_threshold=True, diff_threshold=15)
        goals = _feed_seq(det, [40, 40, 100, 100, 100, 140, 140])
        assert goals == []
        assert det.goals == []
        assert len(det._warmup_p95s) > 0            # 但 P95 已采样

    def test_warmup_median_offset(self):
        """恒定 +10 灰度噪声 → 阈值 = median(P95=10)+8 = 18。"""
        det = _detector(auto_threshold=True, diff_threshold=15)
        noisy = np.full((240, 320, 3), 110, dtype=np.uint8)
        n = int(WARMUP_TARGET_SEC * FPS)
        for i in range(n + 2):
            det.feed(None, i, FPS, frame=noisy)
        assert det._warmup_done
        assert det._auto_threshold_value == 18

    def test_warmup_clamps_high(self):
        """大幅底噪（+135）→ median+8=143 → clamp 上限 50。"""
        det = _detector(auto_threshold=True, diff_threshold=15)
        bright = np.full((240, 320, 3), 235, dtype=np.uint8)
        n = int(WARMUP_TARGET_SEC * FPS)
        for i in range(n + 2):
            det.feed(None, i, FPS, frame=bright)
        assert det._warmup_done
        assert det._auto_threshold_value == 50


class TestRollingBaseline:
    def test_time_trigger_updates_baseline(self):
        """滚动间隔到达 → 基准帧更新（计数器 +1）。"""
        det = _detector(rolling_baseline_sec=60.0)
        for i in range(int(60 * FPS) + 2):
            det.feed(None, i, FPS, frame=_base_frame())
        assert det.diag["baseline_updates"] >= 1

    def test_persistent_blob_triggers_update(self):
        """斑块（球员）常驻 >5s → 提前触发基准更新，不等时间间隔。"""
        det = _detector(rolling_baseline_sec=3600.0)   # 时间触发永不满足
        for i in range(int(5.2 * FPS)):
            det.feed(None, i, FPS, frame=_frame_with_ball(100))
        assert det.diag["baseline_updates"] >= 1


class TestYoloWindowBoundaries:
    def test_ball_far_away_rejected(self):
        """球位置远离篮筐（±1 倍筐宽高之外）→ YOLO 确认失败。"""
        det = _detector()
        far_ball = (500.0, 100.0, 494.0, 94.0, 506.0, 106.0, 0.9)
        for i, cy in enumerate([40, 40, 100, 100, 100]):
            det.feed(far_ball, i, FPS, frame=_frame_with_ball(cy))
        assert det.goals == []
        assert det.diag["yolo_rejected"] >= 1

    def test_prev_frame_ball_confirms_current_trigger(self):
        """触发瞬间 YOLO 漏检、但窗口内前帧有球（筐附近）→ 仍确认（核心容错）。"""
        det = _detector()
        # 前两帧：球在筐上方（YOLO 有球，位置在筐 ±1 倍宽高内）
        det.feed(_ball_pos(45), 0, FPS, frame=_frame_with_ball(45))
        det.feed(_ball_pos(45), 1, FPS, frame=_frame_with_ball(45))
        # 后两帧：斑块进筐（第 2 帧触发 loose），但 YOLO 漏检 ball_pos=None
        det.feed(None, 2, FPS, frame=_frame_with_ball(100))
        g = det.feed(None, 3, FPS, frame=_frame_with_ball(102))  # 第2筐内帧触发
        assert g is not None
        assert det.diag["yolo_confirmed"] >= 1


class TestRoiReuse:
    def test_feed_accepts_precomputed_roi(self):
        """frame_roi 参数与内部计算结果一致（复用不改变行为）。"""
        det1, det2 = _detector(), _detector()
        seq = [40, 40, 100, 100, 140, 140]
        for i, cy in enumerate(seq):
            f = _frame_with_ball(cy)
            det1.feed(_ball_pos(cy), i, FPS, frame=f)
            det2.feed(_ball_pos(cy), i, FPS, frame=f, frame_roi=det2.compute_roi(f))
        assert det1.goals == det2.goals

    def test_has_motion_accepts_precomputed_roi(self):
        det = _detector()
        frame = _frame_with_ball(100, radius=12)
        roi = det.compute_roi(frame)
        assert det.has_motion_near_hoop(frame, frame_roi=roi) is True
        assert det.has_motion_near_hoop(frame) is True


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


class TestEndToEndVideo:
    """合成小视频 → VideoReader 解码 → GoalDetector 检测的集成回归。

    覆盖真实视频 IO 路径（mp4 有损压缩噪声 + pts 换算 + 帧号对齐），
    是"4th.mp4 漏检 4 球"这类回归在单测层的最小可复现防线。
    需要 PyAV（requirements 已含）；无编码器环境自动跳过。
    """

    def _make_video(self, tmp_path, ball_seq, still_frames=15):
        """合成 30fps mp4：静止基准段 + 球下落穿越段 + 静止收尾段。"""
        video = str(tmp_path / "e2e.mp4")
        w = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
        if not w.isOpened():
            pytest.skip("cv2 无 mp4v 编码器")
        base = np.full((240, 320, 3), 100, dtype=np.uint8)
        for _ in range(still_frames):
            w.write(base)
        for cy in ball_seq:
            f = base.copy()
            cv2.circle(f, (CX, int(cy)), 6, (255, 255, 255), -1)
            w.write(f)
        for _ in range(still_frames):
            w.write(base)
        w.release()
        return video, still_frames

    def test_synthetic_video_goal_detected(self, tmp_path):
        av = pytest.importorskip("av")
        from video_io import VideoReader
        # cy 序列：上方 4 帧 → 筐内 2 帧（第 2 帧触发）→ 下方 2 帧
        video, still = self._make_video(tmp_path, [40, 45, 50, 60, 100, 102, 140, 145])
        det = GoalDetector(HOOP, min_gap_sec=3.0, diff_threshold=25, min_blob_area=20,
                           search_margin=60, loose_mode=True, yolo_confirm=False,
                           rolling_baseline_sec=0, min_circularity=0.35,
                           auto_threshold=False, fps=30.0)
        with VideoReader(video) as r:
            for idx, frame in r.iter_frames(batch=1):
                det.feed(None, idx, r.fps, frame=frame)
        assert len(det.goals) == 1
        expected = (still + 5) / 30.0                # 第 2 个筐内帧
        assert det.goals[0] == pytest.approx(expected, abs=1.5 / 30.0)

    def test_synthetic_video_no_goal_without_ball(self, tmp_path):
        av = pytest.importorskip("av")
        from video_io import VideoReader
        # 全程静止 → 0 进球（防止"背景噪声也判进球"的回归）
        video, _ = self._make_video(tmp_path, ball_seq=[])
        det = GoalDetector(HOOP, min_gap_sec=3.0, diff_threshold=25, min_blob_area=20,
                           search_margin=60, loose_mode=True, yolo_confirm=False,
                           rolling_baseline_sec=0, min_circularity=0.35,
                           auto_threshold=False, fps=30.0)
        with VideoReader(video) as r:
            for idx, frame in r.iter_frames(batch=1):
                det.feed(None, idx, r.fps, frame=frame)
        assert det.goals == []
