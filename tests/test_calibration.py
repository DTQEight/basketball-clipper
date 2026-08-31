# -*- coding: utf-8 -*-
"""每场自适应校准（calibrate_clips / apply_calibration）单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import goal_verifier as gv


def _mk_clips(scores, marks=None):
    """构造 clips：score 按序，marks 指定前 len(marks) 个的 mark。"""
    clips = [{"ts": float(i), "score": float(s)} for i, s in enumerate(scores)]
    if marks:
        for c, m in zip(clips, marks):
            if m:
                c["mark"] = m
                c["mark_source"] = "manual"
    return clips


def test_calib_marks_top_n():
    """verify_clips 阶段的挑选逻辑等价实现：高分前 N 标 need（灰区内）。"""
    clips = _mk_clips([0.9, 0.75, 0.6, 0.3, 0.02])
    # 模拟阶段 1.6 的挑选（只挑 0.10 < s < 0.70 的前 5 个高分）
    scored = [c for c in clips if "score" in c]
    for c in scored:  # 先按阈值判决（未校准：只 ×）
        s = c["score"]
        if s <= gv.REJECT_THR:
            c["mark"] = "reject"
    cand = sorted(scored, key=lambda c: -c["score"])[:gv.CALIB_N]
    for c in cand:
        if gv.REJECT_THR < c["score"] < gv.AB_KEEP_THR:
            c["calib"] = "need"
    need = [c for c in clips if c.get("calib") == "need"]
    # 0.9/0.75 已过 √ 阈值（若已校准会自动√）、0.02 已过 × 阈值，都不送校准
    assert [round(c["score"], 2) for c in need] == [0.6, 0.3], need


def test_calibrate_shift_up():
    """悍高 4th 场景：全场分数下移，前 5 确认 3 个真进球 → 平移量正确。"""
    # 排序后 [0.55, 0.50, 0.45, 0.40, 0.35, ...]，前 5 高分含 3 真 2 误
    scores = [0.55, 0.50, 0.45, 0.40, 0.35] + [0.3] * 10
    marks = ["keep", "keep", "reject", "keep", "reject"]
    clips = _mk_clips(scores, marks)
    for c in clips[:5]:
        c["calib"] = "need"
    shift = gv.calibrate_clips(clips, (3, 5))
    # 尾巴分 = 第 (5-3)=2 名 = 0.45；shift = 0.90-0.45 = 0.45 → 封顶 0.25
    assert shift == 0.25, shift
    assert clips[0]["calib_shift"] == 0.25
    # need 标记被消化
    assert not any(c.get("calib") for c in clips)


def test_calibrate_no_shift_when_in_distribution():
    """训练分布内（高分 0.85+）：shift < 0.05 → None 且不写 calib_shift。"""
    scores = [0.88, 0.85, 0.8, 0.2, 0.1]
    marks = ["keep", "keep", "keep", "reject", "reject"]
    clips = _mk_clips(scores, marks)
    for c in clips[:5]:
        c["calib"] = "need"
    shift = gv.calibrate_clips(clips, (3, 5))
    # 尾巴分 = 第 2 名 = 0.85 → shift = 0.05 → < CALIB_MIN_GAIN(0.05)? 0.90-0.85=0.05 不小于 0.05
    # 边界值：恰好等于 → 触发平移。改为更明确的分布内用例：
    scores = [0.95, 0.92, 0.90, 0.2, 0.1]
    clips = _mk_clips(scores, marks)
    for c in clips[:5]:
        c["calib"] = "need"
    shift = gv.calibrate_clips(clips, (3, 5))
    assert shift is None, shift
    assert "calib_shift" not in clips[0]


def test_calibrate_all_rejected():
    """前 5 全否（模型排序完全失灵）→ 不平移。"""
    clips = _mk_clips([0.5, 0.45, 0.4, 0.35, 0.3], ["reject"] * 5)
    for c in clips[:5]:
        c["calib"] = "need"
    shift = gv.calibrate_clips(clips, (0, 5))
    assert shift is None


def test_apply_calibration_promotes():
    """平移后重判：灰区高分被提为自动 √，人工标记不被覆盖。"""
    clips = _mk_clips([0.60, 0.55, 0.30])
    clips[0]["manual_guard"] = True
    clips[0]["calib_shift"] = 0.25
    # 模拟人工标记第三个
    clips[2]["mark"] = "reject"
    clips[2]["mark_source"] = "manual"
    nk, nr = gv.apply_calibration("fake.mp4", clips, None)
    # 0.60+0.25=0.85 ≥ 0.70 自动√；0.55+0.25=0.80 自动√；0.30+0.25=0.55 灰区
    assert nk == 2, nk
    assert clips[0]["mark"] == "keep" and clips[0]["mark_source"] == "auto"
    assert clips[1]["mark"] == "keep"
    # 人工 reject 不被覆盖
    assert clips[2]["mark"] == "reject" and clips[2]["mark_source"] == "manual"
    assert nr == 0  # 人工标记不计入自动 ×
    # 校准分已写
    assert clips[0]["score_calibrated"] == 0.85


def test_apply_calibration_caps_at_1():
    clips = _mk_clips([0.95])
    clips[0]["calib_shift"] = 0.25
    gv.apply_calibration("fake.mp4", clips, None)
    assert clips[0]["score_calibrated"] == 1.0


def test_verify_clips_two_phase():
    """端到端两阶段：未校准只自动×+标 need；确认后平移+自动√。"""
    clips = [{"ts": float(i), "score_lgbm": s, "score_b": s}
             for i, s in enumerate([0.9, 0.6, 0.55, 0.05])]
    # 阶段 1：只应自动 ×（0.05），0.9 不自动 √（未校准），0.6/0.55 标 need
    shift = gv._get_calib_shift(clips)
    assert shift is None
    scored = [c for c in clips if "score" in c or "score_lgbm" in c]
    for c in scored:  # 复刻 verify_clips 阶段 1.6 的逻辑
        s = c["score_lgbm"]  # AB = (lgbm+b)/2 同值
        c["score"] = round((s + s) / 2, 3)
    scored = [c for c in clips if "score" in c]
    for c in scored:
        s = c["score"]
        if shift is not None and s >= gv.AB_KEEP_THR:
            c["mark"] = "keep"
        elif s <= gv.REJECT_THR:
            c["mark"] = "reject"
    assert clips[3]["mark"] == "reject"
    assert "mark" not in clips[0]  # 未校准不自动 √
    # 阶段 2：用户确认前 2 高分片段（0.9=进球, 0.6=进球）
    clips[0]["mark"] = "keep"; clips[0]["mark_source"] = "manual"
    clips[1]["mark"] = "keep"; clips[1]["mark_source"] = "manual"
    for c in clips[:2]:
        c["calib"] = "need"
    # 尾巴 = 第 (2-2)=0 名 = 0.9 → shift = 0 → 无需平移（本就分布内）
    # 换场景：0.6 全确认进球，尾巴 0.6 → shift = 0.30 → 封顶 0.25
    clips2 = [{"ts": 0.0, "score": 0.6, "calib": "need"},
              {"ts": 1.0, "score": 0.05}]
    clips2[0]["mark"] = "keep"; clips2[0]["mark_source"] = "manual"
    shift = gv.calibrate_clips(clips2, (1, 1))
    assert shift == 0.25
    gv.apply_calibration("fake.mp4", clips2, None)
    # 0.6+0.25 = 0.85 但人工已标 keep（manual 优先，apply 跳过）
    assert clips2[0]["mark"] == "keep"
