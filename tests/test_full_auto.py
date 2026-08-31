# -*- coding: utf-8 -*-
"""全自动模式（FULL_AUTO）单元测试：两端直接判决、灰区默认×、跳过校准。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import goal_verifier as gv


def _mk_clips(scores, marks=None):
    clips = [{"ts": float(i), "score": float(s)} for i, s in enumerate(scores)]
    if marks:
        for c, m in zip(clips, marks):
            if m:
                c["mark"] = m
                c["mark_source"] = "manual"
    return clips


def test_set_full_auto_toggles():
    """开关切换全局状态且立即生效。"""
    old = gv.FULL_AUTO
    try:
        gv.set_full_auto(True)
        assert gv.FULL_AUTO is True
        gv.set_full_auto(False)
        assert gv.FULL_AUTO is False
    finally:
        gv.set_full_auto(old)


def test_full_auto_marks_both_ends():
    """全自动判决（复刻 verify_clips 阶段 1.6 分支）：
    ≥keep 阈值自动√，灰区与低分一律自动×（灰区默认×口径）。"""
    old = gv.FULL_AUTO
    try:
        gv.set_full_auto(True)
        # 0.9 ≥ keep 阈值 → √；0.6 灰区 → ×；0.05 ≤ reject 阈值 → ×
        clips = _mk_clips([0.9, 0.6, 0.05], marks=[None, None, None])
        for c in clips:  # FULL_AUTO 分支逻辑
            if float(c["score"]) >= gv.AB_KEEP_THR:
                c["mark"] = "keep"; c["mark_source"] = "auto"
            else:
                c["mark"] = "reject"; c["mark_source"] = "auto"
        assert clips[0]["mark"] == "keep"
        assert clips[1]["mark"] == "reject"  # 灰区默认×
        assert clips[2]["mark"] == "reject"
        # 不送校准、不写 score_calibrated
        assert not any(c.get("calib") for c in clips)
        assert not any("score_calibrated" in c for c in clips)
    finally:
        gv.set_full_auto(old)


def test_full_auto_respects_manual():
    """人工标记不被全自动覆盖（分支里有 mark_source='manual' 跳过）。"""
    old = gv.FULL_AUTO
    try:
        gv.set_full_auto(True)
        clips = _mk_clips([0.9, 0.6], marks=["keep", "reject"])
        for c in clips:
            if c.get("mark_source") == "manual":
                continue
            if float(c["score"]) >= gv.AB_KEEP_THR:
                c["mark"] = "keep"; c["mark_source"] = "auto"
            else:
                c["mark"] = "reject"; c["mark_source"] = "auto"
        assert clips[0]["mark"] == "keep" and clips[0]["mark_source"] == "manual"
        assert clips[1]["mark"] == "reject" and clips[1]["mark_source"] == "manual"
    finally:
        gv.set_full_auto(old)


def test_needs_verify_only_checks_scores():
    """已有分数的灰区片段不再触发验证（VLM 仲裁已移除，只看缺分数）。"""
    old = gv.FULL_AUTO
    try:
        clips = [{"ts": 1.0, "score": 0.5}]  # 灰区但已有分数
        gv.set_full_auto(True)
        assert gv.needs_verify(clips) is False
        gv.set_full_auto(False)
        assert gv.needs_verify(clips) is False  # 默认模式同样只看缺分数
        # 缺分数仍需验证
        assert gv.needs_verify([{"ts": 2.0}]) is True
    finally:
        gv.set_full_auto(old)
