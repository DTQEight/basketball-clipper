# -*- coding: utf-8 -*-
"""③ 轨迹外推 + ④ 篮网/筐框光流：A 臂的补充特征（**已并入 A 臂**）。

来源：2026-09-27 的可行性实验
  转储：cache/_probe_traj_net.py   → cache/_traj_net_dump.jsonl
  分析：cache/_analyze_traj_net.py

**接入方式**：training/extract_features.py 的 extract() 在同一个逐帧循环里调用
  traj_feats() / net_feats()，与原有 35 维合并成一个 dict（共 56 维）。
生产侧 services/goal_verifier._score_lgbm() 通过 extract() 自动拿到这些列，
不留任何额外分支。**新增/改动本文件的特征都会改变 model_fingerprint**。

=== 实验结论（404 事件 / 81 视频，GroupKFold 按视频，5 seeds）===
    A 臂 35 维基线          OOF AUC 0.7771
    + ③（9 维）             0.8022   +0.0251   95%CI [+0.004, +0.038] ✓
    + ④（12 维）            0.8164   +0.0393   95%CI [+0.016, +0.069] ✓
    + ③+④（21 维）          0.8307   +0.0536   95%CI [+0.028, +0.078] ✓
    对照：+9 列纯噪声        0.7557   −0.0214（增益不是「加列虚高」）

  跨机位域（3 场 bba 半场，训练在底线机位域、测试在半场域）：
    A 臂 0.895 → 0.926（+0.026），域外召回@0.68 0.633 → 0.775（+14.2pp）
    代入集成后：召回 +2.4pp，集成 AUC 持平

=== 为什么不单独做「第 5 臂」===
  实测（cache/_fifth_arm.py）：⑤ 永远是全场最弱的臂（域内 0.73 / 域外 0.80），
  加权进 AUC 0.99 的集成只能稀释；且两个域要的权重相反（域内 ≤0.25、域外 1.0）。
  ⇒ 并入 A 臂是唯一两域都为正的形态。
  ⇒ 但「几何量不吃机位」这条规律成立：⑤_几何 是系统里唯一域外比域内更强的信号
     （0.732 → 0.799），而 ④ 的网区图像统计量会退化（f_flow_ratio 0.657 → 0.517）。

⚠ 特征维度变化 → model_fingerprint 变 → 历史 AI 分数全部作废重算，且必须重训 A 臂。
"""
from __future__ import annotations

import numpy as np

# 特征名清单（保持与实验完全一致的顺序，便于与历史 dump 对拍）
TRAJ_FEATS = ("ext_no_traj", "ext_off_q", "ext_in_q", "ext_off_l", "ext_in_l",
              "ext_fit_rmse", "ext_n_pre", "ext_vy_desc", "ext_dist_at_last")

NET_FEATS = ("n_diff_peak_post", "n_diff_ratio", "n_diff_mean_post",
             "n_diff_peak_over_mean", "f_flow_peak_post", "f_flow_ratio",
             "f_flow_mean_post", "f_flow_argmax_rel", "f_flow_peak_over_mean",
             "f_flow_tail_decay", "r_diff_peak_ratio", "r_flow_peak_ratio")

# ④ 的光流用 Farneback（小 ROI，CPU，开销可忽略）
FARNEBACK = dict(flags=0, pyr_scale=0.5, levels=3, winsize=15,
                 iterations=3, poly_n=5, poly_sigma=1.2)


def roi_rects(hoop, frame_w, frame_h, scale=2):
    """由篮筐框推出两个 ROI（缩小坐标系下的 (x1,y1,x2,y2)）。

    net：网区，左右各扩 0.25 筐宽、下沿延伸 2.0 筐高（与 extract_features.py 一致）。
    rim：筐框本身，用于捕捉篮圈振动——A 臂 35 维里**完全没有**这个信号。
    """
    hx1, hy1, hx2, hy2 = [float(v) for v in hoop]
    hw, hh = max(hx2 - hx1, 1.0), max(hy2 - hy1, 1.0)
    sx1 = int(max(0.0, hx1 - 0.25 * hw) / scale)
    sx2 = int(min(float(frame_w), hx2 + 0.25 * hw) / scale)
    sy1 = int(max(0.0, hy2) / scale)
    sy2 = int(min(hy2 + 2.0 * hh, float(frame_h)) / scale)
    rx1 = int(max(0.0, hx1) / scale)
    rx2 = int(min(float(frame_w), hx2) / scale)
    ry1 = int(max(0.0, hy1) / scale)
    ry2 = int(min(float(hy2), float(frame_h)) / scale)
    return (sx1, sy1, sx2, sy2), (rx1, ry1, rx2, ry2)


def farneback_mag(prev_roi, cur_roi) -> float:
    """两帧灰度 ROI 的 Farneback 光流平均幅值。"""
    try:
        import cv2
        flow = cv2.calcOpticalFlowFarneback(prev_roi, cur_roi, None, **FARNEBACK)
    except Exception:
        return 0.0
    return float(np.hypot(flow[..., 0], flow[..., 1]).mean())


def roi_diff(a, b) -> float:
    """两个灰度 ROI 的平均绝对帧差。"""
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


# ---------------------------------------------------------------- ③ 轨迹外推
def traj_feats(detections, hoop, n_frames) -> dict:
    """用「球还没进筐带」的检测点外推轨迹，判断是否指向筐口。

    因果设计：只用入筐带上沿之前的点拟合，完全不用入筐之后的帧——因此它比 A 臂
    已有的 min_dist_center（事后最近距离）严格信息更少，却仍带来 +0.025 增量。

    detections: [(t_rel, cx, cy, x1, y1, x2, y2, conf)] 按时间序（与 extract_features 同构）
    hoop: [hx1, hy1, hx2, hy2]（像素，原始分辨率）
    """
    f = {k: 0.0 for k in TRAJ_FEATS}
    f["ext_no_traj"] = 1.0
    if not detections:
        return f
    hx1, hy1, hx2, hy2 = [float(v) for v in hoop]
    hw = max(hx2 - hx1, 1.0)
    hh = max(hy2 - hy1, 1.0)
    cx_h, cy_h = (hx1 + hx2) / 2, (hy1 + hy2) / 2
    if len(detections) < 4:
        return f

    t = np.array([d[0] for d in detections], dtype=float)
    x = np.array([d[1] for d in detections], dtype=float)
    y = np.array([d[2] for d in detections], dtype=float)

    above = np.where(y < hy1)[0]
    if len(above) < 3:
        above = np.where(y < cy_h)[0]      # 退一步：筐心以上
    if len(above) < 3:
        return f
    i_cut = int(above[-1]) + 1
    tt, xx, yy = t[:i_cut], x[:i_cut], y[:i_cut]
    f["ext_no_traj"] = 0.0
    f["ext_n_pre"] = len(tt) / max(n_frames, 1)
    if len(tt) >= 2:
        f["ext_vy_desc"] = float(np.mean(np.diff(yy) > 0))
    f["ext_dist_at_last"] = float(np.hypot(xx[-1] - cx_h, yy[-1] - cy_h) / hw)

    def _cross_time(poly):
        """解 poly(t)=cy_h，取大于 tt[-1] 的最小实根（即球到达筐心高度的时刻）。"""
        roots = np.roots(list(poly) + [-cy_h])
        real = [float(r.real) for r in roots if abs(r.imag) < 1e-6 and r.real > tt[-1]]
        return min(real) if real else None

    # (a) x 线性 + y 二次（贴合抛体运动）
    if len(tt) >= 3:
        ax = np.polyfit(tt, xx, 1)
        ay = np.polyfit(tt, yy, 2)
        f["ext_fit_rmse"] = float(np.sqrt(((yy - np.polyval(ay, tt)) ** 2).mean()) / hh)
        ts_ = _cross_time(ay)
        if ts_ is not None:
            xp = float(np.polyval(ax, ts_))
            f["ext_off_q"] = float(abs(xp - cx_h) / hw)
            f["ext_in_q"] = 1.0 if hx1 <= xp <= hx2 else 0.0
    # (b) 全线性（短窗口更稳，作为对照）
    ax = np.polyfit(tt, xx, 1)
    ay1 = np.polyfit(tt, yy, 1)
    ts_ = _cross_time(ay1)
    if ts_ is not None:
        xp = float(np.polyval(ax, ts_))
        f["ext_off_l"] = float(abs(xp - cx_h) / hw)
        f["ext_in_l"] = 1.0 if hx1 <= xp <= hx2 else 0.0
    return f


# ---------------------------------------------------------------- ④ 篮网光流
def net_feats(net_small, rim_small, fps, pre_end_idx, post_start_idx) -> dict:
    """网区/筐框的逐帧帧差与光流 → 峰值、相对基线比值、峰值时刻、尾段衰减。

    net_small / rim_small: [[diff, flow], ...] 逐帧（与窗口帧一一对应）
    pre_end_idx:  基线窗口右端（ts-0.5s 处）
    post_start_idx: 后窗起点（ts 处）
    """
    f = {k: 0.0 for k in NET_FEATS}
    net = np.asarray(net_small, dtype=float)
    rim = np.asarray(rim_small, dtype=float)
    if net.ndim != 2 or len(net) < 4 or net.shape[1] < 2:
        return f
    pre = slice(max(0, int(pre_end_idx) - int(fps)), max(1, int(pre_end_idx)))
    post = slice(int(post_start_idx), len(net))
    if post.start >= post.stop:
        return f
    nd, nf = net[:, 0], net[:, 1]
    rd, rf = rim[:, 0], rim[:, 1]

    pre_d = float(np.mean(nd[pre])) if pre.stop > pre.start else 0.0
    pre_f = float(np.mean(nf[pre])) if pre.stop > pre.start else 0.0
    pre_rd = float(np.mean(rd[pre])) if pre.stop > pre.start else 0.0
    pre_rf = float(np.mean(rf[pre])) if pre.stop > pre.start else 0.0

    f["n_diff_peak_post"] = float(np.max(nd[post]))
    f["n_diff_mean_post"] = float(np.mean(nd[post]))
    f["n_diff_ratio"] = f["n_diff_peak_post"] / max(pre_d, 0.5)
    f["n_diff_peak_over_mean"] = f["n_diff_peak_post"] / max(f["n_diff_mean_post"], 0.5)

    f["f_flow_peak_post"] = float(np.max(nf[post]))
    f["f_flow_mean_post"] = float(np.mean(nf[post]))
    f["f_flow_ratio"] = f["f_flow_peak_post"] / max(pre_f, 0.5)
    f["f_flow_peak_over_mean"] = f["f_flow_peak_post"] / max(f["f_flow_mean_post"], 0.05)

    k = int(np.argmax(nf[post])) + post.start
    f["f_flow_argmax_rel"] = (k - int(post_start_idx)) / max(fps, 1.0)

    # 尾段衰减：真进球是「尖峰后回落」；持续运动（争抢篮板、镜头摇动）不会回落
    tail0 = max(post.start, len(nf) - int(0.5 * fps))
    f["f_flow_tail_decay"] = float(np.mean(nf[tail0:])) / max(f["f_flow_peak_post"], 0.05)

    f["r_diff_peak_ratio"] = float(np.max(rd[post])) / max(pre_rd, 0.5)
    f["r_flow_peak_ratio"] = float(np.max(rf[post])) / max(pre_rf, 0.5)
    return f
