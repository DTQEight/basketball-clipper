# -*- coding: utf-8 -*-
"""方向3 运动表征抽取：光流场（3b）+ 球轨迹渲染图（3a）。

--flow：复用已有 frames_b 筐心帧块，相邻帧 Farneback 光流 →
        flow_b/{eid}.npz（15 帧幅度图 + 汇总统计），零解码成本。
--traj：读 tracks_c 球轨迹（extract_frames_c 产出），按篮筐归一化坐标
        渲染 224×224 轨迹图 → traj_c/{eid}.npz（跨场馆坐标不变）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.extract_frames_b import load_dataset_events  # noqa: E402

FRAMES_B = PROJECT_ROOT / "training" / "frames_b"
TRACKS_C = PROJECT_ROOT / "training" / "tracks_c"
FLOW_OUT = PROJECT_ROOT / "training" / "flow_b"
TRAJ_OUT = PROJECT_ROOT / "training" / "traj_c"

MAG_SCALE = 12.0   # 光流幅度→uint8 缩放（典型 0-20px 位移）
CANVAS = 224
RANGE = 2.5        # 轨迹画布覆盖 ±2.5 倍筐高


def run_flow():
    FLOW_OUT.mkdir(parents=True, exist_ok=True)
    events = load_dataset_events()
    todo = [e for e in events
            if (FRAMES_B / f"{e['event_id']}.npz").exists()
            and not (FLOW_OUT / f"{e['event_id']}.npz").exists()]
    print(f"flow: events={len(events)} 待抽={len(todo)}", flush=True)
    t0 = time.time()
    for i, ev in enumerate(todo):
        eid = ev["event_id"]
        x = np.load(FRAMES_B / f"{eid}.npz")["x"]  # (16,224,224,3) BGR
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in x]
        mags = np.zeros((len(gray) - 1, CANVAS, CANVAS), dtype=np.uint8)
        for j in range(len(gray) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray[j], gray[j + 1], None, 0.5, 3, 21, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mags[j] = np.clip(mag * MAG_SCALE, 0, 255).astype(np.uint8)
        np.savez_compressed(FLOW_OUT / f"{eid}.npz",
                            mag=mags,
                            mag_mean=float(mags.mean()),
                            mag_max=float(mags.max()),
                            mag_std=float(mags.astype(np.float32).std()))
        if (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(f"  [{i + 1}/{len(todo)}] {dt / 60:.1f}min", flush=True)
    print(f"flow 完成 {len(todo)}，耗时 {(time.time() - t0) / 60:.1f}min")


def _to_px(u, v):
    """筐归一化坐标 → 画布像素（v 向下为正，与图像一致）。"""
    s = CANVAS / (2 * RANGE)
    return int(round(CANVAS / 2 + u * s)), int(round(CANVAS / 2 + v * s))


def run_traj():
    TRAJ_OUT.mkdir(parents=True, exist_ok=True)
    events = load_dataset_events()
    todo = [e for e in events
            if (TRACKS_C / f"{e['event_id']}.npz").exists()
            and not (TRAJ_OUT / f"{e['event_id']}.npz").exists()]
    print(f"traj: events={len(events)} 待渲染={len(todo)}", flush=True)
    t0 = time.time()
    for i, ev in enumerate(todo):
        eid = ev["event_id"]
        t = np.load(TRACKS_C / f"{eid}.npz")
        x1, y1, x2, y2 = [float(v) for v in ev["hoop"]]
        hc, vc = (x1 + x2) / 2, (y1 + y2) / 2
        hh = max(y2 - y1, 1.0)
        hw = max(x2 - x1, 1.0)

        img = np.full((CANVAS, CANVAS, 3), 20, dtype=np.uint8)
        # 筐框（归一化后固定绘制，坐标不随场馆变化）
        p1 = _to_px(-hw / hh / 2, -0.5)
        p2 = _to_px(hw / hh / 2, 0.5)
        cv2.rectangle(img, p1, p2, (0, 200, 0), 2)
        cv2.line(img, (p1[0] - 12, p1[1]), (p2[0] + 12, p1[1]), (0, 200, 0), 1)

        if len(t["cx"]) >= 2:
            u = (t["cx"] - hc) / hh
            v = (t["cy"] - vc) / hh
            pts = np.array([_to_px(a, b) for a, b in zip(u, v)], dtype=np.int32)
            # 按时间染色（蓝→红）画轨迹
            for k in range(len(pts) - 1):
                c = int(255 * k / max(len(pts) - 2, 1))
                cv2.line(img, tuple(pts[k]), tuple(pts[k + 1]), (255 - c, 40, c), 2)
            cv2.circle(img, tuple(pts[0]), 5, (255, 255, 0), -1)   # 起点黄
            cv2.circle(img, tuple(pts[-1]), 5, (0, 0, 255), -1)    # 终点红
        np.savez_compressed(TRAJ_OUT / f"{eid}.npz", x=img)
        if (i + 1) % 300 == 0:
            print(f"  [{i + 1}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    print(f"traj 完成 {len(todo)}，耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", action="store_true")
    ap.add_argument("--traj", action="store_true")
    args = ap.parse_args()
    if args.flow or not (args.flow or args.traj):
        run_flow()
    if args.traj:
        run_traj()
