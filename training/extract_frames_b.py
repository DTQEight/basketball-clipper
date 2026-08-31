# -*- coding: utf-8 -*-
"""B 轮数据准备：为每个已标注事件抽取 16 帧篮筐中心裁剪块，存成 npz。

路线（4GB 显存可跑的 B 轮）：
  1. 本脚本离线抽帧写盘（uint8，约 2.5GB）
  2. train_temporal.py 用 ResNet18 逐帧抽特征（GPU 批量）+ Temporal MLP 训练

帧方案沿用 VLM 实测最优的 21 帧加密版抽到 16 帧（穿筐瞬间密采）：
  t=-1.5,-1.0,-0.5,-0.4,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05, 0, +0.1, +0.3, +0.5, +0.8, +1.4
裁剪：以 hoop 为中心，短边 = 3.2×hoop_h（约覆盖篮板+网+下方 1.5 篮高），再 letterbox 到 224。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from video_io import VideoReader  # noqa: E402

DATASET_FILE = PROJECT_ROOT / "training" / "dataset_v1.json"
FEATURES_FILE = PROJECT_ROOT / "training" / "features.jsonl"
HOOP_RECOVERED_FILE = PROJECT_ROOT / "training" / "hoop_recovered.json"


def load_dataset_events():
    """读 export 产物（已合并离线+UI 标签），解析视频路径。

    额外合并「历史清理丢失事件」：detection_history.json 被清理到 50 条后，
    dataset_v1.json 重新 export 缩水；features.jsonl（A 轮累积）里多出的事件
    若视频在盘且 hoop 可从 hoop_recovered.json（recover_hoop.py 产出）恢复，
    也一并纳入 B 轮训练集。
    """
    records = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    from training.annotate import _resolve_video_path
    events = []
    seen = set()
    for r in records:
        resolved = _resolve_video_path(r["video"])
        if not resolved:
            continue
        if not r.get("hoop"):
            continue
        seen.add(r["event_id"])
        events.append({"event_id": r["event_id"], "video": r["video"],
                       "resolved": resolved, "ts": float(r["ts"]),
                       "hoop": r["hoop"], "label": "pos" if r["label"] == 1 else "neg"})

    if HOOP_RECOVERED_FILE.exists() and FEATURES_FILE.exists():
        hoop_map = json.loads(HOOP_RECOVERED_FILE.read_text(encoding="utf-8"))
        if hoop_map:
            n_rec = 0
            for line in FEATURES_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["event_id"] in seen or r["video"] not in hoop_map:
                    continue
                resolved = _resolve_video_path(r["video"])
                if not resolved:
                    continue
                seen.add(r["event_id"])
                events.append({"event_id": r["event_id"], "video": r["video"],
                               "resolved": resolved, "ts": float(r["ts"]),
                               "hoop": hoop_map[r["video"]],
                               "label": "pos" if r["label"] == 1 else "neg"})
                n_rec += 1
            print(f"[recover] 从 features.jsonl 恢复 {n_rec} 个丢失事件")
    return events

OUT_DIR = PROJECT_ROOT / "training" / "frames_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_OFFS = [-1.5, -1.0, -0.5, -0.4, -0.3, -0.25, -0.2, -0.15,
              -0.1, -0.05, 0.0, 0.1, 0.3, 0.5, 0.8, 1.4]
SIZE = 224
ZOOM = 3.2  # 裁剪短边 = ZOOM × hoop 高


def crop_hoop(frame: np.ndarray, hoop, zoom: float = ZOOM, size: int = SIZE):
    """以 hoop 为中心裁剪正方形，letterbox 缩放到 size×size。返回 (块, 有效标志)。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in hoop]
    hc, vc = (x1 + x2) / 2, (y1 + y2) / 2
    hh = max(y2 - y1, 1.0)
    side = zoom * hh
    cx1, cy1 = int(hc - side / 2), int(vc - side / 2)
    cx2, cy2 = cx1 + int(side), cy1 + int(side)
    # 边界 clamp（超出部分填灰）
    px1, py1 = max(0, cx1), max(0, cy1)
    px2, py2 = min(w, cx2), min(h, cy2)
    out = np.full((int(side), int(side), 3), 114, dtype=np.uint8)
    if px2 > px1 and py2 > py1:
        out[py1 - cy1:py2 - cy1, px1 - cx1:px2 - cx1] = frame[py1:py2, px1:px2]
    # 等比缩到 size（最近邻+区域混合：先整型放大再线性）
    import cv2
    resized = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return resized


def main():
    events = load_dataset_events()
    events = [e for e in events if e.get("label") in ("pos", "neg") and e.get("hoop")]
    print(f"events: {len(events)}")

    done = 0
    skipped = 0
    t0 = time.time()
    for ev in events:
        eid = ev["event_id"]
        out_path = OUT_DIR / f"{eid}.npz"
        if out_path.exists():
            skipped += 1
            continue
        resolved = ev.get("resolved") or ev.get("video")
        try:
            reader = VideoReader(resolved)
        except Exception:
            print(f"[SKIP] open fail: {ev['video']}")
            continue
        try:
            fps = reader.fps
            total = reader.total
            ts = float(ev["ts"])
            hoop = ev["hoop"]
            frames = np.zeros((len(FRAME_OFFS), SIZE, SIZE, 3), dtype=np.uint8)
            # offsets 单调递增 → 用 iter_frames 顺序解码到最后一帧
            last = int((ts + FRAME_OFFS[-1]) * fps)
            last = max(0, min(last, total - 1))
            want = {}
            for i, off in enumerate(FRAME_OFFS):
                fidx = int((ts + off) * fps)
                fidx = max(0, min(fidx, total - 1))
                want[fidx] = i
            start = min(want)
            got = 0
            for fidx, frame in reader.iter_frames(start=start, end=last + 1):
                if fidx in want:
                    frames[want[fidx]] = crop_hoop(frame, hoop)
                    got += 1
                    if got >= len(FRAME_OFFS):
                        break
            np.savez_compressed(out_path, x=frames)
            done += 1
            if done % 50 == 0:
                dt = time.time() - t0
                print(f"  [{done}/{len(events)}] {dt:.0f}s ({done/dt*60:.0f} ev/min)", flush=True)
        finally:
            reader.close()
    print(f"完成：新写 {done}，已有跳过 {skipped}，总 {done + skipped}")


if __name__ == "__main__":
    main()
