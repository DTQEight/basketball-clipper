# -*- coding: utf-8 -*-
"""构建 YOLO 篮球微调数据集（针对「穿网瞬间球变形漏检」）。

数据来源（全部现成，无人工标注）：
  难例正样本  features.jsonl 正事件 ts±0.5s × 11 帧 —— 球入网/缠网瞬间，
              用当前模型 conf=0.05 低置信度检测做伪标签（正是要拉起来的场景）
  难例副样本  负事件（误报候选）ts±0.4s × 5 帧 —— 篮筐附近有球/有运动，
              同样有球可见，防止微调后只对「进球」敏感
  易例        每视频随机 15 帧，conf≥0.4 的框直接当标签（保住普通场景能力）
  背景        每视频随机 10 帧零检测（暂停/观众/无球画面）

伪标签清洗（防垃圾框进训练集）：
  - 框中心必须在篮筐邻域（±2.5×筐半宽，难例）或全场（易例 conf≥0.4）
  - 框面积必须在每视频中位球面积的 [0.2×, 5×] 内
  - 难例帧若 0.05 下完全无框 → 丢弃该帧（宁缺毋滥：球可能可见但模型
    全盲，标成背景会强化盲区）

划分：按比赛日留出 20251121 / 20260711 / 20260822 全部帧 → val（盲测），
其余 → train。金联 20250104 不在 features.jsonl，端到端复跑天然盲测。

产物：training/yolo_data/{images,labels}/{train,val}/ + data.yaml + manifest.jsonl
用法：env\\python.exe training\\build_yolo_dataset.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import get_ball_model, get_ball_class_ids  # noqa: E402
from training.extract_frames_b import load_dataset_events  # noqa: E402
from training.train_temporal import norm_game  # noqa: E402
from video_io import VideoReader, get_video_info, read_frame  # noqa: E402

OUT_DIR = PROJECT_ROOT / "training" / "yolo_data"
HOLDOUT_GAMES = {"20251121", "20260711", "20260822"}
POS_OFFS = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
NEG_OFFS = [-0.4, -0.2, 0.0, 0.2, 0.4]
N_EASY, N_BG, EASY_TRY, BG_TRY = 15, 10, 20, 16
IMG_LONG = 1280          # 存储分辨率长边（训练 imgsz=960，留余量）
JPEG_Q = 88
GLOBAL_MED_AREA = 4000.0  # 球面积全局兜底中位（px²，1080p 下球 40-90px）
random.seed(42)


def detect_batch(model, cls, frames, conf=0.05):
    """批量推理：返回每帧 [(x1,y1,x2,y2,conf), ...]。"""
    out = []
    res = model.predict(frames, conf=conf, imgsz=960, classes=cls,
                        verbose=False, device=0)
    for r in res:
        boxes = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            boxes = [(float(a), float(b), float(c), float(d), float(e))
                     for (a, b, c, d), e in zip(xyxy, cf)]
        out.append(boxes)
    return out


def save_img_labels(img, boxes, split_dir, stem, manifest_rows, meta):
    """保存 JPEG + YOLO txt + manifest 行，返回是否收录。"""
    h, w = img.shape[:2]
    scale = IMG_LONG / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    img_rel = f"images/{split_dir}/{stem}.jpg"
    path = OUT_DIR / img_rel
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    if boxes:
        lines = []
        for x1, y1, x2, y2, _c in boxes:
            lines.append(f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                         f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        (OUT_DIR / f"labels/{split_dir}/{stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8")
    row = {"file": img_rel, "frame_wh": [w, h], "boxes": boxes,
           "kept": len(boxes)}
    row.update(meta)
    manifest_rows.append(row)
    return True


def main():
    t0 = time.time()
    events = load_dataset_events()
    if not events:
        print("ERROR: 无事件"); sys.exit(1)

    by_video = defaultdict(list)
    for e in events:
        by_video[e["resolved"]].append(e)
    print(f"事件 {len(events)} / 视频 {len(by_video)}")

    for d in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)

    model, weights = get_ball_model()
    cls = get_ball_class_ids(model, weights)
    print(f"伪标签模型: {Path(weights).name}, classes={cls}")

    manifest, stats = [], defaultdict(int)
    global_med = GLOBAL_MED_AREA
    frame_no = 0

    for vidx, (video, evs) in enumerate(sorted(by_video.items()), 1):
        name = Path(video).stem
        try:
            info = get_video_info(video)
        except Exception as e:
            print(f"[{vidx}] SKIP {name}: {e}"); continue
        fps, total = info["fps"], info["total"]
        split = "val" if norm_game(video) in HOLDOUT_GAMES else "train"

        # ---- 阶段 0：随机帧（易例/背景），先跑以获得本视频中位球面积 ----
        easy_boxes_all = []
        cand = random.sample(range(total), min(EASY_TRY + BG_TRY, total))
        cand_frames = []
        for fidx in cand:
            try:
                f = read_frame(video, fidx, total=total, fps=fps)
            except Exception:
                continue
            if f is not None:
                cand_frames.append((fidx, f))
        cand_boxes = detect_batch(model, cls,
                                  [f for _, f in cand_frames]) if cand_frames else []
        for (fidx, _f), boxes in zip(cand_frames, cand_boxes):
            if any(c >= 0.4 for *_, c in boxes):
                easy_boxes_all.append((fidx, boxes))
        n_easy = n_bg = 0
        for (fidx, f), boxes in zip(cand_frames, cand_boxes):
            if n_easy < N_EASY and any(c >= 0.4 for *_, c in boxes):
                frame_no += 1
                save_img_labels(
                    f, [b for b in boxes if b[4] >= 0.4], split,
                    f"v{vidx:03d}_{frame_no:06d}", manifest,
                    {"source": "easy", "video": name})
                n_easy += 1
            elif n_bg < N_BG and not boxes:
                frame_no += 1
                save_img_labels(f, [], split, f"v{vidx:03d}_{frame_no:06d}",
                                manifest, {"source": "bg", "video": name})
                n_bg += 1
            if n_easy >= N_EASY and n_bg >= N_BG:
                break
        stats["easy"] += n_easy; stats["bg"] += n_bg

        areas = [max((x2 - x1) * (y2 - y1), 1.0)
                 for _, bs in easy_boxes_all for x1, y1, x2, y2, _ in bs
                 if True]
        # 中位球面积：易例框可能含远处小人误检？单类 basketball 无此虞
        med_area = float(np.median(areas)) if len(areas) >= 8 else global_med

        # ---- 阶段 1：事件窗口（难例）----
        evs.sort(key=lambda e: e["ts"])
        n_ev_img = n_ev_drop = n_ev_lbl = 0
        reader = None
        try:
            reader = VideoReader(video)
            for ev in evs:
                offs = POS_OFFS if ev["label"] == "pos" else NEG_OFFS
                ts = ev["ts"]
                h1, v1, h2, v2 = [float(v) for v in ev["hoop"]]
                hx, hy = (h1 + h2) / 2, (v1 + v2) / 2
                hw = max((h2 - h1) / 2, 8.0)
                want = {}
                for off in offs:
                    fidx = max(0, min(int((ts + off) * fps), total - 1))
                    want[fidx] = off
                last = max(want)
                got = {}
                for fidx, frame in reader.iter_frames(start=min(want),
                                                      end=last + 1):
                    if fidx in want:
                        got[fidx] = frame
                if not got:
                    continue
                frames = [got[k] for k in sorted(got)]
                offs_got = [want[k] for k in sorted(got)]
                boxes_per = detect_batch(model, cls, frames)
                for frame, off, boxes in zip(frames, offs_got, boxes_per):
                    near = [b for b in boxes
                            if abs((b[0] + b[2]) / 2 - hx) < hw * 2.5
                            and abs((b[1] + b[3]) / 2 - hy) < hw * 2.5]
                    keep = [b for b in near
                            if 0.2 * med_area <= max((b[2] - b[0]) * (b[3] - b[1]), 1.0) <= 5.0 * med_area]
                    keep += [b for b in boxes if b[4] >= 0.4
                             and b not in near]
                    if not keep:
                        n_ev_drop += 1   # 无可信框：丢弃（防强化盲区）
                        continue
                    frame_no += 1
                    save_img_labels(frame, keep, split,
                                    f"v{vidx:03d}_{frame_no:06d}", manifest,
                                    {"source": "event", "video": name,
                                     "event_id": ev["event_id"],
                                     "label": ev["label"], "ts": ts, "off": off,
                                     "hoop": [h1, v1, h2, v2]})
                    n_ev_img += 1; n_ev_lbl += len(keep)
        except Exception as e:
            print(f"[{vidx}] {name} 事件阶段异常: {e}")
        finally:
            if reader is not None:
                try: reader.close()
                except Exception: pass
        stats["event_img"] += n_ev_img
        stats["event_drop"] += n_ev_drop
        stats["event_lbl"] += n_ev_lbl
        if len(areas) >= 8:
            global_med = 0.7 * global_med + 0.3 * med_area
        print(f"[{vidx}/{len(by_video)}] {name}: split={split} "
              f"事件图 {n_ev_img}（弃 {n_ev_drop}）标签 {n_ev_lbl} | "
              f"易例 {n_easy} 背景 {n_bg} | med_area={med_area:.0f} "
              f"({(time.time()-t0)/60:.1f}min)", flush=True)

    # ---- data.yaml + manifest ----
    (OUT_DIR / "data.yaml").write_text(
        f"path: {OUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: basketball\n", encoding="utf-8")
    with open(OUT_DIR / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_val = sum(1 for r in manifest if r["file"].split("/")[1] == "val")
    print(f"\n完成: 图像 {len(manifest)}（train {len(manifest)-n_val} / "
          f"val {n_val}）标签框 {sum(r['kept'] for r in manifest)}")
    print(f"统计: {dict(stats)}")
    print(f"耗时 {(time.time()-t0)/60:.1f}min → {OUT_DIR}")


if __name__ == "__main__":
    main()
