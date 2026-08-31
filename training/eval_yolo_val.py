# -*- coding: utf-8 -*-
"""YOLO 篮球模型验收评估：在 yolo_data val 集上测检出率/误报。

用法：env\\python.exe training\\eval_yolo_val.py <weights.pt>
产物：cache/eval_yolo_{权重名}.json

指标（基线与微调后同口径对比）：
  pos_recall@conf      正事件 ts±0.3s 帧中，任一帧在篮筐邻域检出 conf≥阈值框
                       的事件比例（工作点 0.30 = 线上检测阈值）
  hard_frame_conf      难例帧（构建时伪标签 conf 0.05~0.4 的帧）当前最高
                       筐域 conf 分布 —— 微调应整体右移
  easy_keep            易例帧仍能以 conf≥0.4 检出的比例（防普通场景退化）
  bg_false             背景帧 conf≥0.30 检出框总数（防误报上升）
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402

VAL_DIR = PROJECT_ROOT / "training" / "yolo_data" / "images" / "val"
MANIFEST = PROJECT_ROOT / "training" / "yolo_data" / "manifest.jsonl"


def main(weights: str):
    rows = [json.loads(l) for l in
            MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    val = [r for r in rows if r["file"].split("/")[1] == "val"]
    print(f"val 图像 {len(val)}（正事件帧 "
          f"{sum(1 for r in val if r.get('label')=='pos' and r['source']=='event')}）")

    model = YOLO(weights)
    files = [str(PROJECT_ROOT / "training" / "yolo_data" / r["file"])
             for r in val]
    res = []
    for i in range(0, len(files), 16):  # 分批防 OOM（4GB 显存）
        res.extend(model.predict(files[i:i + 16], conf=0.05, imgsz=960,
                                 verbose=False, device=0))
        if (i // 16) % 20 == 0:
            print(f"  推理 {min(i + 16, len(files))}/{len(files)}", flush=True)

    # 每帧筐域最高 conf（按 manifest 的 hoop 换算到图像坐标）
    # manifest 存的是原始帧坐标 + frame_wh；保存时等比缩放 → hoop 同比例缩放
    top_hoop_conf = []
    for r, det in zip(val, res):
        ow, oh = r["frame_wh"]
        imh, imw = det.orig_shape
        sx, sy = imw / ow, imh / oh
        h1, v1, h2, v2 = r.get("hoop", (0, 0, 0, 0))
        if r["source"] != "event" or not h1:
            top_hoop_conf.append(None); continue
        hx, hy = (h1 + h2) / 2 * sx, (v1 + v2) / 2 * sy
        hw = max((h2 - h1) / 2 * sx, 8.0)
        best = 0.0
        if det.boxes is not None and len(det.boxes):
            xyxy = det.boxes.xyxy.cpu().numpy()
            cf = det.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, cf):
                if (abs((x1 + x2) / 2 - hx) < hw * 2.5
                        and abs((y1 + y2) / 2 - hy) < hw * 2.5):
                    best = max(best, float(c))
        top_hoop_conf.append(best)

    # ---- 事件级召回（ts±0.3s 内任一帧筐域 conf≥thr）----
    ev_frames = defaultdict(list)
    for r, c in zip(val, top_hoop_conf):
        if r["source"] == "event" and c is not None:
            ev_frames[r["event_id"]].append((r, c))
    metrics = {}
    for thr in (0.30, 0.20):
        hit = tot = 0
        for eid, items in ev_frames.items():
            if items[0][0].get("label") != "pos":
                continue
            tot += 1
            if any(c >= thr for r, c in items if abs(r["off"]) <= 0.3):
                hit += 1
        metrics[f"pos_recall@{thr}"] = round(hit / max(tot, 1), 4)
    metrics["pos_events"] = sum(1 for i in ev_frames.values()
                                if i[0][0].get("label") == "pos")

    # ---- 难例帧 conf 分布（构建时伪标签 0.05~0.4 的帧）----
    hard = [c for r, c in zip(val, top_hoop_conf)
            if c is not None and any(0.05 <= b[4] < 0.4 for b in r["boxes"])
            and r["source"] == "event"]
    if hard:
        metrics["hard_frames"] = len(hard)
        metrics["hard_conf_p50/p90"] = [
            round(float(np.median(hard)), 3),
            round(float(np.quantile(hard, 0.9)), 3)]
        metrics["hard_ge_030"] = round(
            sum(1 for c in hard if c >= 0.30) / len(hard), 4)

    # ---- 易例保持 / 背景误报 ----
    easy_keep = tot_easy = 0
    bg_false = 0
    for r, det in zip(val, res):
        if r["source"] == "easy":
            tot_easy += 1
            if det.boxes is not None and len(det.boxes) and bool(
                    (det.boxes.conf.cpu().numpy() >= 0.4).any()):
                easy_keep += 1
        elif r["source"] == "bg":
            if det.boxes is not None and len(det.boxes):
                bg_false += int((det.boxes.conf.cpu().numpy() >= 0.30).sum())
    metrics["easy_keep@0.4"] = round(easy_keep / max(tot_easy, 1), 4)
    metrics["easy_total"] = tot_easy
    metrics["bg_false_boxes@0.3"] = bg_false

    out = {"weights": weights, "metrics": metrics}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    dst = PROJECT_ROOT / "cache" / f"eval_yolo_{Path(weights).stem}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"已保存: {dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         str(PROJECT_ROOT / "weights" / "basketball_custom.pt"))
