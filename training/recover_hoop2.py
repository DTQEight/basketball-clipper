# -*- coding: utf-8 -*-
"""B 轮数据补齐：批量恢复丢失的 hoop 标定。

两类恢复：
  1. 同名副本复制：dataset_v1.json 里的同名视频（源已不在盘）有 hoop，
     在盘的 test 副本分辨率一致（已验证 11/12 确认同分辨率、1 个占比
     6.5% 略小但无错位风险）→ 直接复制坐标。
  2. YOLO 球轨迹反推：features.jsonl 的事件 ts 都是"检测器判定球进筐区"
     的时刻（正样本球穿筐 / 负样本球在筐附近）→ 对事件附近帧跑球检测，
     球位置中位数 = 篮筐中心；hoop 尺寸按全库统计中位（高=0.09×视频高，
     宽=0.90×高）。与同比赛日已有标定交叉校验。

输出：
  training/hoop_recovered.json   合并写入（extract_frames_b 自动读取）
  training/hoop_recovered2_report.json  来源与置信度报告
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.annotate import _resolve_video_path  # noqa: E402
from training.train_temporal import norm_game  # noqa: E402
from video_io import VideoReader, read_frame, get_video_info  # noqa: E402

TR = PROJECT_ROOT / "training"
DATASET_FILE = TR / "dataset_v1.json"
FEATURES_FILE = TR / "features.jsonl"
RECOVERED_FILE = TR / "hoop_recovered.json"
REPORT_FILE = TR / "hoop_recovered2_report.json"

HOOP_H_RATIO = 0.09   # hoop 高 / 视频高（全库中位）
HOOP_W_OVER_H = 0.90  # hoop 宽 / hoop 高（全库中位）
MAX_EVENTS_PER_VIDEO = 24
FRAME_OFFSETS = (-0.2, 0.0, 0.2)
BALL_CONF = 0.40


def load_pool():
    """全量事件池 + 已有 hoop 标定。"""
    have = {}
    ds = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    by_name = defaultdict(list)
    for r in ds:
        if r.get("hoop"):
            have[r["video"]] = r["hoop"]
            by_name[Path(r["video"]).name].append(r)
    hr = json.loads(RECOVERED_FILE.read_text(encoding="utf-8"))
    have.update(hr)

    events = defaultdict(list)  # video -> [(ts, label)]
    for line in FEATURES_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        events[r["video"]].append((float(r["ts"]), int(r["label"])))
    return have, by_name, events


def copy_same_name(miss_vids, by_name, report):
    """同名副本 hoop 直接复制。"""
    out = {}
    for v in miss_vids:
        name = Path(v).name
        cands = by_name.get(name)
        if not cands:
            continue
        hoop = cands[0]["hoop"]
        out[v] = hoop
        report[v] = {"method": "same_name_copy", "hoop": hoop,
                     "src": cands[0]["video"], "n_events": None}
    return out


def yolo_infer_hoop(video, ev_list, model, report):
    """YOLO 球轨迹反推 hoop：事件帧球位置中位数。"""
    resolved = _resolve_video_path(video)
    if not resolved:
        report[video] = {"method": "yolo_track", "error": "视频不在盘"}
        return None
    info = get_video_info(resolved)
    fps = info["fps"]

    # 正样本优先（球必穿筐），不足再用负样本补
    ev_sorted = sorted(ev_list, key=lambda e: -e[1])[:MAX_EVENTS_PER_VIDEO]
    pts = []
    for ts, _label in ev_sorted:
        for off in FRAME_OFFSETS:
            fidx = int((ts + off) * fps)
            if fidx < 0 or fidx >= info["total"]:
                continue
            frame = read_frame(resolved, fidx, info["total"], fps)
            if frame is None:
                continue
            res = model.predict(frame, verbose=False, conf=BALL_CONF,
                                classes=[0], imgsz=640)
            for b in res[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                pts.append(((x1 + x2) / 2, (y1 + y2) / 2))
    if len(pts) < 8:
        report[video] = {"method": "yolo_track", "error":
                         f"球位置点不足（{len(pts)} < 8）"}
        return None
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    cx, cy = float(np.median(xs)), float(np.median(ys))
    # 离散度：中位绝对偏差（像素）—— 大于 hoop 半高则不可靠
    mad_x, mad_y = float(np.median(np.abs(xs - cx))), float(np.median(np.abs(ys - cy)))
    hh = info["height"] * HOOP_H_RATIO
    ww = hh * HOOP_W_OVER_H
    hoop = [round(cx - ww / 2), round(cy - hh / 2),
            round(cx + ww / 2), round(cy + hh / 2)]
    report[video] = {
        "method": "yolo_track", "hoop": hoop, "n_pts": len(pts),
        "mad": [round(mad_x, 1), round(mad_y, 1)],
        "n_events_used": len(ev_sorted),
        "resolution": f"{info['width']}x{info['height']}",
    }
    return hoop


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    have, by_name, events = load_pool()
    miss_vids = [v for v in events if v not in have]
    print(f"缺 hoop 视频: {len(miss_vids)} 个")

    report = {}
    # 1) 同名复制
    copied = copy_same_name(miss_vids, by_name, report)
    print(f"\n[同名复制] {len(copied)} 个:")
    for v, h in copied.items():
        print(f"  {Path(v).name}  hoop={h}")

    # 2) YOLO 轨迹反推剩余
    rest = [v for v in miss_vids if v not in copied]
    print(f"\n[YOLO 反推] {len(rest)} 个:")
    yolo_hoops = {}
    if rest:
        from ultralytics import YOLO
        model = YOLO(str(PROJECT_ROOT / "weights" / "basketball_custom.pt"))
        t0 = time.time()
        for v in rest:
            hoop = yolo_infer_hoop(v, events[v], model, report)
            if hoop:
                yolo_hoops[v] = hoop
                r = report[v]
                print(f"  {Path(v).name}  hoop={hoop}  "
                      f"(pts={r['n_pts']} mad={r['mad']}) "
                      f"[{time.time() - t0:.0f}s]")
            else:
                print(f"  {Path(v).name}  失败: {report[v].get('error')}")

    # 3) 与同比赛日已有标定交叉校验
    print("\n[交叉校验]（同比赛日已有标定 vs 反推结果）")
    for v, h in yolo_hoops.items():
        g = norm_game(v)
        sibs = [(hv, hh) for hv, hh in have.items() if norm_game(hv) == g]
        for hv, hh in sibs[:2]:
            dx = (h[0] + h[2]) / 2 - (hh[0] + hh[2]) / 2
            dy = (h[1] + h[3]) / 2 - (hh[1] + hh[3]) / 2
            print(f"  {Path(v).name} vs {Path(hv).name}: 偏移 dx={dx:.0f} dy={dy:.0f}")

    # 4) 合并写入（不覆盖已恢复的）
    merged = dict(json.loads(RECOVERED_FILE.read_text(encoding="utf-8")))
    n_new = 0
    for v, h in {**copied, **yolo_hoops}.items():
        if v not in merged:
            merged[v] = h
            n_new += 1
    RECOVERED_FILE.write_text(
        json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n合并写入 {n_new} 个新标定 → {RECOVERED_FILE.name}")
    print(f"报告 → {REPORT_FILE.name}")
    n_fail = len(rest) - len(yolo_hoops)
    if n_fail:
        print(f"仍缺（需 UI 人工标定）: {n_fail} 个")


if __name__ == "__main__":
    main()
