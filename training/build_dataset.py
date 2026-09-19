# -*- coding: utf-8 -*-
"""把 dataset_20260918/blocks_index.json 的事件转成训练流水线要求的 training/dataset_v1.json。

L4 各脚本只用到 event_id / video / ts / hoop / label 五个字段：
    extract_frames_b.py → load_dataset_events()
    extract_features.py → 直读 dataset_v1.json
event_id 沿用其格式 <视频hash8>_<帧号10位>。

用法：
    env\\python.exe training\\build_dataset.py

要换数据源（例如攒够新标注后改为从 cache/history 重建）时，只替换读 ITEMS 那一步，
其余字段格式保持不变，下游脚本无需改动。
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TR = ROOT / "training"
DS = ROOT / "dataset_20260918"

items = json.loads((DS / "blocks_index.json").read_text(encoding="utf-8"))
print("数据源 blocks_index 事件数: %d" % len(items))

rows = []
vhash = {}
for it in items:
    v = it["video"]
    if v not in vhash:
        vhash[v] = hashlib.md5(v.encode("utf-8")).hexdigest()[:8]
    fps = float(it["fps"])
    ts = float(it["ts"])
    fidx = int(round(ts * fps))
    rows.append({
        "event_id": "%s_%010d" % (vhash[v], fidx),
        "video": v,
        "ts": round(ts, 3),
        "clip_path": None,
        "label": 1 if it["label"] == "kept" else 0,
        "hoop": [int(x) for x in it["hoop"]],
        "video_fps": fps,
        "video_width": None,
        "video_height": None,
        "video_duration_sec": None,
        "yolo_confirmed_history": None,
        "yolo_rejected_history": None,
        "detect_time": None,
        "label_time": None,
        "label_source": "offline",
    })

# event_id 去重检查（同一视频同一帧号只应出现一次）
eids = [r["event_id"] for r in rows]
dup = len(eids) - len(set(eids))
print("event_id 唯一性: %d 个，重复 %d" % (len(set(eids)), dup))
if dup:
    seen = {}
    for r in rows:
        k = r["event_id"]
        seen[k] = seen.get(k, 0) + 1
        if seen[k] > 1:
            r["event_id"] = "%s_dup%d" % (k, seen[k])
    print("  已加后缀去重")

(TR / "dataset_v1.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

npos = sum(1 for r in rows if r["label"] == 1)
print("")
print("写出 %s" % (TR / "dataset_v1.json"))
print("  事件 %d   正例 %d / 负例 %d   视频 %d"
      % (len(rows), npos, len(rows) - npos, len(vhash)))
