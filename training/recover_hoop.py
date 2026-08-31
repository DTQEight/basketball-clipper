# -*- coding: utf-8 -*-
"""一次性脚本：从所有可用的历史备份恢复「丢失事件」的 hoop 映射。

背景：detection_history.json 被自动清理到 50 条后，dataset_v1.json 重新 export
缩水到 1383 事件；features.jsonl（2108 事件）里多出的 755 个事件的视频仍在盘，
但 hoop 标定随历史清理丢了。本脚本把三个 hoop 来源合并成 video→hoop 映射：
  1. cache/detection_history.json（当前 50 条）
  2. cache/detection_history - 副本.json（旧备份 30 条）
  3. E:\\ad-project\\cache\\detection_history.json（NAS 备份 9 条）
  4. dataset_v1.json 现有事件的 hoop（同视频可复用）
输出 training/hoop_recovered.json，供 extract_frames_b.load_dataset_events 合并。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
OUT_FILE = TRAINING_DIR / "hoop_recovered.json"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    feat = {}
    for l in open(TRAINING_DIR / "features.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            feat[r["event_id"]] = r
    ds = set()
    ds_hoops = {}
    for r in json.loads((TRAINING_DIR / "dataset_v1.json").read_text(encoding="utf-8")):
        ds.add(r["event_id"])
        if r.get("hoop"):
            ds_hoops[r["video"]] = r["hoop"]
    lost = {e: r for e, r in feat.items() if e not in ds}
    lost_videos = {r["video"] for r in lost.values()}
    print(f"features: {len(feat)}, dataset_v1: {len(ds)}, 丢失: {len(lost)}")

    hoop_map = {}
    sources = [
        PROJECT_ROOT / "cache" / "detection_history.json",
        PROJECT_ROOT / "cache" / "detection_history - 副本.json",
        Path(r"E:\ad-project\cache\detection_history.json"),
    ]
    for src in sources:
        if not src.exists():
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        n = 0
        for h in data:
            v = h.get("video")
            hp = h.get("hoop")
            if v and hp and v in lost_videos and v not in hoop_map:
                hoop_map[v] = hp
                n += 1
        print(f"{src.name}: +{n} 视频")
    # dataset_v1 同视频 hoop 复用
    n = 0
    for v in lost_videos:
        if v not in hoop_map and v in ds_hoops:
            hoop_map[v] = ds_hoops[v]
            n += 1
    print(f"dataset_v1 复用: +{n} 视频")

    recoverable = [e for e, r in lost.items() if r["video"] in hoop_map]
    import collections
    lab = collections.Counter(r["label"] for e, r in lost.items() if r["video"] in hoop_map)
    print(f"\n可恢复: {len(recoverable)}/{len(lost)} 事件（{len(hoop_map)}/{len(lost_videos)} 视频）"
          f"  标签: {dict(lab)}")
    unrec_videos = lost_videos - set(hoop_map)
    unrec_events = [e for e, r in lost.items() if r["video"] in unrec_videos]
    lab2 = collections.Counter(r["label"] for e, r in lost.items() if r["video"] in unrec_videos)
    print(f"无法恢复（hoop 彻底丢失）: {len(unrec_events)} 事件 / {len(unrec_videos)} 视频"
          f"  标签: {dict(lab2)}")
    print("\n无法恢复的视频（需在 UI 重新标定）:")
    import collections as C
    cnt = C.Counter(r["video"] for r in lost.values() if r["video"] in unrec_videos)
    for v, n_ in sorted(cnt.items()):
        print(f"  {n_:4d}  {v}")

    OUT_FILE.write_text(json.dumps(hoop_map, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n已写 {OUT_FILE}（{len(hoop_map)} 条 video→hoop）")


if __name__ == "__main__":
    main()
