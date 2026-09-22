# -*- coding: utf-8 -*-
"""对比 AI 自动通过与人工标记：算「自动通过」这一端的精确率/召回率。

用法：
    env\\python.exe training\\eval_ai_vs_manual.py 2026.09.01-3rd.mp4

依赖 cache/clip_cache.json（各臂分与 auto）与 cache/history（人工标签）。
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import state  # noqa: E402

target = sys.argv[1] if len(sys.argv) > 1 else "2026.09.01-3rd.mp4"

records = state.load_history()
rec = next((r for r in records if os.path.basename(r.get("video", "")) == target), None)
if rec is None:
    print("未找到历史记录:", target)
    sys.exit(1)
video = rec["video"]
labels = state.get_labels(video)
# 线上口径：人工 √/× ∪ 模型自动 √/×。只看 kept/deleted 会把模型自动 √ 当"未标注"，
# 在评估里直接变成假阴性（见 services/state.py 的 label_sets）
kept, deleted = state.label_sets(labels)
print(f"视频: {video}")
print(f"标签时间: {labels.get('label_time')}  √={len(kept)}  ×={len(deleted)}  "
      f"未标={len(rec.get('goals', [])) - len(kept) - len(deleted)}")

data = json.loads((ROOT / "cache" / "clip_cache.json").read_text(encoding="utf-8"))
entry = next((i for i in data if i["video"] == video
              and len(i["goals"]) == len(rec.get("goals", []))), None)
if entry is None:
    print("片段缓存里没有匹配条目")
    sys.exit(1)
clips = {round(float(c["ts"]), 3): c for c in entry["clips"]}
print(f"缓存片段: {len(clips)}  带分数: {sum(1 for c in clips.values() if 'score' in c)}")

rows = []
for ts in sorted(clips):
    c = clips[ts]
    mark = "√" if ts in kept else ("×" if ts in deleted else "?")
    rows.append((ts, c.get("score"), bool(c.get("auto")), mark,
                 c.get("score_lgbm"), c.get("score_b"), c.get("score_flow"), c.get("score_vm")))

auto = [r for r in rows if r[2]]
print(f"\n自动通过 {len(auto)} 个")
for r in auto:
    print(f"  ts={r[0]:>8.2f} 分={r[1]} 人工={r[3]}  "
          f"A={r[4]} B={r[5]} Flow={r[6]} VM={r[7]}")

tp = sum(1 for r in auto if r[3] == "√")
fp = sum(1 for r in auto if r[3] == "×")
skip = sum(1 for r in auto if r[3] == "?")
n_kept = len(kept)
print(f"\n自动通过中：√ {tp} / × {fp} / 未标 {skip}")
if tp + fp:
    print(f"自动通过精确率（按已人工标注的算）= {tp}/{tp + fp} = {tp / (tp + fp):.3f}")
hit = sum(1 for r in rows if r[2] and r[3] == "√")
print(f"√ 总数 {n_kept}，其中被自动通过命中 {hit} → 占 √ 的 {hit / max(n_kept, 1):.1%}")

print("\n全部片段（按分数降序）:")
for r in sorted(rows, key=lambda x: -(x[1] or 0)):
    print(f"  ts={r[0]:>8.2f} 分={str(r[1]):>6} auto={str(r[2]):>5} 人工={r[3]}")
