# -*- coding: utf-8 -*-
"""单场 AUC + 不同阈值下「自动通过」的精确率/召回率。

用法：
    env\\python.exe training\\eval_thresholds.py "Y:\\全场录像\\2026.09.01\\2026.09.01-3rd.mp4"

依赖 cache/clip_cache.json（含各片段 score）与 cache/history（人工标签）。
未打分的片段自动跳过，不会 KeyError。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import state  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

v = sys.argv[1] if len(sys.argv) > 1 else r"Y:\全场录像\2026.09.01\2026.09.01-3rd.mp4"
lab = state.get_labels(v)
kept = {round(float(t), 3) for t in (lab.get("kept") or [])}
dele = {round(float(t), 3) for t in (lab.get("deleted") or [])}
data = json.loads((ROOT / "cache" / "clip_cache.json").read_text(encoding="utf-8"))
e = next(i for i in data if i["video"] == v)

y, s = [], []
skipped = 0
for c in e["clips"]:
    ts = round(float(c["ts"]), 3)
    if ts not in kept and ts not in dele:
        continue
    if c.get("score") is None:
        skipped += 1
        continue
    y.append(1 if ts in kept else 0)
    s.append(float(c["score"]))

print(f"已标注且有分数 {len(y)} 个（√ {sum(y)} / × {len(y) - sum(y)}）"
      + (f"  跳过未打分 {skipped} 个" if skipped else ""))
print("本场 AUC =", round(roc_auc_score(y, s), 4))
for th in (0.834, 0.80, 0.75, 0.72, 0.70, 0.65, 0.60):
    tp = sum(1 for a, b in zip(y, s) if b >= th and a == 1)
    fp = sum(1 for a, b in zip(y, s) if b >= th and a == 0)
    print(f"  阈值 {th:.3f}: 自动通过 {tp + fp:>2} 个  精确率 {tp / max(tp + fp, 1):.3f}"
          f"  召回 {tp / sum(y):.1%}")
