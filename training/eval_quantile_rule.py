# -*- coding: utf-8 -*-
"""评估「分位数保底」规则：绝对阈值 OR 本场 top-k（k = ceil(r*n)）。

这是决定「是否启用分位保底」的评估器——每场自动通过的精确率是关键指标，
只要出现一个错误自动 √（精度 < 1.0）就不应启用。

用法：
    env\\python.exe training\\eval_quantile_rule.py

自动遍历 cache/clip_cache.json 里所有「已人工标注且有分数」的场次，
阈值取自 training/model_temporal_meta.json 的 ensemble.keep_thr（避免硬编码漂移）。
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import state  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

META = json.loads((ROOT / "training" / "model_temporal_meta.json").read_text(encoding="utf-8"))
THR = float(META.get("ensemble", {}).get("keep_thr", 0.72))
CACHE = json.loads((ROOT / "cache" / "clip_cache.json").read_text(encoding="utf-8"))
FRACS = (0.05, 0.10, 0.15, 0.20)


def labelled(entry):
    """[(ts, score, is_goal)]，只含已标注且有分数的片段（标注取人工 ∪ 模型）。"""
    lab = state.get_labels(entry["video"])
    kept, dele = state.label_sets(lab)
    out = []
    for c in entry["clips"]:
        ts = round(float(c["ts"]), 3)
        if (ts not in kept and ts not in dele) or c.get("score") is None:
            continue
        out.append((ts, float(c["score"]), ts in kept))
    return out


videos = []
for entry in CACHE:
    rows = labelled(entry)
    if rows:
        videos.append((Path(entry["video"]).name, rows))
videos.sort()
if not videos:
    print("没有任何「已标注且有分数」的场次——先跑几场检测并人工标注")
    sys.exit(1)

print(f"阈值 keep_thr = {THR}（来自 training/model_temporal_meta.json）")
print(f"可用场次 {len(videos)}\n")

print("=" * 92)
print("逐场：绝对阈值")
print("=" * 92)
print(f"{'视频':<24}{'候选':>5}{'√':>5}{'AUC':>9}{'自动√':>7}{'精度':>8}{'召回':>8}")
tot_sel = tot_tp = tot_pos = tot_n = 0
for name, rows in videos:
    y = [1 if g else 0 for _, _, g in rows]
    s = [sc for _, sc, _ in rows]
    sel = [r for r in rows if r[1] >= THR]
    tp = sum(1 for r in sel if r[2])
    n_pos = sum(y)
    print(f"{name[:23]:<24}{len(rows):>5}{n_pos:>5}"
          f"{roc_auc_score(y, s) if 0 < n_pos < len(rows) else float('nan'):>9.4f}"
          f"{len(sel):>7}"
          f"{tp / max(len(sel), 1):>8.3f}{tp / max(n_pos, 1):>8.1%}")
    tot_sel += len(sel)
    tot_tp += tp
    tot_pos += n_pos
    tot_n += len(rows)
print("-" * 92)
print(f"{'合计':<24}{tot_n:>5}{tot_pos:>5}{'':>9}{tot_sel:>7}"
      f"{tot_tp / max(tot_sel, 1):>8.3f}{tot_tp / max(tot_pos, 1):>8.1%}")

print()
print("=" * 92)
print("合并规则（绝对阈值 OR 本场 top-k）")
print("=" * 92)
print(f"{'规则':<20}{'自动√':>8}{'其中√':>8}{'精度':>9}{'召回':>9}{'相对绝对阈值新增':>18}")
for frac in FRACS:
    sel_tp = sel_n = 0
    delta = 0
    for _, rows in videos:
        n_pos = sum(1 for r in rows if r[2])
        k = max(1, math.ceil(frac * len(rows)))
        top = sorted(rows, key=lambda r: -r[1])[:k]
        picked = {r[0] for r in rows if r[1] >= THR} | {r[0] for r in top}
        tp = sum(1 for r in rows if r[0] in picked and r[2])
        abs_n = sum(1 for r in rows if r[1] >= THR)
        sel_tp += tp
        sel_n += len(picked)
        delta += len(picked) - abs_n
    print(f"{'top ' + format(frac, '.0%'):<20}{sel_n:>8}{sel_tp:>8}"
          f"{sel_tp / max(sel_n, 1):>9.3f}{sel_tp / max(tot_pos, 1):>9.1%}{delta:>18}")

print()
print("判据：精度必须恒为 1.000 才可启用。注意 k = max(1, …) 意味着每场都会强制")
print("自动通过至少 1 个，在进球率低于 10% 或 0 进球的场次可能凭空产生错误 √。")
