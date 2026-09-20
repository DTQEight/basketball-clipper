# -*- coding: utf-8 -*-
"""砍臂可行性验证（消融）：真实视频（概率加权，线上口径）+ OOF（rank 平均，离线口径）。

用来回答「某个臂能不能去掉/降权」——同时给出线上同构口径的 AUC 与零误报工作点，
避免只按离线 OOF 做决策（离线口径曾给出相反的结论）。

用法：
    env\\python.exe training\\eval_arm_ablation.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import state  # noqa: E402

CACHE = json.loads((ROOT / "cache" / "clip_cache.json").read_text(encoding="utf-8"))

# 组合定义：(名称, 权重) —— 权重为 0 的臂不参与，缺臂按剩余权重重归一化（同 combine）
VARIANTS = [
    ("现役 a_b2_f_vm   (A .5 B 2 F 1 V 1)", {"lgbm": .5, "b": 2, "flow": 1, "vm": 1}),
    ("去A · 等权3臂     (B 1 F 1 V 1)     ", {"b": 1, "flow": 1, "vm": 1}),
    ("去A · 同比例      (B .5 F 1 V 1)    ", {"b": .5, "flow": 1, "vm": 1}),
    ("去B · 等权3臂     (A 1 F 1 V 1)     ", {"lgbm": 1, "flow": 1, "vm": 1}),
    ("去B · A半权       (A .5 F 1 V 1)    ", {"lgbm": .5, "flow": 1, "vm": 1}),
    ("去B去A · 只视觉     (F 1 V 1)        ", {"flow": 1, "vm": 1}),
    ("去VM · 三臂        (A .5 B 2 F 1)   ", {"lgbm": .5, "b": 2, "flow": 1}),
]


def ens_prob(rows, w):
    out = []
    for r in rows:
        got = [(w[k], r.get("score_" + k)) for k in w
               if w.get(k, 0) > 0 and r.get("score_" + k) is not None]
        ws = sum(a for a, _ in got)
        out.append(sum(a * b for a, b in got) / ws if ws else None)
    return np.array([np.nan if v is None else v for v in out], dtype=float)


def zero_err_point(y, s):
    """最高召回且精确率=1 的阈值（等价于取到最高分那个负例之上）。"""
    best = (0, 0.0, 0)
    for th in np.unique(np.round(s, 3)):
        sel = s >= th
        tp = int((sel & (y == 1)).sum())
        fp = int((sel & (y == 0)).sum())
        if tp and fp == 0 and tp / y.sum() > best[0]:
            best = (tp / y.sum(), float(th), tp)
    return best


def collect():
    """所有「已标注且有全部分数」的场次。"""
    out = []
    for entry in CACHE:
        lab = state.get_labels(entry["video"])
        kept, dele = state.label_sets(lab)   # 人工 ∪ 模型（见 state.label_sets）
        rows = [c for c in entry["clips"]
                if round(float(c["ts"]), 3) in kept | dele and c.get("score") is not None]
        if rows:
            out.append((Path(entry["video"]).name, rows, kept))
    return sorted(out)


videos = collect()
if not videos:
    print("没有可用的已标注场次")
    sys.exit(1)

print("=" * 86)
print(f"真实视频（{sum(len(v[1]) for v in videos)} 个已人工标注的候选，线上概率加权口径）")
print("=" * 86)
y_all, s_all = [], {v[0]: [] for v in VARIANTS}
for name, rows, kept in videos:
    y = np.array([1 if round(float(c["ts"]), 3) in kept else 0 for c in rows])
    print(f"\n{name}: {len(rows)} 个（√ {int(y.sum())} / × {len(y) - int(y.sum())}）")
    for vname, w in VARIANTS:
        s = ens_prob(rows, w)
        ok = ~np.isnan(s)
        rec, th, n = zero_err_point(y[ok], s[ok])
        print(f"  {vname}  AUC={roc_auc_score(y[ok], s[ok]):.4f}  零误报阈值={th:.2f} → "
              f"通过 {n} 个，精确率 1.000，召回 {rec:.1%}")
        s_all[vname].append(ens_prob(rows, w))
    y_all.append(y)

print("\n" + "-" * 86)
print(f"合并（{sum(len(v[1]) for v in videos)} 个）")
yy = np.concatenate(y_all)
for vname, _ in VARIANTS:
    s = np.concatenate(s_all[vname])
    ok = ~np.isnan(s)
    rec, th, n = zero_err_point(yy[ok], s[ok])
    print(f"  {vname}  AUC={roc_auc_score(yy[ok], s[ok]):.4f}  零误报阈值={th:.2f} → "
          f"通过 {n} 个，精确率 1.000，召回 {rec:.1%}")

# ===== OOF =====
OOF = ROOT / "training" / "oof_directions.jsonl"
if OOF.exists():
    print("\n" + "=" * 86)
    print("OOF（rank 平均口径，与 train_directions 的集成搜索一致）")
    print("=" * 86)
    oof = [json.loads(l) for l in OOF.read_text(encoding="utf-8").splitlines() if l.strip()]
    yo = np.array([r["label"] for r in oof])

    def rk(v):
        return rankdata(v) / len(v)

    combos = {
        "a+b+flow_t+vm": ["pred_a", "pred_b", "pred_flow_t", "pred_vm"],
        "b+flow_t+vm（去A）": ["pred_b", "pred_flow_t", "pred_vm"],
        "a+flow_t+vm（去B）": ["pred_a", "pred_flow_t", "pred_vm"],
        "flow_t+vm": ["pred_flow_t", "pred_vm"],
        "a+b+flow_t+vm+c_cat": ["pred_a", "pred_b", "pred_flow_t", "pred_vm", "pred_c_cat"],
    }
    for name, cols in combos.items():
        if not all(c in oof[0] for c in cols):
            continue
        s = np.mean([rk([r[c] for r in oof]) for c in cols], axis=0)
        print(f"  {name}: AUC={roc_auc_score(yo, s):.4f}")
