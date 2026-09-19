# -*- coding: utf-8 -*-
"""B 臂换 SimCLR 骨干后，重定集成权重与阈值（OOF 831 事件）。

背景：线上 ensemble 的权重与阈值本来由 `train_directions.py --deploy` 在 OOF 上标定，
但换骨干（B: ImageNet ResNet18 → SimCLR 自监督编码器）后没有对应的 --deploy 流程，
这个脚本做的是同一件事：按 OOF AUC 择优 + precision>=0.95 的最大召回工作点。
结果写入 training/model_temporal_meta.json 的 ensemble 段（注意：重跑 --deploy 会覆盖回旧口径）。

使用的前提产物（均在 training/）：
    oof_directions.jsonl        —— A / Flow / VM / 旧 B 的折外预测
    oof_temporal_simclr.jsonl   —— 新 B（SimCLR 骨干）的折外预测

用法：
    env\\python.exe training\\recalib_ensemble.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
TR = ROOT / "training"


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


d = {r["event_id"]: r for r in read_jsonl(TR / "oof_directions.jsonl")}
nb = {r["event_id"]: r for r in read_jsonl(TR / "oof_temporal_simclr.jsonl")}
ids = [k for k in d if k in nb]
y = np.array([d[k]["label"] for k in ids])
print(f"对齐事件: {len(ids)}（正 {int(y.sum())}）")

P = {
    "a": np.array([d[k]["pred_a"] for k in ids]),
    "b_old": np.array([d[k]["pred_b"] for k in ids]),      # rank 平均（旧口径）
    "b_new": np.array([nb[k]["pred"] for k in ids]),       # 概率均值（新，线上同构）
    "flow": np.array([d[k]["pred_flow_t"] for k in ids]),
    "vm": np.array([d[k]["pred_vm"] for k in ids]),
}


def rk(v):
    return rankdata(v) / len(v)


print("\n单臂 OOF AUC")
print(f"  A(LGBM)        {roc_auc_score(y, P['a']):.4f}")
print(f"  B 旧(ResNet18) {roc_auc_score(y, P['b_old']):.4f}")
print(f"  B 新(SimCLR)   {roc_auc_score(y, P['b_new']):.4f}")
print(f"  Flow_T         {roc_auc_score(y, P['flow']):.4f}")
print(f"  VM             {roc_auc_score(y, P['vm']):.4f}")

print("\nrank 口径（与之前分析可比，旧 B）")
for name, cols in {"a+b_old+flow+vm": ["a", "b_old", "flow", "vm"],
                   "b_old+flow+vm": ["b_old", "flow", "vm"]}.items():
    s = np.mean([rk(P[c]) for c in cols], axis=0)
    print(f"  {name}: {roc_auc_score(y, s):.4f}")

print("\nrank 口径（新 B）")
for name, cols in {"a+b_new+flow+vm": ["a", "b_new", "flow", "vm"],
                   "b_new+flow+vm": ["b_new", "flow", "vm"],
                   "a+b_new+flow": ["a", "b_new", "flow"],
                   "b_new+flow": ["b_new", "flow"]}.items():
    s = np.mean([rk(P[c]) for c in cols], axis=0)
    print(f"  {name}: {roc_auc_score(y, s):.4f}")

print("\n概率加权口径（线上同构，新 B）—— 权重与工作点")
CANDS = {
    "mean4 等权 (a b f v)": {"a": 1, "b_new": 1, "flow": 1, "vm": 1},
    "a半权 (a.5 b1 f1 v1)": {"a": .5, "b_new": 1, "flow": 1, "vm": 1},
    "老配比 (a.5 b.5 f1 v1)": {"a": .5, "b_new": .5, "flow": 1, "vm": 1},
    "去A 等权 (b f v)": {"b_new": 1, "flow": 1, "vm": 1},
    "B为尊 (a.5 b2 f1 v1)": {"a": .5, "b_new": 2, "flow": 1, "vm": 1},
}
n_pos = int(y.sum())
best = None
for name, w in CANDS.items():
    ws = sum(w.values())
    s = sum(w[k] * P[k] for k in w) / ws
    auc = roc_auc_score(y, s)
    # precision>=0.95 的最大召回工作点：取**最低**的达标阈值（同 deploy 的 _wp_threshold）
    wp = None
    for th in np.unique(np.round(s, 3)):
        sel = s >= th
        tp = int((sel & (y == 1)).sum())
        fp = int((sel & (y == 0)).sum())
        if tp and tp / (tp + fp) >= 0.95:
            wp = (float(th), tp / (tp + fp), tp / n_pos)
            break
    print(f"  {name:26s} AUC={auc:.4f}" +
          (f"   p95工作点 th={wp[0]:.3f} 精度={wp[1]:.3f} 召回={wp[2]:.3f}"
           if wp else "   p95不可达"))
    if best is None or auc > best[1]:
        best = (name, auc)
print(f"\n最优: {best[0]}  AUC={best[1]:.4f}")
