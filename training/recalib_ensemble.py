# -*- coding: utf-8 -*-
"""换骨干后重定集成权重与阈值（OOF 831 事件）。

背景：线上 ensemble 的权重与阈值本来由 `train_directions.py --deploy` 在 OOF 上
标定，但换骨干（B: ImageNet ResNet18 → SimCLR；Flow: ImageNet ResNet18 →
光流域 SimCLR）后没有对应的 --deploy 流程，这个脚本做的是同一件事：
按 OOF AUC 择优 + precision>=0.95 的最大召回工作点。
（线上同构口径：各臂 sigmoid 概率按权重重归一化加权平均。）

可用产物（均在 training/，存在才纳入）：
    oof_directions.jsonl        —— A / 旧 B / 旧 Flow / VM 的折外预测
    oof_temporal_simclr.jsonl   —— 新 B（SimCLR 骨干）
    oof_flow_simclr.jsonl       —— 新 Flow（光流域 SimCLR 骨干）
缺哪个就退回该臂的旧版本，并在输出里标注。
注意：VM 换时序头的实验（training/exp_vm_head/）未部署，故不纳入——
真实视频上 OOF 增益没兑现，见 goal_verifier 里 VM 臂段的说明。

用法：
    env\\python.exe training\\recalib_ensemble.py
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
TR = ROOT / "training"

# 权重网格（线上概率加权口径）。刻意不取连续值：831 事件上细网格会过拟合 OOF，
# 原 --deploy 也是在一小组手工候选里择优。
A_W = (0.0, 0.25, 0.5, 1.0)
B_W = (0.5, 1.0, 1.5, 2.0)
F_W = (0.5, 1.0, 1.5, 2.0)
V_W = (0.5, 1.0, 1.5, 2.0)
INCUMBENT = {"a": 0.5, "b": 2.0, "flow": 1.0, "vm": 1.0}


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def load(path, key="pred"):
    """读折外预测 → {event_id: pred}；文件不存在返回 None。"""
    if not path.exists():
        return None
    return {r["event_id"]: r[key] for r in read_jsonl(path)}


d = {r["event_id"]: r for r in read_jsonl(TR / "oof_directions.jsonl")}
new_b = load(TR / "oof_temporal_simclr.jsonl")
new_f = load(TR / "oof_flow_simclr.jsonl")
ids = [k for k in d if (new_b is None or k in new_b)
       and (new_f is None or k in new_f)]
y = np.array([d[k]["label"] for k in ids])
print(f"对齐事件: {len(ids)}（正 {int(y.sum())}）")

P = {
    "a": np.array([d[k]["pred_a"] for k in ids]),
    "vm": np.array([d[k]["pred_vm"] for k in ids]),
    # 换骨干的臂：新版本存在则用新版本（线上也是优先新版本）
    "b": np.array([(new_b or {k: d[k]["pred_b"] for k in ids})[k] for k in ids]),
    "flow": np.array([(new_f or {k: d[k]["pred_flow_t"] for k in ids})[k]
                      for k in ids]),
}

print("\n单臂 OOF AUC")
print(f"  A(LGBM)              {roc_auc_score(y, P['a']):.4f}")
if new_b is None:
    print(f"  B(ResNet18 旧)       {roc_auc_score(y, P['b']):.4f}   ← 无新 B 产物")
else:
    print(f"  B(SimCLR 新)         {roc_auc_score(y, P['b']):.4f}")
    print(f"  B(ResNet18 旧)       "
          f"{roc_auc_score(y, np.array([d[k]['pred_b'] for k in ids])):.4f}")
if new_f is None:
    print(f"  Flow(ImageNet 旧)    {roc_auc_score(y, P['flow']):.4f}   ← 无新 Flow 产物")
else:
    print(f"  Flow(SimCLR 新)      {roc_auc_score(y, P['flow']):.4f}")
    print(f"  Flow(ImageNet 旧)    "
          f"{roc_auc_score(y, np.array([d[k]['pred_flow_t'] for k in ids])):.4f}")
print(f"  VM(VideoMAE)         {roc_auc_score(y, P['vm']):.4f}")

print("\nrank 口径（与历史分析可比，仅参考）")


def rk(v):
    return rankdata(v) / len(v)


for name, cols in {"a+b+flow+vm": ["a", "b", "flow", "vm"],
                   "b+flow+vm": ["b", "flow", "vm"],
                   "a+flow+vm（去B）": ["a", "flow", "vm"],
                   "a+b+vm（去Flow）": ["a", "b", "vm"]}.items():
    s = np.mean([rk(P[c]) for c in cols], axis=0)
    print(f"  {name}: {roc_auc_score(y, s):.4f}")


def work_point(s, y, n_pos, target=0.95):
    """precision>=target 的最大召回工作点：取**最低**的达标阈值（同 deploy）。"""
    for th in np.unique(np.round(s, 3)):
        sel = s >= th
        tp = int((sel & (y == 1)).sum())
        fp = int((sel & (y == 0)).sum())
        if tp and tp / (tp + fp) >= target:
            return float(th), tp / (tp + fp), tp / n_pos
    return None


n_pos = int(y.sum())
print("\n概率加权口径（线上同构）—— 全网格按 OOF AUC 排序 Top 12")
rows = []
for wa, wb, wf, wv in product(A_W, B_W, F_W, V_W):
    if wa + wb + wf + wv <= 0:
        continue
    s = (wa * P["a"] + wb * P["b"] + wf * P["flow"] + wv * P["vm"]) / (wa + wb + wf + wv)
    rows.append((roc_auc_score(y, s), (wa, wb, wf, wv), s))
rows.sort(key=lambda r: -r[0])
for auc, w, s in rows[:12]:
    wp = work_point(s, y, n_pos)
    tag = "  ← 现役" if w == tuple(INCUMBENT[k] for k in ("a", "b", "flow", "vm")) else ""
    print(f"  a{w[0]:<5} b{w[1]:<4} f{w[2]:<4} v{w[3]:<4} AUC={auc:.4f}"
          + (f"   p95 th={wp[0]:.3f} 精度={wp[1]:.3f} 召回={wp[2]:.3f}" if wp
             else "   p95不可达") + tag)

best_auc, best_w, best_s = rows[0]
print(f"\n最优: a{best_w[0]} b{best_w[1]} f{best_w[2]} v{best_w[3]}  AUC={best_auc:.4f}")
wp = work_point(best_s, y, n_pos)
if wp:
    print(f"  p95 工作点: keep_thr={wp[0]:.3f}  精度={wp[1]:.3f}  召回={wp[2]:.3f}")
inc = [r for r in rows if r[1] == tuple(INCUMBENT[k] for k in ("a", "b", "flow", "vm"))]
if inc:
    auc_i, _, s_i = inc[0]
    wp_i = work_point(s_i, y, n_pos)
    print(f"现役 a.5 b2 f1 v1  AUC={auc_i:.4f}"
          + (f"  p95 工作点: keep_thr={wp_i[0]:.3f} 精度={wp_i[1]:.3f} 召回={wp_i[2]:.3f}"
             if wp_i else "  p95不可达"))
