# -*- coding: utf-8 -*-
"""训练后误差分析（一次性脚本）：按视频 AUC + 误报/漏检特征画像。"""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_lgbm import load_data, N_FOLDS, SEED  # noqa: E402

import lightgbm as lgb  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

rows, feat_keys, X, y, videos = load_data()
gkf = GroupKFold(n_splits=min(N_FOLDS, len(set(videos))))
oof = np.zeros(len(y))
for tr, va in gkf.split(X, y, groups=videos):
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                             min_child_samples=10, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=SEED, verbose=-1)
    w = np.ones(len(tr))
    w[y[tr] == 1] = 291 / 215
    clf.fit(X[tr], y[tr], sample_weight=w)
    oof[va] = clf.predict_proba(X[va])[:, 1]

print("— 按视频 OOF AUC（样本>=8 且两类都有）—")
by_v = collections.defaultdict(list)
for i, v in enumerate(videos):
    by_v[v].append(i)
for v in sorted(by_v, key=lambda k: -len(by_v[k])):
    idx = by_v[v]
    if len(idx) >= 8 and len(set(y[idx])) > 1:
        auc = roc_auc_score(y[idx], oof[idx])
        print(f"  {Path(v).name:<28} n={len(idx):<4} pos={int(y[idx].sum()):<4} AUC={auc:.3f}")

SHOW = ("net_post", "net_ratio", "cross_progress", "post_drop_depth",
        "x_off_final", "conf_mean", "vy_down_rate", "min_dist_center",
        "below_in_x_rate", "bounce_up_after_band", "vx_drop_ratio")

print("\n— 最危险的 10 个误报（p>=0.5 实际不是进球）—")
fp_idx = sorted((i for i in range(len(y)) if y[i] == 0 and oof[i] >= 0.5),
                key=lambda i: -oof[i])
for i in fp_idx[:10]:
    r = rows[i]
    kv = "  ".join(f"{k}={r[k]:.2f}" if isinstance(r[k], float) else f"{k}={r[k]}"
                   for k in SHOW)
    print(f"  p={oof[i]:.2f} {kv}")

print("\n— 最可惜的 10 个漏检（实际进球但 p<0.5）—")
fn_idx = sorted((i for i in range(len(y)) if y[i] == 1 and oof[i] < 0.5),
                key=lambda i: oof[i])
for i in fn_idx[:10]:
    r = rows[i]
    kv = "  ".join(f"{k}={r[k]:.2f}" if isinstance(r[k], float) else f"{k}={r[k]}"
                   for k in SHOW)
    print(f"  p={oof[i]:.2f} {kv}")
