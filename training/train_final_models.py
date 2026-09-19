# -*- coding: utf-8 -*-
"""训练并保存冠军集成 AB+Flow_T+VM 的全部运行时产物。

产物（均写入 training/）：
  model_flow.pt          Flow_T 时序头（全量重训，口径同 train_temporal 正式版）
  model_flow_meta.json   Flow_T 元信息（抽帧/光流参数 + OOF 指标）
  model_vm_lgbm.txt      VideoMAE 768 维 → LGBM（Booster 文本格式）
  model_stack.pkl        堆叠器（logistic，输入 [a, b, flow, vm] 四路概率）
  model_stack_novm.pkl   降级堆叠器（无 VM 时三路）
  model_stack_meta.json  堆叠器 OOF 指标 + 推荐阈值

堆叠器用 oof_directions.jsonl 的逐事件 OOF 预测拟合（基模型已是 OOF，
无泄漏）；堆叠器自身按比赛日分组 5 折评估，工作点诚实。
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TR = PROJECT_ROOT / "training"
OOF_DETAIL = TR / "oof_directions.jsonl"

MODEL_FLOW_OUT = TR / "model_flow.pt"
FLOW_META_OUT = TR / "model_flow_meta.json"
MODEL_VM_OUT = TR / "model_vm_lgbm.txt"
STACK_OUT = TR / "model_stack.pkl"
STACK_NOVM_OUT = TR / "model_stack_novm.pkl"
STACK_META_OUT = TR / "model_stack_meta.json"

# 运行时复现口径所需的全部常量（与 extract_motion / train_directions 一致）
FRAME_OFFS = [-1.5, -1.0, -0.5, -0.4, -0.3, -0.25, -0.2, -0.15,
              -0.1, -0.05, 0.0, 0.1, 0.3, 0.5, 0.8, 1.4]
ZOOM = 3.2
SIZE = 224
MAG_SCALE = 12.0


# ================= 1. Flow_T 最终模型 =================

def train_flow_final():
    from training.compare_ab import load_ab_data
    from training.train_temporal import (build_folds_by_game, report_metrics,
                                         train_b_oof, train_final_model)
    from training.train_directions import FEAT_DIRS, load_feat_matrix

    events, Xa, Xs_b, y, games, feat_keys, use_aug = load_ab_data()
    variants = ("orig", "vflip", "dark", "bright")
    Xs = {}
    for v in variants:
        m = load_feat_matrix(events, FEAT_DIRS["flow"], v)
        if m is None:
            print(f"ERROR: 缺光流特征变体 {v}，先跑 feat_flow_only.py")
            sys.exit(1)
        Xs[v] = m

    folds = build_folds_by_game([e["video"] for e in events])
    oof, fold_aucs, best_eps = train_b_oof(Xs, y, games, folds, arch="bigru",
                                           use_aug=use_aug, verbose=True)
    metrics = report_metrics(y, oof, "Flow_T(final OOF)")

    eps = max(5, int(np.median(best_eps)))
    print(f"\n全量重训 Flow_T（{eps} epochs）...")
    net = train_final_model(Xs, y, "bigru", use_aug, eps)
    import torch
    torch.save({
        "state_dict": net.state_dict(),
        "arch": "bigru", "hidden": 128, "dim": 512,
        "input": "Farneback 光流幅度图序列 (15,224,224) → 复制3通道 → /255 → ImageNet 归一化 → ResNet18(IMAGENET1K_V1, fc→Identity) 512 维",
        "frame_offsets": FRAME_OFFS, "zoom": ZOOM, "size": SIZE,
        "mag_scale": MAG_SCALE,
        "backbone": "torchvision resnet18 IMAGENET1K_V1 (fc→Identity)",
        "aug_variants": variants[1:],
    }, MODEL_FLOW_OUT)
    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_events": int(len(y)), "n_pos": int(y.sum()),
        "fold_aucs": [round(float(a), 4) for a in fold_aucs],
        "final_epochs": eps, "oof": metrics,
        "frame_offsets": FRAME_OFFS, "zoom": ZOOM, "size": SIZE,
        "mag_scale": MAG_SCALE,
    }
    FLOW_META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"已保存: {MODEL_FLOW_OUT.name} / {FLOW_META_OUT.name}")
    return oof


# ================= 2. VM LGBM =================

def train_vm_final():
    import lightgbm as lgb
    from training.compare_ab import load_ab_data
    from training.train_directions import VM_FEAT

    events, Xa, Xs_b, y, games, feat_keys, use_aug = load_ab_data()
    vm = []
    for e in events:
        p = VM_FEAT / f"{e['event_id']}.npz"
        if not p.exists():
            print("ERROR: 缺 VM 特征，先跑 extract_videomae.py")
            sys.exit(1)
        vm.append(np.load(p)["x"].astype(np.float32).ravel())
    X = np.stack(vm)

    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    w = np.ones(len(y))
    w[y == 1] = max(1.0, n_neg / max(n_pos, 1))
    clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, verbose=-1)
    clf.fit(X, y, sample_weight=w)
    clf.booster_.save_model(str(MODEL_VM_OUT))
    print(f"已保存: {MODEL_VM_OUT.name}（{X.shape[1]} 维，{len(y)} 事件）")


# ================= 3. 堆叠器 =================

def _eval_at_precision(y, p, target):
    """返回 (阈值, 实际精度, 召回)：满足精度 >= target 的最大召回点。"""
    y = np.asarray(y)
    order = np.argsort(-p)
    best = None
    tp = fp = 0
    n_pos = max(int(y.sum()), 1)
    for i, idx in enumerate(order):
        tp += y[idx] == 1
        fp += y[idx] == 0
        prec = tp / (i + 1)
        if prec >= target - 1e-9:
            best = (float(p[idx]), float(prec), tp / n_pos)
    return best


def _reject_threshold(y, p, target=0.96):
    """自动×阈值：score < thr 的样本中"真负例"精度 >= target 的最大 thr。"""
    y = np.asarray(y)
    order = np.argsort(p)
    n_neg = max(int((y == 0).sum()), 1)
    best = 0.0
    tn = fn = 0
    for i, idx in enumerate(order):
        tn += y[idx] == 0
        fn += y[idx] == 1
        prec_neg = tn / (i + 1)
        if prec_neg >= target - 1e-9:
            best = float(p[idx])
    return best


def train_stacker():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from training.train_temporal import build_folds_by_game, norm_game

    rows = [json.loads(l) for l in OOF_DETAIL.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    y = np.array([r["label"] for r in rows], dtype=np.float32)
    videos = [r["video"] for r in rows]
    games = np.array([norm_game(v) for v in videos])
    F4 = ["pred_a", "pred_b", "pred_flow_t", "pred_vm"]
    F3 = ["pred_a", "pred_b", "pred_flow_t"]
    X4 = np.array([[r[k] for k in F4] for r in rows], dtype=np.float32)
    X3 = np.array([[r[k] for k in F3] for r in rows], dtype=np.float32)

    folds = build_folds_by_game(videos)

    def stack_oof(X):
        oof = np.zeros(len(y), dtype=np.float32)
        for val_games in folds:
            va = np.isin(games, list(val_games))
            lr = LogisticRegression(C=1.0, max_iter=2000)
            lr.fit(X[~va], y[~va])
            oof[va] = lr.predict_proba(X[va])[:, 1]
        return oof

    oof4 = stack_oof(X4)
    oof3 = stack_oof(X3)
    auc4 = roc_auc_score(y, oof4)
    auc3 = roc_auc_score(y, oof3)

    from training.train_temporal import report_metrics
    m4 = report_metrics(y, oof4, "STACK(4路)")
    m3 = report_metrics(y, oof3, "STACK(3路,无VM)")

    # 最终堆叠器（全量拟合）
    lr4 = LogisticRegression(C=1.0, max_iter=2000).fit(X4, y)
    lr3 = LogisticRegression(C=1.0, max_iter=2000).fit(X3, y)
    with open(STACK_OUT, "wb") as f:
        pickle.dump(lr4, f)
    with open(STACK_NOVM_OUT, "wb") as f:
        pickle.dump(lr3, f)

    # 推荐阈值：keep = P90 工作点；reject = 负例精度 96% 的最大分
    r90 = _eval_at_precision(y, oof4, 0.90)
    r95 = _eval_at_precision(y, oof4, 0.95)
    rej = _reject_threshold(y, oof4, 0.96)
    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": F4, "features_novm": F3,
        "oof_auc_4": round(float(auc4), 4),
        "oof_auc_3_novm": round(float(auc3), 4),
        "metrics_4": m4, "metrics_3_novm": m3,
        "recommended_keep_thr": round(r90[0], 4) if r90 else None,
        "working_point_95": ({"threshold": round(r95[0], 4),
                              "precision": round(r95[1], 4),
                              "recall": round(r95[2], 4)} if r95 else None),
        "recommended_reject_thr": round(rej, 4),
        "note": "keep_thr=P90 工作点（自动√），reject_thr=负例精度96%（自动×）；堆叠器输入为四路概率 [A, B_ens, Flow_T, VM]",
    }
    STACK_META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n堆叠器: 4路 OOF AUC={auc4:.4f} / 3路(无VM)={auc3:.4f}")
    print(f"推荐阈值: keep={meta['recommended_keep_thr']} "
          f"reject={meta['recommended_reject_thr']}")
    print(f"已保存: {STACK_OUT.name} / {STACK_NOVM_OUT.name} / {STACK_META_OUT.name}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("== 1/3 Flow_T 最终模型 ==")
    train_flow_final()
    print("\n== 2/3 VM LGBM ==")
    train_vm_final()
    print("\n== 3/3 堆叠器 ==")
    train_stacker()


if __name__ == "__main__":
    main()
