# -*- coding: utf-8 -*-
"""在刷新后的数据集上重算各臂 OOF（**只测量，不写任何部署产物**）。

背景：dataset_v1.json 已改为从 cache/history 的标注池重建（见 build_dataset.py），
事件数 831→870、且 793 个 event_id 沿用以复用既有特征。本脚本回答两件事：

  1) 换到刷新数据后，现役四臂口径的 OOF 与工作点阈值是否变化；
  2) VM 臂换时序头（training/exp_vm_head/）在刷新数据上是否仍成立。

为可比性，同时给出「新旧数据共有事件子集」上的 OOF AUC（跨事件集直接比 AUC 不严谨）。

**不写 model_*.pt / model_temporal_meta.json 等部署产物**，只输出：
    training/refresh_oof.jsonl     逐事件四臂 OOF（供后续标定复用）
    training/refresh_report.json   指标汇总

用法：env\\Scripts\\python.exe training\\refresh_oof.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.extract_frames_b import load_dataset_events  # noqa: E402
from training.train_temporal import (  # noqa: E402
    AUG_VARIANTS, build_folds_by_game, norm_game, report_metrics, train_b_oof)

TR = PROJECT_ROOT / "training"
FEAT_B_SIMCLR = TR / "frames_b_feat_simclr"
FLOW_FEAT_SIMCLR = TR / "flow_feat_simclr"
VM_FEAT = TR / "vm_feat"
VM_FEAT_T = TR / "vm_feat_t"
A_FEATURES = TR / "features.jsonl"
OLD_OOF = TR / "oof_directions.jsonl"
OOF_OUT = TR / "refresh_oof.jsonl"
REPORT = TR / "refresh_report.json"

# 与 train_lgbm.py 一致
LGBM_KW = dict(n_estimators=400, learning_rate=0.05, num_leaves=15,
               min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
               reg_lambda=1.0, random_state=42, verbose=-1)
# 线上现役集成权重
W = {"a": 0.5, "b": 2.0, "flow": 1.0, "vm": 1.0}


def load_seq(root, events, variant="orig"):
    sub = root if variant == "orig" else root / variant
    arrs = []
    for e in events:
        p = sub / f"{e['event_id']}.npz"
        if not p.exists():
            raise FileNotFoundError(f"缺特征 {p}")
        arrs.append(np.load(p)["x"].astype(np.float32))
    return np.stack(arrs)


def load_seq_variants(root, events):
    return {v: load_seq(root, events, v) for v in ("orig",) + AUG_VARIANTS}


def load_a(events):
    """features.jsonl → (N,35) 矩阵（列取并集，缺列补 0）。"""
    rows = {}
    for line in A_FEATURES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["event_id"]] = r
    miss = [e["event_id"] for e in events if e["event_id"] not in rows]
    if miss:
        print(f"[A] 缺 {len(miss)}/{len(events)} 个事件 → A 臂跳过")
        return None
    keys = sorted(k for k in rows[events[0]["event_id"]]
                  if k not in {"event_id", "label", "video", "ts"})
    print(f"[A] 特征列 {len(keys)}")
    return np.array([[float(rows[e["event_id"]].get(k, 0.0)) for k in keys]
                     for e in events], dtype=np.float32)


def lgbm_oof(X, y, games, folds):
    """按比赛日分折的 LGBM OOF（少数类加权，同 train_lgbm.py）。"""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    games = np.asarray(games)
    oof = np.zeros(len(y), dtype=np.float32)
    aucs = []
    for val_games in folds:
        va = np.isin(games, list(val_games))
        tr = ~va
        w = np.ones(int(tr.sum()))
        w[y[tr] == 1] = max(1.0, float((y[tr] == 0).sum())
                            / max(float((y[tr] == 1).sum()), 1.0))
        clf = lgb.LGBMClassifier(**LGBM_KW)
        clf.fit(X[tr], y[tr], sample_weight=w)
        oof[va] = clf.predict_proba(X[va])[:, 1]
        aucs.append(float(roc_auc_score(y[va], oof[va])))
    return oof, aucs


def work_point(y, s, target=0.95):
    for th in np.unique(np.round(s, 3)):
        sel = s >= th
        tp = int((sel & (y == 1)).sum())
        fp = int((sel & (y == 0)).sum())
        if tp and tp / (tp + fp) >= target:
            return float(th), tp / (tp + fp), tp / int(y.sum())
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    events = load_dataset_events()
    y = np.array([1.0 if e["label"] == "pos" else 0.0 for e in events], dtype=np.float32)
    games = np.array([norm_game(e["video"]) for e in events])
    folds = build_folds_by_game([e["video"] for e in events])
    print(f"{len(y)} 事件（正 {int(y.sum())}）  {len(set(games.tolist()))} 比赛日  "
          f"→ {len(folds)} 折\n")

    P = {}
    Xa = load_a(events)
    if Xa is not None:
        P["a"], _ = lgbm_oof(Xa, y, games, folds)
        report_metrics(y, P["a"], "A(LGBM)")

    Xb = load_seq_variants(FEAT_B_SIMCLR, events)
    o1, _, _ = train_b_oof(Xb, y, games, folds, arch="bigru", verbose=False)
    o2, _, _ = train_b_oof(Xb, y, games, folds, arch="pool", verbose=False)
    P["b"] = (o1 + o2) / 2
    report_metrics(y, P["b"], "B(SimCLR 双头均值)")

    Xf = load_seq_variants(FLOW_FEAT_SIMCLR, events)
    P["flow"], _, _ = train_b_oof(Xf, y, games, folds, arch="bigru", verbose=False)
    report_metrics(y, P["flow"], "Flow(SimCLR)")

    P["vm_lgbm"], _ = lgbm_oof(load_seq(VM_FEAT, events), y, games, folds)
    report_metrics(y, P["vm_lgbm"], "VM(LGBM 旧头)")

    if all((VM_FEAT_T / f"{e['event_id']}.npz").exists() for e in events):
        Xv = load_seq_variants(VM_FEAT_T, events)
        P["vm_t"], _, _ = train_b_oof(Xv, y, games, folds, arch="pool", verbose=False)
        report_metrics(y, P["vm_t"], "VM(时序头 新)")
    else:
        print("\n[跳过] VM 时序头对比：缺 vm_feat_t（该实验已结案不部署，"
              "故未抽该变体；见 training/exp_vm_head/）")

    # ===== 集成：现役口径下两种 VM 头 =====
    print(f"\n{'配置':<24}{'集成 AUC':>10}{'p95 th':>9}{'精度':>8}{'召回':>8}")
    ens = {}
    cfgs = [("现役（VM=LGBM 旧头）", "vm_lgbm")]
    if "vm_t" in P:
        cfgs.append(("换成 VM 时序头", "vm_t"))
    for tag, vmkey in cfgs:
        s = (W["a"] * P.get("a", 0) + W["b"] * P["b"] + W["flow"] * P["flow"]
             + W["vm"] * P[vmkey]) / (sum(W.values()) if "a" in P else 3.0)
        ens[vmkey] = s
        r = work_point(y, ens[vmkey])
        print(f"{tag:<24}{roc_auc_score(y, s):>10.4f}"
              + (f"{r[0]:>9.3f}{r[1]:>8.3f}{r[2]:>8.3f}" if r else "  p95 不可达"))

    # ===== 与刷新前的同子集对比（cross-event-set 比 AUC 不严谨）=====
    ids = [e["event_id"] for e in events]
    if OLD_OOF.exists():
        old = {r["event_id"]: r for r in
               (json.loads(l) for l in OLD_OOF.read_text(encoding="utf-8").splitlines()
                if l.strip())}
        common = [i for i in ids if i in old]
        m = np.isin(ids, common)
        print(f"\n新旧数据集共有事件 {len(common)} 个，在该子集上对比 OOF AUC：")
        print(f"  {'臂':<16}{'刷新前':>9}{'刷新后':>9}")
        for k, ok in (("a", "pred_a"), ("b", "pred_b"), ("flow", "pred_flow_t"),
                      ("vm_lgbm", "pred_vm")):
            if k not in P:
                continue
            yo = np.array([old[i]["label"] for i in common])
            print(f"  {k:<16}{roc_auc_score(yo, [old[i][ok] for i in common]):>9.4f}"
                  f"{roc_auc_score(y[m], P[k][m]):>9.4f}")

    lines = []
    for i, (e, lab) in enumerate(zip(events, y)):
        rec = {"event_id": e["event_id"], "video": e["video"], "ts": e["ts"],
               "label": int(lab)}
        for k, v in P.items():
            if len(v) == len(y):
                rec[k] = round(float(v[i]), 6)
        lines.append(json.dumps(rec, ensure_ascii=False))
    OOF_OUT.write_text("\n".join(lines), encoding="utf-8")
    REPORT.write_text(json.dumps({
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_events": int(len(y)), "n_pos": int(y.sum()),
        "n_games": len(set(games.tolist())),
        "arm_oof_auc": {k: round(float(roc_auc_score(y, v)), 4) for k, v in P.items()},
        "ensemble_auc": {k: round(float(roc_auc_score(y, v)), 4) for k, v in ens.items()},
        "ensemble_p95": {k: work_point(y, v) for k, v in ens.items()},
        "weights": W,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {OOF_OUT.name} / {REPORT.name}（耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
