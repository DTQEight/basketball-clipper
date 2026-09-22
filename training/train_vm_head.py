# -*- coding: utf-8 -*-
"""VM 臂换头：把 LGBM 头换成时序头，并（可选）改用逐时序位置特征保留运动结构。

背景：线上 VM 臂 = VideoMAE 冻结主干 → 全 token 均值池化 768 维 → LGBM。
分诊发现两处浪费：
  1. 全局均值池化把 16 帧（=8 个 tubelet 时序位置）的运动信息整体拍平，
     时空自注意力的优势没用上 → `--temporal` 特征（8,768）保留时序轴；
  2. 头只用 orig 特征、且是 LGBM（无增强变体、无非线性）→ 换 TemporalNet。

本脚本内部把所有候选头在同一分折/增强协议下比完，按**集成 OOF AUC**择优，
再按中位 epoch 全量重训、存自包含检查点（服务端只认这一个文件）。

特征口径（服务端必须照抄）：
    16 帧筐心块 → VideoMAE v2，(x/255-0.5)/0.5
    feat="mean"      last_hidden_state 全 token 均值 → (768,)
    feat="temporal"  按 token 索引 t*196+h*14+w 还原为 (8,196,768) → 空间取均值 → (8,768)

用法：env\\Scripts\\python.exe training\\train_vm_head.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.extract_frames_b import load_dataset_events  # noqa: E402
from training.train_temporal import (  # noqa: E402
    AUG_VARIANTS, build_folds_by_game, norm_game, report_metrics,
    train_b_oof, train_final_model)

TR = PROJECT_ROOT / "training"
FEAT_MEAN = TR / "vm_feat"
FEAT_TEMPORAL = TR / "vm_feat_t"
OUT = TR / "model_vm_head.pt"
META_OUT = TR / "model_vm_head_meta.json"
OOF_OUT = TR / "oof_vm_t.jsonl"
PREPROCESS = "VideoMAE v2：(x/255-0.5)/0.5，16 帧 → last_hidden_state 逐时序位置池化"

# (标签, 特征模式, 头结构, 是否用增强变体)
CONFIGS = [
    ("mean + pool 头", "mean", "pool", True),
    ("mean + bigru 头", "mean", "bigru", True),
    ("temporal + bigru 头", "temporal", "bigru", True),
    ("temporal + pool 头", "temporal", "pool", True),
    ("temporal + bigru（仅 orig）", "temporal", "bigru", False),
]


def load_feat(events, mode, variant="orig"):
    root = FEAT_MEAN if mode == "mean" else FEAT_TEMPORAL
    sub = root if variant == "orig" else root / variant
    arrs = []
    for ev in events:
        p = sub / f"{ev['event_id']}.npz"
        if not p.exists():
            raise FileNotFoundError(f"缺 VM 特征 {p}")
        x = np.load(p)["x"].astype(np.float32)
        arrs.append(x[None, :] if x.ndim == 1 else x)   # mean→(1,768) / temporal→(8,768)
    return np.stack(arrs)


def rj(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def work_point(y, s, n_pos, target=0.95):
    for th in np.unique(np.round(s, 3)):
        sel = s >= th
        tp = int((sel & (y == 1)).sum())
        fp = int((sel & (y == 0)).sum())
        if tp and tp / (tp + fp) >= target:
            return float(th), tp / (tp + fp), tp / n_pos
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    events = load_dataset_events()
    y = np.array([1.0 if e["label"] == "pos" else 0.0 for e in events], dtype=np.float32)
    games = np.array([norm_game(e["video"]) for e in events])
    folds = build_folds_by_game([e["video"] for e in events])
    n_pos = int(y.sum())
    print(f"{len(y)} 事件（正 {n_pos}）  比赛日 {len(set(games.tolist()))} 个  "
          f"→ {len(folds)} 折\n", flush=True)

    # 其余三臂的 OOF（用于算集成影响，与线上概率加权口径一致）
    d = {r["event_id"]: r for r in rj(TR / "oof_directions.jsonl")}
    nb = {r["event_id"]: r["pred"] for r in rj(TR / "oof_temporal_simclr.jsonl")}
    nf = {r["event_id"]: r["pred"] for r in rj(TR / "oof_flow_simclr.jsonl")}
    ids = [e["event_id"] for e in events]
    oa = np.array([d[k]["pred_a"] for k in ids])
    ob = np.array([nb[k] for k in ids])
    of = np.array([nf[k] for k in ids])
    W = {"a": .5, "b": 2, "fn": 1, "vm": 1}

    cache = {}
    print(f"{'候选头':<30}{'单臂 AUC':>10}{'集成 AUC':>10}{'p95 th':>9}{'精度':>8}{'召回':>8}")
    print(f"{'LGBM（线上现状）':<30}{0.8208:>10.4f}"
          + f"{0.9640:>10.4f}{0.640:>9.3f}{0.951:>8.3f}{0.783:>8.3f}")
    results = []
    for tag, mode, arch, use_aug in CONFIGS:
        Xs = cache.get((mode, use_aug))
        if Xs is None:
            Xs = {v: load_feat(events, mode, v) for v in ("orig",) + AUG_VARIANTS}
            cache[(mode, use_aug)] = Xs
        oof, fold_aucs, eps = train_b_oof(Xs, y, games, folds, arch=arch,
                                          use_aug=use_aug, verbose=False)
        vm = np.array(oof)
        ens = (W["a"] * oa + W["b"] * ob + W["fn"] * of + W["vm"] * vm) / sum(W.values())
        from sklearn.metrics import roc_auc_score
        auc_vm, auc_ens = roc_auc_score(y, vm), roc_auc_score(y, ens)
        wp = work_point(y, ens, n_pos)
        print(f"{tag:<30}{auc_vm:>10.4f}{auc_ens:>10.4f}"
              + (f"{wp[0]:>9.3f}{wp[1]:>8.3f}{wp[2]:>8.3f}" if wp else "   p95 不可达"),
              flush=True)
        results.append({"tag": tag, "mode": mode, "arch": arch, "use_aug": use_aug,
                        "auc_vm": auc_vm, "auc_ens": auc_ens, "wp": wp,
                        "oof": vm, "fold_aucs": fold_aucs, "eps": eps})

    best = max(results, key=lambda r: r["auc_ens"])
    print(f"\n择优（按集成 OOF AUC）: {best['tag']}  "
          f"集成 {best['auc_ens']:.4f}（线上现状 0.9640）/ 单臂 {best['auc_vm']:.4f}")
    report_metrics(y, best["oof"], f"VM 终模型 [{best['tag']}]")

    OOF_OUT.write_text("\n".join(json.dumps(
        {"event_id": e["event_id"], "video": e["video"], "ts": e["ts"],
         "label": int(lab), "pred": round(float(p), 6)}, ensure_ascii=False)
        for e, lab, p in zip(events, y, best["oof"])), encoding="utf-8")

    Xs = cache[(best["mode"], best["use_aug"])]
    eps = max(5, int(np.median(best["eps"])))
    print(f"\n全量重训 [{best['arch']}]（{eps} epochs）...", flush=True)
    net = train_final_model(Xs, y, best["arch"], best["use_aug"], eps)

    steps = int(Xs["orig"].shape[1])
    torch.save({
        "heads": [{"arch": best["arch"], "final_epochs": eps,
                   "state_dict": net.state_dict()}],
        "dim": int(Xs["orig"].shape[-1]),
        "feat": best["mode"],
        "steps": steps,
        "hidden": 128,
        "preprocess": PREPROCESS,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "oof_auc": round(float(best["auc_vm"]), 4),
    }, OUT)
    META_OUT.write_text(json.dumps({
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": "VideoMAE v2 base (MCG-NJU/videomae-base-finetuned-kinetics)，冻结",
        "head": f"TemporalNet[{best['arch']}]  · 特征模式 {best['mode']}"
                + ("（含增强变体）" if best["use_aug"] else "（仅 orig）"),
        "n_events": len(y), "n_pos": n_pos,
        "fold_aucs": [round(float(a), 4) for a in best["fold_aucs"]],
        "final_epochs": eps,
        "baseline_lgbm_mean_oof": 0.8208,
        "oof_auc": round(float(best["auc_vm"]), 4),
        "candidates": [{k: v for k, v in r.items() if k not in ("oof", "fold_aucs", "eps")}
                       for r in results],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存: {OUT}\n已保存: {META_OUT}\n已保存: {OOF_OUT}")


if __name__ == "__main__":
    main()
