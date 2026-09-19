# -*- coding: utf-8 -*-
"""B 臂换骨干：用 SimCLR 域内自监督编码器替换 ImageNet ResNet18 重训时序模型。

动机：B 臂（ImageNet ResNet18 冻结特征 + BiGRU）在 831 事件上 OOF 只有
0.63，是四臂里最弱的；而 SimCLR 用本项目自有筐心裁剪自监督预训练，
已消除固定机位篮球场景的域差。

与 train_temporal 完全同折同口径（build_folds_by_game 按比赛日分组、增强
变体只进训练折、OOF 择 epoch 中位数后全量重训），因此 OOF 数值可与
B 臂（ResNet18）直接对比。

特征口径（必须与 simclr_pretrain.pretrain 一致，服务端也要照抄）：
    BGR uint8 帧块 → RGB /255 → **不做** ImageNet mean/std

产物：training/model_b_simclr.pt —— 自包含（含 SimCLR 骨干 + bigru/pool 两个
时序头 + 预处理参数），服务端 goal_verifier 加载它作为 B 臂，Flow 臂仍用
ImageNet ResNet18。
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

from training.extract_frames_b import (  # noqa: E402
    FRAME_OFFS, SIZE, ZOOM, load_dataset_events)
from training.train_temporal import (  # noqa: E402
    AUG_VARIANTS, build_folds_by_game, norm_game, report_metrics,
    train_b_oof, train_final_model)

FEAT_SIMCLR = PROJECT_ROOT / "training" / "frames_b_feat_simclr"
BACKBONE_CKPT = PROJECT_ROOT / "training" / "simclr_resnet18.pt"
OUT = PROJECT_ROOT / "training" / "model_b_simclr.pt"
META_OUT = PROJECT_ROOT / "training" / "model_b_simclr_meta.json"
OOF_OUT = PROJECT_ROOT / "training" / "oof_temporal_simclr.jsonl"


def load_feat(events, variant="orig"):
    """按 events 顺序载入 SimCLR 特征 (N,16,512)。"""
    sub = FEAT_SIMCLR if variant == "orig" else FEAT_SIMCLR / variant
    arrs = []
    for ev in events:
        p = sub / f"{ev['event_id']}.npz"
        if not p.exists():
            raise FileNotFoundError(f"缺 SimCLR 特征 {p}，先跑 simclr_pretrain.py --feat")
        arrs.append(np.load(p)["x"].astype(np.float32))
    return np.stack(arrs)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    events = load_dataset_events()
    variants = ("orig",) + AUG_VARIANTS
    Xs = {v: load_feat(events, v) for v in variants}
    y = np.array([1.0 if e["label"] == "pos" else 0.0 for e in events],
                 dtype=np.float32)
    games = np.array([norm_game(e["video"]) for e in events])
    print(f"载入: {len(y)} 事件（正 {int(y.sum())} / 负 {int((1 - y).sum())}）"
          f"  比赛日 {len(set(games.tolist()))} 个  dim={Xs['orig'].shape[-1]}")

    folds = build_folds_by_game([e["video"] for e in events])
    print(f"→ {len(folds)} 折（与 B 臂 ResNet18 完全同折）\n", flush=True)

    oofs, eps_by_arch, folds_by_arch = {}, {}, {}
    for arch in ("bigru", "pool"):
        print(f"== SimCLR-B[{arch}] ==", flush=True)
        oof, fold_aucs, best_eps = train_b_oof(Xs, y, games, folds, arch=arch,
                                               use_aug=True)
        oofs[arch] = oof
        eps_by_arch[arch] = best_eps
        folds_by_arch[arch] = fold_aucs

    # 线上 B 臂口径：双结构 sigmoid 概率均值（与 train_directions --deploy 一致）
    oof_b = (oofs["bigru"] + oofs["pool"]) / 2.0
    metrics = report_metrics(y, oof_b, "B_simclr(bigru+pool 均值)")
    for arch in ("bigru", "pool"):
        report_metrics(y, oofs[arch], f"B_simclr[{arch}] 单结构")
    print(f"\n对照：B(ResNet18) 线上口径 OOF = 0.6305")

    detail = [{"event_id": e["event_id"], "video": e["video"], "ts": e["ts"],
               "label": int(yy), "pred_bigru": float(a), "pred_pool": float(b),
               "pred": float(c)}
              for e, yy, a, b, c in zip(events, y, oofs["bigru"], oofs["pool"], oof_b)]
    OOF_OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in detail),
                       encoding="utf-8")

    backbone = torch.load(BACKBONE_CKPT, map_location="cpu",
                          weights_only=False)["backbone"]
    nets = []
    for arch in ("bigru", "pool"):
        eps = max(5, int(np.median(eps_by_arch[arch])))
        print(f"\n全量重训 [{arch}]（{eps} epochs）...", flush=True)
        net = train_final_model(Xs, y, arch, True, eps)
        nets.append({"arch": arch, "final_epochs": eps,
                     "state_dict": net.state_dict()})

    torch.save({
        "backbone": backbone,
        "nets": nets,
        "dim": int(Xs["orig"].shape[-1]),
        "hidden": 128,
        "frame_offsets": FRAME_OFFS, "zoom": ZOOM, "size": SIZE,
        "preprocess": "BGR uint8 帧块 → RGB /255（不做 ImageNet mean/std）",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "oof_auc": metrics.get("oof_auc"),
    }, OUT)
    META_OUT.write_text(json.dumps({
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": "SimCLR 自监督 resnet18（无标签，2026-08-29）",
        "n_events": int(len(y)), "n_pos": int(y.sum()),
        "fold_aucs": {a: [round(float(x), 4) for x in folds_by_arch[a]]
                      for a in folds_by_arch},
        "final_epochs": {n["arch"]: n["final_epochs"] for n in nets},
        **metrics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {OUT}")
    print(f"已保存: {META_OUT}")
    print(f"OOF 明细: {OOF_OUT}")


if __name__ == "__main__":
    main()
