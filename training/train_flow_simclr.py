# -*- coding: utf-8 -*-
"""Flow 臂换骨干：用光流域 SimCLR 自监督编码器替换 ImageNet ResNet18 重训时序模型。

动机：Flow 臂的输入是 Farneback 光流幅度图（灰度、运动边缘主导），
与 ImageNet 的自然图像分布相差比彩色帧更远——用 ImageNet 冻结特征提这种图，
失配比 B 臂当年还严重。B 臂换域内自监督骨干后单臂 OOF 从 0.6322 涨到 0.9217，
本脚本对 Flow 臂做同一件事。

与 train_temporal 完全同折同口径（build_folds_by_game 按比赛日分组、增强变体
只进训练折、OOF 择 epoch 后全量重训），因此 OOF 数值可与 Flow 臂（ImageNet，
OOF 0.8593）直接对比。

**只换骨干，不动结构**：Flow 臂现役是单个 bigru 头（B 臂的双头是 B 自己的设计），
本脚本保持单 bigru，把「换骨干」隔离成唯一变量。

特征口径（必须与 simclr_pretrain.pretrain --flow 一致，服务端也要照抄）：
    flow_b/{eid}.npz 的 mag (15,224,224) uint8 → 复制成 3 通道 → /255
    → **不做** ImageNet mean/std（编码器是在裸 /255 上自监督训出来的）
    三通道相同，故 BGR→RGB 反转是空操作

产物：training/model_flow_simclr.pt —— 自包含（SimCLR 骨干 + bigru 时序头 +
      预处理口径），服务端 goal_verifier 优先加载它，缺失时退回 ImageNet 老口径。
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

FEAT = PROJECT_ROOT / "training" / "flow_feat_simclr"
BACKBONE_CKPT = PROJECT_ROOT / "training" / "flow_simclr_resnet18.pt"
OUT = PROJECT_ROOT / "training" / "model_flow_simclr.pt"
META_OUT = PROJECT_ROOT / "training" / "model_flow_simclr_meta.json"
OOF_OUT = PROJECT_ROOT / "training" / "oof_flow_simclr.jsonl"

ARCH = "bigru"          # 与现役 Flow 臂一致（单头），把「换骨干」隔离成唯一变量
PREPROCESS = "光流幅度 /255（不做 ImageNet mean/std；三通道相同）"


def load_feat(events, variant="orig"):
    """按 events 顺序载入光流 SimCLR 特征 (N,15,512)。"""
    sub = FEAT if variant == "orig" else FEAT / variant
    arrs = []
    for ev in events:
        p = sub / f"{ev['event_id']}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"缺光流 SimCLR 特征 {p}，先跑 simclr_pretrain.py --flow")
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
          f"  比赛日 {len(set(games.tolist()))} 个  "
          f"序列长={Xs['orig'].shape[1]}  dim={Xs['orig'].shape[-1]}")

    folds = build_folds_by_game([e["video"] for e in events])
    print(f"→ {len(folds)} 折（与现役 Flow 臂 ImageNet 完全同折）\n", flush=True)

    print(f"== Flow-SimCLR[{ARCH}] ==", flush=True)
    oof, fold_aucs, best_eps = train_b_oof(Xs, y, games, folds, arch=ARCH,
                                           use_aug=True)
    metrics = report_metrics(y, oof, f"Flow_simclr[{ARCH}]")
    print(f"\n对照：Flow(ImageNet ResNet18) 现役 OOF = 0.8593")

    OOF_OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in (
        {"event_id": e["event_id"], "video": e["video"], "ts": e["ts"],
         "label": int(yy), "pred": float(p)}
        for e, yy, p in zip(events, y, oof))), encoding="utf-8")

    backbone = torch.load(BACKBONE_CKPT, map_location="cpu",
                          weights_only=False)
    eps = max(5, int(np.median(best_eps)))
    print(f"\n全量重训 [{ARCH}]（{eps} epochs）...", flush=True)
    net = train_final_model(Xs, y, ARCH, True, eps)

    torch.save({
        "backbone": backbone["backbone"],
        "backbone_pool": backbone.get("pool"),
        "nets": [{"arch": ARCH, "final_epochs": eps,
                  "state_dict": net.state_dict()}],
        "dim": int(Xs["orig"].shape[-1]),
        "hidden": 128,
        "frames": int(Xs["orig"].shape[1]),
        "frame_offsets": FRAME_OFFS, "zoom": ZOOM, "size": SIZE,
        "preprocess": PREPROCESS,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "oof_auc": metrics.get("oof_auc"),
    }, OUT)
    META_OUT.write_text(json.dumps({
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": "光流域 SimCLR 自监督 resnet18"
                    f"（{backbone.get('pool')}，无标签）",
        "n_events": int(len(y)), "n_pos": int(y.sum()),
        "arch": ARCH,
        "fold_aucs": [round(float(x), 4) for x in fold_aucs],
        "final_epochs": eps,
        "baseline_flow_imagenet_oof": 0.8593,
        **metrics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {OUT}")
    print(f"已保存: {META_OUT}")
    print(f"OOF 明细: {OOF_OUT}")


if __name__ == "__main__":
    main()
