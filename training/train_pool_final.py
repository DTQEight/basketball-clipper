# -*- coding: utf-8 -*-
"""补训 B 臂 pool 终模型（model_temporal_pool.pt）到最新事件池。

背景：train_temporal.py 主流程只正式存盘 bigru；pool 终模型历史上由
临时脚本产出，收编新数据后需同步补训，否则线上 pool 臂停在旧事件池。
口径与 train_temporal 正式版一致：增强变体只进训练折，OOF 择 epoch 中位数
后全量重训。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from training.extract_frames_b import load_dataset_events  # noqa: E402
from training.train_temporal import (  # noqa: E402
    AUG_VARIANTS, FEAT_DIR, FRAME_OFFS, SIZE, ZOOM,
    build_folds_by_game, load_feature_matrix, norm_game, report_metrics,
    train_b_oof, train_final_model)

POOL_OUT = PROJECT_ROOT / "training" / "model_temporal_pool.pt"
POOL_META_OUT = PROJECT_ROOT / "training" / "model_temporal_pool_meta.json"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    events = load_dataset_events()
    variants = ("orig",) + AUG_VARIANTS
    Xs = {v: load_feature_matrix(events, v) for v in variants}
    y = np.array([1.0 if e["label"] == "pos" else 0.0 for e in events],
                 dtype=np.float32)
    games = np.array([norm_game(e["video"]) for e in events])
    print(f"载入: {len(y)} 事件（正 {int(y.sum())}）")
    folds = build_folds_by_game([e["video"] for e in events])
    oof, fold_aucs, best_eps = train_b_oof(Xs, y, games, folds,
                                           arch="pool", use_aug=True)
    metrics = report_metrics(y, oof, "B_pool(final)")
    eps = max(5, int(np.median(best_eps)))
    print(f"全量重训 pool（{eps} epochs）...")
    net = train_final_model(Xs, y, "pool", True, eps)
    ckpt = {
        "state_dict": net.state_dict(),
        "arch": "pool", "hidden": 128, "dim": 512,
        "frame_offsets": FRAME_OFFS, "zoom": ZOOM, "size": SIZE,
        "backbone": "torchvision resnet18 IMAGENET1K_V1 (fc→Identity)",
        "input": "BGR uint8 帧块 → RGB /255 → ImageNet mean/std",
        "aug_variants": AUG_VARIANTS,
    }
    torch.save(ckpt, POOL_OUT)
    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arch": "pool", "use_aug": True,
        "n_events": len(y), "n_pos": int(y.sum()),
        "n_games": len(set(games.tolist())),
        "fold_aucs": [round(a, 4) for a in fold_aucs],
        "final_epochs": eps, **metrics,
    }
    POOL_META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"已保存: {POOL_OUT}")
    print(f"已保存: {POOL_META_OUT}")


if __name__ == "__main__":
    main()
