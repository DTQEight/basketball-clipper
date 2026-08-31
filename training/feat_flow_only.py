# -*- coding: utf-8 -*-
"""只提光流序列的 ResNet18 特征（train_directions.run_feat 的定向版，
避免在 frames_c 未就绪时碰其他方向）。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train_directions import (  # noqa: E402
    extract_resnet_features, load_flow_maps, FEAT_DIRS)
from training.train_temporal import AUG_VARIANTS  # noqa: E402

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    extract_resnet_features(load_flow_maps, FEAT_DIRS["flow"],
                            ("orig",) + AUG_VARIANTS)
