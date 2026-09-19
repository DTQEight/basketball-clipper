# -*- coding: utf-8 -*-
"""YOLO 篮球检测微调：basketball_custom.pt 续训（GTX 1650 4GB 预算）。

数据：training/yolo_data（build_yolo_dataset.py 产出，val=留出比赛日盲测）
目标：拉起「穿网瞬间变形球」的置信度（0.06~0.17 → 0.3+），保住普通场景。

用法：env\\python.exe training\\train_yolo_ft.py [batch]
产物：training/runs/basketball_ft/weights/best.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "training" / "yolo_data" / "data.yaml"
BASE = PROJECT_ROOT / "weights" / "basketball_custom.pt"

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def main():
    from ultralytics import YOLO

    model = YOLO(str(BASE))
    model.train(
        # imgsz=768：960 会超 4G 物理显存触发 WDDM 共享内存溢出（7.6s/it）；
        # 768 训练在 1280 存储图上球 ≈8.4px，接近线上 960 推理 1920 视频的
        # 真实球尺寸（~7px），比 960 训练（10.5px）更贴部署分布
        data=str(DATA), epochs=25, imgsz=768, batch=batch, device=0,
        workers=2, patience=10, cos_lr=True, lr0=0.001, lrf=0.01,
        # 相邻 offset 帧冗余高，子采样 60% 足够覆盖全部事件
        fraction=0.6,
        # 微调保守增强：远机位球仅 ~14px；mosaic 会把小球再缩一半，禁用
        mosaic=0.0, scale=0.2, translate=0.05, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.0,
        project=str(PROJECT_ROOT / "training" / "runs"),
        name="basketball_ft", exist_ok=True, save_period=10, val=True,
    )
    print("best:", PROJECT_ROOT / "training" / "runs" / "basketball_ft"
          / "weights" / "best.pt")


if __name__ == "__main__":
    main()
