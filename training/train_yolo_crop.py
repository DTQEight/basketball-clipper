# -*- coding: utf-8 -*-
"""YOLO 篮球检测微调——**裁剪档配方**（训练 imgsz=640，与线上主档同尺度）。

与 training/train_yolo_ft.py（整帧配方：imgsz 768 / 25 轮 / fraction 0.6）并列：
本脚本吃的数据由 training/build_yolo_crop_dataset.py 产出（整帧 + 部署几何裁剪块），
目的是让模型在中见过「筐接受框裁出来的竖版小图」，堵住「裁剪档把真球置信度压低 →
整窗无证据 → 静默漏球」的缺口（详见 doc/BENCHMARKS.md「③」）。

微调起点仍是 weights/basketball_ft.pt（**不动线上权重**，产物落 training/runs/）。

用法:
  env\\Scripts\\python.exe -u training\\train_yolo_crop.py                      # 6 轮 / batch 4 / fraction 1.0
  env\\Scripts\\python.exe -u training\\train_yolo_crop.py --epochs=6 --fraction=0.4
  env\\Scripts\\python.exe -u training\\train_yolo_crop.py --data=yolo_crop_data_10v
产物: training/runs/basketball_ft_crop_<tag>_e<轮数>/weights/best.pt
      <tag> 取自 --data（默认把 yolo_crop_data 前缀去掉：yolo_crop_data → full，
      yolo_crop_data_10v → 10v），用来区分不同样本集，避免同名 run 互相覆盖。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "weights" / "basketball_ft.pt"


def _tag_of(data_name: str) -> str:
    """样本集标识：yolo_crop_data → full，yolo_crop_data_10v → 10v。"""
    t = data_name[len("yolo_crop_data"):].lstrip("_")
    return t or "full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="yolo_crop_data",
                    help="training/ 下的数据集目录名（须含 data.yaml）")
    ap.add_argument("--batch", type=int, default=4,
                    help="batch 4：GTX 1650 4GB 下 batch 8 会 OOM（AdamW 状态额外占显存）")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="每轮子采样比例。全库 7202 张用 0.4 ≈ 2881 张/轮，"
                         "只减每场帧数、不影响视频覆盖")
    ap.add_argument("--tag", default=None, help="样本集标识（默认由 --data 推出）")
    args = ap.parse_args()

    data = PROJECT_ROOT / "training" / args.data / "data.yaml"
    if not data.exists():
        print("ERROR: 找不到 %s（先用 build_yolo_crop_dataset.py 生成）" % data)
        sys.exit(1)
    tag = args.tag or _tag_of(args.data)
    run_name = "basketball_ft_crop_%s_e%d" % (tag, args.epochs)

    from ultralytics import YOLO

    model = YOLO(str(BASE))
    model.train(
        data=str(data), epochs=args.epochs, imgsz=640, batch=args.batch, device=0,
        # workers=0：本机 C: 剩余空间小，spawn 出来的 dataloader 子进程各自加载一遍
        # CUDA DLL 会报 WinError 1455（页面文件太小）直接崩。标签 .cache 已生成、
        # 训练图在 E: 本盘时单进程读图足够快。
        # 另：训练期间必须机器独占——并发跑评估会顶爆系统内存，主进程读图会
        # 抛 cv2.error: Insufficient memory。
        workers=0, patience=20, cos_lr=True, lr0=0.001, lrf=0.01,
        fraction=args.fraction,
        # 与线上一致：远机位球小，禁用 mosaic（会把小球再缩一半）；其余增强保守
        mosaic=0.0, scale=0.2, translate=0.05, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.0,
        project=str(PROJECT_ROOT / "training" / "runs"),
        name=run_name, exist_ok=True, save_period=5, val=True,
    )
    print("best:", PROJECT_ROOT / "training" / "runs" / run_name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
