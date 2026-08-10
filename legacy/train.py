"""YOLOv8 模型训练/微调脚本。

数据集获取方式（推荐 Roboflow Universe）：
  1. 打开 https://universe.roboflow.com/
  2. 搜索 "basketball hoop" 或 "basketball detection"
  3. 选一个含 basketball + hoop 两类的数据集，导出格式选 YOLOv8
  4. 下载解压后得到 data.yaml，里面指定了 train/val 路径和类别

下载好数据集后，把 data.yaml 路径填到下面 --data 参数即可训练。

示例:
    python train.py --data datasets/hoop/data.yaml --epochs 100 --img 640
    # GTX 1650 4G 用 yolov8n，batch 别太大
    python train.py --data datasets/hoop/data.yaml --epochs 100 --img 640 --batch 8
"""
import argparse
from ultralytics import YOLO


def train(data_yaml, epochs=100, img=640, batch=16, base="yolov8n.pt"):
    """从官方预训练权重微调。

    base: 基础权重。GTX 1650 4G 显存有限，建议用 yolov8n.pt。
    """
    model = YOLO(base)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img,
        batch=batch,
        device="cuda:0",
        project="runs",
        name="basketball",
    )
    print("训练完成，权重在 runs/basketball/weights/best.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="训练 YOLOv8 篮球检测模型")
    ap.add_argument("--data", required=True, help="数据集 data.yaml 路径")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", default="yolov8n.pt", help="基础预训练权重")
    args = ap.parse_args()
    train(args.data, args.epochs, args.img, args.batch, args.base)
