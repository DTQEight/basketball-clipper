"""从用户视频自动抽帧 + 伪标注 + 训练 YOLOv8n 篮球检测模型。

流程：
  1. 从视频均匀抽取 N 帧
  2. 用 basketball_finetuned.pt + 极低阈值检测 Ball 类
  3. 用橙色 HSV 过滤误检（只保留含橙色像素的框）
  4. 生成 YOLO 格式数据集
  5. 训练 yolov8n 50 epochs
  6. 保存到 weights/basketball_custom.pt

用法:
    E:\bball-env\python.exe auto_train.py --video "path/to/video.mp4"
    E:\bball-env\python.exe auto_train.py --video "video.mp4" --frames 300 --epochs 50
"""
import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# 缓存目录
_CACHE_ROOT = r"E:\bball_cache"
os.environ["ULTRALYTICS_CONFIG_DIR"] = os.path.join(_CACHE_ROOT, "ultralytics")
os.environ["TORCH_HOME"] = os.path.join(_CACHE_ROOT, "torch")
os.makedirs(os.environ["ULTRALYTICS_CONFIG_DIR"], exist_ok=True)

from video_io import VideoReader, get_video_info

# 橙色篮球 HSV 范围
_ORANGE_LOW = np.array([5, 80, 80], dtype=np.uint8)
_ORANGE_HIGH = np.array([25, 255, 255], dtype=np.uint8)

# 篮球微调模型（用于自动标注）
_FINETUNED_WEIGHTS = str(ROOT / "weights" / "basketball_finetuned.pt")


def extract_frames(video_path, num_frames=300):
    """从视频均匀抽取 N 帧，用 ffmpeg seek 快速抽帧。

    返回 [(frame_idx, frame_bgr), ...]
    """
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"

    info = get_video_info(video_path)
    total = info["total"]
    fps = info["fps"]
    print(f"视频: {total} 帧, {fps:.1f} fps, {info['width']}x{info['height']}")

    # 均匀分布的帧号
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    indices = sorted(set(indices.tolist()))
    print(f"抽取 {len(indices)} 帧（用 ffmpeg seek）...")

    tmp_dir = os.path.join(_CACHE_ROOT, "train_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    frames = []
    for i, fidx in enumerate(indices):
        timestamp = fidx / fps  # 秒
        out_path = os.path.join(tmp_dir, f"frame_{fidx:06d}.jpg")
        # 用 -ss seek 到时间点，抽一帧
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-ss", f"{timestamp:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            out_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30,
                           creationflags=0x08000000)
            if os.path.exists(out_path):
                frame = cv2.imread(out_path)
                if frame is not None:
                    frames.append((fidx, frame))
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  已抽 {i+1}/{len(indices)} 帧")

    print(f"抽取完成: {len(frames)} 帧")
    return frames


def auto_label(frames, conf_threshold=0.01, imgsz=1280):
    """用 basketball_finetuned.pt 自动标注 Ball 类，再用橙色过滤误检。

    返回 [(frame_idx, frame, [(x1,y1,x2,y2), ...]), ...]
    """
    from ultralytics import YOLO
    model = YOLO(_FINETUNED_WEIGHTS)
    try:
        model.to("cuda:0")
    except Exception:
        pass

    labeled = []
    total_balls = 0

    for i, (fidx, frame) in enumerate(frames):
        # YOLO 检测
        res = model.predict(frame, conf=conf_threshold, imgsz=imgsz,
                            device="cuda:0", verbose=False)[0]
        balls = []
        if res.boxes is not None and len(res.boxes) > 0:
            clses = res.boxes.cls.cpu().numpy().astype(int)
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            names = res.names
            for j, c in enumerate(clses):
                n = names.get(c, "").lower()
                if "ball" in n:
                    x1, y1, x2, y2 = xyxy[j]
                    # 橙色过滤：检查框内是否有橙色像素
                    h, w = frame.shape[:2]
                    x1i, y1i = max(0, int(x1)), max(0, int(y1))
                    x2i, y2i = min(w, int(x2)), min(h, int(y2))
                    if x2i <= x1i or y2i <= y1i:
                        continue
                    roi = frame[y1i:y2i, x1i:x2i]
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, _ORANGE_LOW, _ORANGE_HIGH)
                    orange_ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])
                    if orange_ratio > 0.15:  # 橙色像素占比 > 15% 才保留
                        balls.append((float(x1), float(y1), float(x2), float(y2)))

        if balls:
            total_balls += len(balls)
        labeled.append((fidx, frame, balls))

        if (i + 1) % 50 == 0:
            print(f"  标注 {i+1}/{len(frames)} 帧, 累计 {total_balls} 个球")

    print(f"标注完成: {total_balls} 个球 / {len(frames)} 帧")
    return labeled


def color_label_fallback(frames, min_area=80, max_area=2000):
    """纯颜色标注回退方案：当 YOLO 检测不到球时，用橙色 HSV 找球。

    严格约束：
    - 圆度：宽高比 0.8-1.25（接近正方形/圆形，排除长条形的腿/手臂）
    - 面积：80-2000 像素（篮球在 1080p 视频里大约 300-1500 像素）
    - 填充率：连通域面积/外接矩形面积 > 0.5（排除稀疏噪声）
    - 排除靠近画面边缘的（可能是场外干扰）

    返回 [(frame_idx, frame, [(x1,y1,x2,y2), ...]), ...]
    """
    labeled = []
    total_balls = 0

    for i, (fidx, frame) in enumerate(frames):
        h_img, w_img = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _ORANGE_LOW, _ORANGE_HIGH)
        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 找连通域
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        balls = []
        for k in range(1, num):
            area = stats[k, cv2.CC_STAT_AREA]
            if not (min_area <= area <= max_area):
                continue
            x = stats[k, cv2.CC_STAT_LEFT]
            y = stats[k, cv2.CC_STAT_TOP]
            w = stats[k, cv2.CC_STAT_WIDTH]
            h = stats[k, cv2.CC_STAT_HEIGHT]
            # 圆度约束：宽高比 0.8-1.25（接近圆形，排除长条形腿/手臂）
            if w <= 0 or h <= 0:
                continue
            aspect = w / h
            if not (0.8 <= aspect <= 1.25):
                continue
            # 填充率：连通域面积/外接矩形面积 > 0.5
            fill_ratio = area / (w * h)
            if fill_ratio < 0.5:
                continue
            # 排除靠近画面边缘的（避免场外干扰）
            margin = 20
            if x < margin or y < margin or x + w > w_img - margin or y + h > h_img - margin:
                continue
            balls.append((float(x), float(y), float(x + w), float(y + h)))

        if balls:
            total_balls += len(balls)
        labeled.append((fidx, frame, balls))

        if (i + 1) % 50 == 0:
            print(f"  [颜色] 标注 {i+1}/{len(frames)} 帧, 累计 {total_balls} 个球")

    print(f"颜色标注完成: {total_balls} 个球 / {len(frames)} 帧")
    return labeled


def save_dataset(labeled, out_dir="train_data"):
    """保存为 YOLO 格式数据集。

    目录结构:
      train_data/
        images/train/
        images/val/
        labels/train/
        labels/val/
        data.yaml
    """
    out = ROOT / out_dir
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    # 只保存有标注的帧
    has_label = [(fidx, frame, balls) for fidx, frame, balls in labeled if balls]
    if not has_label:
        print("警告: 没有任何标注！无法训练。")
        return None

    # 80% 训练, 20% 验证
    np.random.seed(42)
    indices = np.random.permutation(len(has_label))
    n_train = int(len(has_label) * 0.8)
    train_idx = set(indices[:n_train].tolist())

    h, w = has_label[0][1].shape[:2]
    for i, (fidx, frame, balls) in enumerate(has_label):
        split = "train" if i in train_idx else "val"
        img_name = f"frame_{fidx:06d}.jpg"
        lbl_name = f"frame_{fidx:06d}.txt"

        # 保存图片
        cv2.imwrite(str(out / "images" / split / img_name), frame)

        # 保存标签（YOLO 格式：class x_center y_center width height，归一化）
        with open(out / "labels" / split / lbl_name, "w") as f:
            for x1, y1, x2, y2 in balls:
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    # data.yaml
    yaml_content = (
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: basketball\n"
    )
    (out / "data.yaml").write_text(yaml_content, encoding="utf-8")

    print(f"数据集已保存到 {out}")
    print(f"  训练集: {n_train} 张, 验证集: {len(has_label) - n_train} 张")
    return out / "data.yaml"


def train(data_yaml, epochs=100, imgsz=1280, model_name="yolov8s.pt",
          batch=4, patience=20, name="ball_manual"):
    """训练 YOLOv8。imgsz=1280 让小目标篮球可见。

    小数据集策略：yolov8s（比 n 大，参数约 3 倍）+ 强数据增强 + 早停。
    GTX 1650 4G 显存：yolov8s + imgsz=1280 + batch=4 约 3G 显存，可跑。
    """
    from ultralytics import YOLO
    print(f"\n开始训练: model={model_name}, epochs={epochs}, imgsz={imgsz}, "
          f"batch={batch}, patience={patience}")
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device="cuda:0",
        batch=batch,
        workers=0,  # Windows 页面文件太小，禁用多进程
        project=str(ROOT / "runs"),
        name=name,
        exist_ok=True,
        patience=patience,  # 早停：验证指标连续 patience 轮无提升则停止
        # 数据增强（小数据集关键，模拟不同光照/角度/颜色）
        hsv_h=0.02,    # 色调变化（适应不同光照下的橙色）
        hsv_s=0.7,     # 饱和度变化
        hsv_v=0.5,     # 亮度变化
        degrees=10,    # 旋转 ±10°
        scale=0.3,     # 缩放 ±30%
        fliplr=0.5,    # 水平翻转概率
        mosaic=1.0,    # mosaic 拼接增强
        mixup=0.1,     # mixup 混合
        verbose=True,
    )
    # 复制最佳权重
    best = ROOT / "runs" / name / "weights" / "best.pt"
    out = ROOT / "weights" / "basketball_custom.pt"
    if best.exists():
        import shutil
        shutil.copy(str(best), str(out))
        print(f"\n训练完成! 权重已保存到: {out}")
        return out
    else:
        print("训练完成但未找到 best.pt")
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser(description="篮球检测模型训练")
    ap.add_argument("--video", default=None, help="输入视频路径（自动标注模式需要）")
    ap.add_argument("--frames", type=int, default=300, help="抽取帧数")
    ap.add_argument("--epochs", type=int, default=100, help="训练轮数")
    ap.add_argument("--model", default="yolov8s.pt", help="基础模型 yolov8n/s/m")
    ap.add_argument("--batch", type=int, default=4, help="batch size")
    ap.add_argument("--patience", type=int, default=20, help="早停耐心值")
    ap.add_argument("--method", choices=["yolo", "color", "both"], default="both",
                    help="标注方法（仅自动模式）: yolo=YOLO检测, color=纯颜色, both=两者结合")
    ap.add_argument("--manual", action="store_true",
                    help="直接训练 train_data_manual/data.yaml（跳过抽帧和自动标注）")
    args = ap.parse_args()

    t0 = time.time()

    # ===== 手动标注模式：直接训练用户标注的数据集 =====
    if args.manual:
        data_yaml = ROOT / "train_data_manual" / "data.yaml"
        if not data_yaml.exists():
            print(f"找不到 {data_yaml}")
            print("请先用 label_tool.py 标注数据：")
            print(f"  E:\\bball-env\\python.exe label_tool.py --video \"xxx.mp4\" --frames 200 --prelabel")
            return
        # 统计标注数量
        n_train = len(list((ROOT / "train_data_manual" / "images" / "train").glob("*.jpg")))
        n_val = len(list((ROOT / "train_data_manual" / "images" / "val").glob("*.jpg")))
        print(f"=== 手动标注模式 ===")
        print(f"数据集: {data_yaml}")
        print(f"训练集 {n_train} 张, 验证集 {n_val} 张")
        if n_train + n_val < 50:
            print(f"\n⚠ 警告: 仅 {n_train + n_val} 帧标注，建议至少 200 帧再训练")
            print(f"  继续训练（数据少时强增强 + 早停会缓解过拟合）...")
        weights = train(data_yaml, epochs=args.epochs, model_name=args.model,
                        batch=args.batch, patience=args.patience, name="ball_manual")
        elapsed = time.time() - t0
        print(f"\n训练完成! 耗时 {elapsed:.0f}s")
        if weights:
            print(f"权重: {weights}")
            print(f"重启 app.py 即可使用新模型")
        return

    # ===== 自动标注模式 =====
    if not args.video:
        print("自动模式需要 --video 参数，或用 --manual 训练手动标注数据")
        ap.print_help()
        return

    # 1. 抽帧
    print("\n=== 1. 抽取视频帧 ===")
    frames = extract_frames(args.video, args.frames)

    # 2. 自动标注
    print("\n=== 2. 自动标注 ===")
    if args.method in ("yolo", "both"):
        labeled = auto_label(frames)
        # 如果 YOLO 标注的球太少，用颜色补充
        ball_count = sum(len(b) for _, _, b in labeled)
        if ball_count < 20 and args.method == "both":
            print(f"YOLO 只检测到 {ball_count} 个球，用颜色检测补充...")
            color_labeled = color_label_fallback(frames)
            # 合并：YOLO 优先，颜色补充空帧
            for i in range(len(labeled)):
                if not labeled[i][2] and color_labeled[i][2]:
                    labeled[i] = (color_labeled[i][0], color_labeled[i][1], color_labeled[i][2])
    elif args.method == "color":
        labeled = color_label_fallback(frames)

    # 3. 保存数据集
    print("\n=== 3. 保存数据集 ===")
    data_yaml = save_dataset(labeled)
    if data_yaml is None:
        print("无法训练：没有标注数据")
        return

    # 4. 训练
    print("\n=== 4. 训练模型 ===")
    weights = train(data_yaml, epochs=args.epochs, model_name=args.model,
                    batch=args.batch, patience=args.patience, name="basketball_train")

    elapsed = time.time() - t0
    print(f"\n全部完成! 耗时 {elapsed:.0f}s")
    if weights:
        print(f"权重: {weights}")
        print(f"请重启 app.py 使用新模型检测")


if __name__ == "__main__":
    main()
