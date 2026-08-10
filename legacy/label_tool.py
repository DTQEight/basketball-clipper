"""篮球标注工具：快速手动标注篮球，生成 YOLO 格式数据集。

操作：
  1. 自动抽帧（跳过已标注帧，支持时间范围）
  2. 显示每帧，用鼠标拖拽框选篮球
  3. 快捷键：
     n 下一帧 / p 上一帧 / s 跳过 / u 撤销最后一个框
     a 预标注（用现有模型预测球框，自动填入）
     d 删除当前帧所有标注
     w 保存进度（不退出，可继续标注）
     q 保存并退出 / ESC 直接退出（不保存当前帧）

用法:
    # 追加标注 200 帧（自动跳过已标注的 37 帧）
    E:\bball-env\python.exe label_tool.py --video "E:\bball_tmp\bball_preview.mp4" --frames 200

    # 从第 1000 秒到第 2000 秒抽帧标注
    E:\bball-env\python.exe label_tool.py --video "xxx.mp4" --frames 100 --start 1000 --end 2000

    # 用预标注辅助（现有模型预测，用户只需修正）
    E:\bball-env\python.exe label_tool.py --video "xxx.mp4" --frames 100 --prelabel
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
from video_io import get_video_info

_CACHE_ROOT = r"E:\bball_cache"
OUT_DIR = ROOT / "train_data_manual"


def load_existing_labels(out_dir):
    """读取已标注的 fidx 集合 和 {fidx: [(x1,y1,x2,y2),...]} 标签（像素坐标）。

    用于：1) 抽帧时跳过已标注帧  2) 标注时预加载已有标注  3) 导出时合并
    """
    labeled_fidx = set()
    labels = {}  # fidx -> [(x1,y1,x2,y2), ...] 像素坐标
    for split in ("train", "val"):
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*.jpg"):
            # 文件名 frame_XXXXXX.jpg -> fidx
            try:
                fidx = int(img_path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            labeled_fidx.add(fidx)
            # 读取对应标签
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            boxes = []
            if lbl_path.exists():
                # 需要图像尺寸来反归一化
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    for line in lbl_path.read_text().strip().splitlines():
                        parts = line.split()
                        if len(parts) >= 5:
                            xc, yc, bw, bh = map(float, parts[1:5])
                            x1 = (xc - bw / 2) * w
                            y1 = (yc - bh / 2) * h
                            x2 = (xc + bw / 2) * w
                            y2 = (yc + bh / 2) * h
                            boxes.append((x1, y1, x2, y2))
            if boxes:
                labels[fidx] = boxes
    return labeled_fidx, labels


class Labeler:
    def __init__(self, frames, out_dir, existing_labels=None, prelabel_model=None):
        """frames: [(fidx, frame), ...]
        existing_labels: {fidx: [(x1,y1,x2,y2),...]} 已有标注，用于预加载
        prelabel_model: YOLO 模型路径或 None，用于 'a' 键预标注
        """
        self.frames = frames
        self.out_dir = out_dir
        self.idx = 0
        self.drawing = False
        self.start = None
        self.end = None
        self.current_boxes = []
        # 合并已有标注（预加载到 labels 字典）
        self.labels = dict(existing_labels) if existing_labels else {}
        # 当前帧初始化：如果已有标注，加载到 current_boxes
        if frames:
            fidx0 = frames[0][0]
            self.current_boxes = list(self.labels.get(fidx0, []))
        self.prelabel_model = prelabel_model
        self._yo = None  # 懒加载 YOLO

        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            (out_dir / sub).mkdir(parents=True, exist_ok=True)

    def _get_yolo(self):
        if self._yo is None and self.prelabel_model:
            from ultralytics import YOLO
            self._yo = YOLO(self.prelabel_model)
        return self._yo

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start and self.end:
                x1 = min(self.start[0], self.end[0])
                y1 = min(self.start[1], self.end[1])
                x2 = max(self.start[0], self.end[0])
                y2 = max(self.start[1], self.end[1])
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    self.current_boxes.append((x1, y1, x2, y2))
            self.start = None
            self.end = None

    def draw_frame(self):
        fidx, frame = self.frames[self.idx]
        img = frame.copy()
        # 已有框（红）
        for (x1, y1, x2, y2) in self.current_boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        # 正在拖拽（绿）
        if self.drawing and self.start and self.end:
            cv2.rectangle(img, self.start, self.end, (0, 255, 0), 2)
        # 信息栏
        labeled_count = sum(1 for f, _ in self.frames if f in self.labels and self.labels[f])
        info = (f"Frame {self.idx+1}/{len(self.frames)} (fidx={fidx}) | "
                f"balls: {len(self.current_boxes)} | 已标注 {labeled_count}/{len(self.frames)} | "
                f"[n]next [p]prev [s]skip [u]undo [a]prelabel [d]del [w]save [q]quit")
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        return img

    def save_current(self):
        fidx, _ = self.frames[self.idx]
        if self.current_boxes:
            self.labels[fidx] = [tuple(b) for b in self.current_boxes]
        elif fidx in self.labels:
            # 当前帧清空了框，从 labels 删除
            del self.labels[fidx]

    def prelabel_current(self):
        """用现有模型预测当前帧的球框，自动填入。"""
        yolo = self._get_yolo()
        if yolo is None:
            return
        fidx, frame = self.frames[self.idx]
        try:
            res = yolo.predict(frame, conf=0.25, imgsz=1280, device="cuda:0", verbose=False)[0]
        except Exception as e:
            print(f"预标注失败: {e}")
            return
        if res.boxes is None or len(res.boxes) == 0:
            print(f"帧 {fidx}: 模型未检测到目标")
            return
        names = res.names
        clses = res.boxes.cls.cpu().numpy().astype(int)
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        added = 0
        for i, c in enumerate(clses):
            n = names.get(c, "").lower()
            if "ball" in n or "basketball" in n:
                x1, y1, x2, y2 = xyxy[i]
                self.current_boxes.append((float(x1), float(y1), float(x2), float(y2)))
                added += 1
        print(f"帧 {fidx}: 预标注添加 {added} 个框 (conf>0.25)")

    def run(self):
        cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Labeler", 1280, 720)
        cv2.setMouseCallback("Labeler", self.mouse_callback)

        while True:
            img = self.draw_frame()
            cv2.imshow("Labeler", img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('n'):  # next
                self.save_current()
                nxt = min(self.idx + 1, len(self.frames) - 1)
                self.current_boxes = list(self.labels.get(self.frames[nxt][0], []))
                self.idx = nxt
            elif key == ord('p'):  # prev
                self.save_current()
                self.idx = max(self.idx - 1, 0)
                self.current_boxes = list(self.labels.get(self.frames[self.idx][0], []))
            elif key == ord('s'):  # skip (不保存当前帧)
                self.current_boxes = []
                self.idx = min(self.idx + 1, len(self.frames) - 1)
            elif key == ord('u'):  # undo last box
                if self.current_boxes:
                    self.current_boxes.pop()
            elif key == ord('a'):  # 预标注辅助
                self.prelabel_current()
            elif key == ord('d'):  # 删除当前帧所有标注
                self.current_boxes = []
                fidx = self.frames[self.idx][0]
                if fidx in self.labels:
                    del self.labels[fidx]
                print(f"帧 {fidx}: 已删除所有标注")
            elif key == ord('w'):  # 保存进度（不退出）
                self.save_current()
                self.export()
                print("进度已保存，可继续标注")
            elif key == ord('q'):  # save & quit
                self.save_current()
                break
            elif key == 27:  # ESC 直接退出不保存当前帧
                break

        cv2.destroyAllWindows()
        self.export()

    def export(self):
        """导出为 YOLO 格式数据集（合并已有标注）。"""
        if not self.labels:
            print("没有标注任何帧！")
            return

        # 收集所有标注帧（来自 self.frames 中已标注的）
        labeled = []
        frame_map = {fidx: frame for fidx, frame in self.frames}
        for fidx, boxes in self.labels.items():
            if fidx in frame_map:
                labeled.append((fidx, frame_map[fidx], boxes))

        if not labeled:
            print("没有可导出的标注！")
            return

        # 80% train, 20% val（固定种子保证可复现，且尽量保持已有划分稳定）
        np.random.seed(42)
        indices = np.random.permutation(len(labeled))
        n_train = int(len(labeled) * 0.8)
        train_idx = set(indices[:n_train].tolist())

        h, w = labeled[0][1].shape[:2]
        for i, (fidx, frame, boxes) in enumerate(labeled):
            split = "train" if i in train_idx else "val"
            img_name = f"frame_{fidx:06d}.jpg"
            lbl_name = f"frame_{fidx:06d}.txt"
            cv2.imwrite(str(self.out_dir / "images" / split / img_name), frame)
            with open(self.out_dir / "labels" / split / lbl_name, "w") as f:
                for x1, y1, x2, y2 in boxes:
                    xc = (x1 + x2) / 2 / w
                    yc = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        yaml_content = (
            f"path: {self.out_dir.resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n  0: basketball\n"
        )
        (self.out_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

        print(f"导出完成! 共 {len(labeled)} 帧, 训练集 {n_train}, 验证集 {len(labeled)-n_train}")
        print(f"数据集: {self.out_dir}")
        print(f"data.yaml: {self.out_dir / 'data.yaml'}")


def extract_frames(video_path, num_frames=80, start_sec=None, end_sec=None,
                   skip_existing=None):
    """从视频均匀抽取 N 帧。

    start_sec/end_sec: 时间范围（秒），None 表示从开头/结尾
    skip_existing: set of fidx，跳过这些帧（已标注）
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

    # 时间范围转帧范围
    start_fidx = int(start_sec * fps) if start_sec is not None else 0
    end_fidx = int(end_sec * fps) if end_sec is not None else total
    start_fidx = max(0, min(start_fidx, total - 1))
    end_fidx = max(start_fidx + 1, min(end_fidx, total))
    print(f"抽帧范围: 帧 {start_fidx}-{end_fidx} ({start_fidx/fps:.1f}s-{end_fidx/fps:.1f}s)")

    if skip_existing:
        print(f"跳过已标注的 {len(skip_existing)} 帧")

    # 均匀抽帧
    indices = np.linspace(start_fidx, end_fidx - 1, num_frames, dtype=int)
    indices = sorted(set(indices.tolist()))
    # 过滤已标注的
    if skip_existing:
        before = len(indices)
        indices = [i for i in indices if i not in skip_existing]
        print(f"过滤已标注: {before} -> {len(indices)} 帧（跳过 {before - len(indices)}）")

    if not indices:
        print("没有新帧可抽取（全部已标注或范围太小）！")
        return []

    print(f"开始抽取 {len(indices)} 帧...")
    tmp_dir = os.path.join(_CACHE_ROOT, "label_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    frames = []
    for i, fidx in enumerate(indices):
        timestamp = fidx / fps
        out_path = os.path.join(tmp_dir, f"frame_{fidx:06d}.jpg")
        cmd = [ffmpeg, "-y", "-loglevel", "error",
               "-ss", f"{timestamp:.3f}", "-i", video_path,
               "-frames:v", "1", "-q:v", "2", out_path]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30,
                           creationflags=0x08000000)
            if os.path.exists(out_path):
                frame = cv2.imread(out_path)
                if frame is not None:
                    frames.append((fidx, frame))
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"  已抽 {i+1}/{len(indices)} 帧")

    print(f"抽取完成: {len(frames)} 帧")
    return frames


def main():
    import argparse
    ap = argparse.ArgumentParser(description="篮球标注工具（支持追加标注）")
    ap.add_argument("--video", required=True, help="输入视频路径")
    ap.add_argument("--frames", type=int, default=80, help="抽取帧数")
    ap.add_argument("--start", type=float, default=None, help="起始时间（秒）")
    ap.add_argument("--end", type=float, default=None, help="结束时间（秒）")
    ap.add_argument("--prelabel", action="store_true",
                    help="启用预标注辅助（用现有模型预测，按 a 应用）")
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="不跳过已标注帧（默认跳过）")
    args = ap.parse_args()

    # 加载已有标注
    skip_existing = None
    existing_labels = None
    if not args.no_skip_existing:
        labeled_fidx, existing_labels = load_existing_labels(OUT_DIR)
        skip_existing = labeled_fidx
        print(f"已有标注: {len(labeled_fidx)} 帧")

    frames = extract_frames(args.video, args.frames,
                            start_sec=args.start, end_sec=args.end,
                            skip_existing=skip_existing)
    if not frames:
        print("没有新帧可标注！可尝试增大 --frames 或用 --no-skip-existing")
        return

    # 预标注模型
    prelabel_model = None
    if args.prelabel:
        wpath = ROOT / "weights" / "basketball_custom.pt"
        if wpath.exists():
            prelabel_model = str(wpath)
            print(f"预标注模型: {wpath}")
        else:
            print("警告: 未找到 weights/basketball_custom.pt，预标注禁用")

    labeler = Labeler(frames, OUT_DIR, existing_labels=existing_labels,
                      prelabel_model=prelabel_model)
    labeler.run()


if __name__ == "__main__":
    main()
