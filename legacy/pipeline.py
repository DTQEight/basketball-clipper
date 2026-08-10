"""主流程：视频 → YOLOv8 检测 → 篮筐标定 → 进球判断 → ffmpeg 剪辑。

用法:
    python pipeline.py --video path/to/game.mp4
    python pipeline.py --video game.mp4 --config config.yaml

注意：本模块用 PyAV（而非 OpenCV）读取视频，兼容 HEVC 编码。
"""
import argparse
import sys
from pathlib import Path

import yaml

# 本脚本位于 legacy/ 下：项目根（video_io/cutter）与 legacy（detector/logic）都要入路径
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from video_io import get_video_info, VideoReader
from detector.yolo_detector import YoloDetector
from detector.hoop_calibrator import HoopCalibrator
from logic.shot_judge import ShotJudge
from cutter.ffmpeg_cutter import cut_clips


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_class_indices(class_names):
    """根据 config 中的类别名称，确定球和篮筐的类别索引。"""
    ball_cls, hoop_cls = 0, 1  # 默认
    for i, name in enumerate(class_names):
        n = name.lower()
        if n in ("basketball", "ball"):
            ball_cls = i
        elif n in ("hoop", "rim", "basket"):
            hoop_cls = i
    return ball_cls, hoop_cls


def run(video_path, config_path="config.yaml"):
    cfg = load_config(config_path)
    m, h, j, c = cfg["model"], cfg["hoop"], cfg["judge"], cfg["cutter"]

    detector = YoloDetector(m["weights"], m["device"], m["conf_threshold"])
    calibrator = HoopCalibrator(h["manual_box"], h["calibrate_frames"])
    ball_cls, hoop_cls = resolve_class_indices(m["classes"])

    # 用 PyAV 获取视频信息（兼容 HEVC）
    info = get_video_info(video_path)
    fps = info["fps"] or 30.0
    print(f"视频: {video_path}")
    print(f"  {fps:.2f} fps | {info['width']}x{info['height']} | {info['codec']} | {info['total']} 帧")

    judge = None
    score_timestamps = []
    checked = 0

    # 用 VideoReader 顺序解码（比 OpenCV 稳定，支持 HEVC）
    with VideoReader(video_path) as reader:
        for frame_idx, frame in reader.iter_frames(batch=j["frame_batch"]):
            checked += 1
            dets = detector.detect(frame)

            # 取置信度最高的球
            ball_box = None
            hoop_boxes = []
            for cls, x1, y1, x2, y2, conf in dets:
                if cls == ball_cls:
                    if ball_box is None or conf > ball_box[4]:
                        ball_box = [x1, y1, x2, y2, conf]
                elif cls == hoop_cls:
                    hoop_boxes.append([x1, y1, x2, y2, conf])

            # 篮筐标定（固定机位，标定完成后锁定）
            hoop = calibrator.update(hoop_boxes)
            if calibrator.ready and judge is None:
                judge = ShotJudge(hoop, j["ball_motion_threshold"], j["hoop_above_margin"])
                print(f"篮筐标定完成: {[round(v, 1) for v in hoop]}")

            # 进球判断
            if judge is not None:
                ball_b = ball_box[:4] if ball_box else None
                event = judge.feed(ball_b)
                if event == "SCORE":
                    ts = frame_idx / fps
                    score_timestamps.append(ts)
                    print(f"[进球] 第 {len(score_timestamps)} 个 @ {ts:.2f}s")

    print(f"\n检测完成: 共 {len(score_timestamps)} 个进球"
          f"（judge 统计 score={judge.score_count if judge else 0}, "
          f"miss={judge.miss_count if judge else 0}）")

    if score_timestamps:
        cut_clips(
            video_path, score_timestamps,
            pre_roll=c["pre_roll"], post_roll=c["post_roll"],
            min_gap=c["min_gap"], output_path=c["output"],
            codec=c["codec"], ffmpeg_path=c.get("ffmpeg_path", ""),
        )
    else:
        print("未检测到进球，不生成集锦。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="篮球录像进球自动剪辑")
    ap.add_argument("--video", required=True, help="输入视频路径")
    ap.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = ap.parse_args()
    run(args.video, args.config)
