"""诊断脚本：在视频里找出球离篮筐最近的帧，导出图片供人工核对篮筐标定。

用法:
    E:\\bball-env\\python.exe diag_hoop.py
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

_CACHE_ROOT = r"E:\bball_cache"
os.environ["GRADIO_TEMP_DIR"] = os.path.join(_CACHE_ROOT, "gradio")

import cv2
import numpy as np
from video_io import VideoReader, get_video_info
from app import get_ball_model

VIDEO = r"E:\bball_cache\gradio\326de539ca77a062c476a6b8def084f37d4e0e3124d9127c53f05486a34919f8\2026.07.05 2nd.mp4"
HOOP = (313, 130, 366, 236)  # 你刚才标定的篮筐框
BALL_CONF = 0.3
SAMPLE_EVERY = 3  # 每 3 帧采样 1 帧（加速）


def main():
    info = get_video_info(VIDEO)
    print(f"视频: {info['total']} 帧 | {info['fps']:.1f} fps | {info['codec']}")
    print(f"篮筐框: {HOOP}")
    print("加载 YOLO...")
    model, wp = get_ball_model()
    print(f"权重: {wp}")

    out_dir = Path(_CACHE_ROOT) / "diag_hoop"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有检测到球且离篮筐中心较近的帧
    hx = (HOOP[0] + HOOP[2]) / 2
    hy = (HOOP[1] + HOOP[3]) / 2
    candidates = []  # (dist, fidx, cx, cy, conf)

    reader = VideoReader(VIDEO)
    n = 0
    try:
        for fidx, frame in reader.iter_frames(start=0, end=info["total"], batch=SAMPLE_EVERY):
            n += 1
            try:
                res = model.predict(frame, conf=BALL_CONF, imgsz=1280,
                                    device="cuda:0", verbose=False)[0]
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                clses = res.boxes.cls.cpu().numpy().astype(int)
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                best = None
                for j, c in enumerate(clses):
                    nm = res.names.get(c, "").lower()
                    if "ball" in nm or "basketball" in nm:
                        if best is None or confs[j] > confs[best]:
                            best = j
                if best is None:
                    continue
                x1, y1, x2, y2 = xyxy[best]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                d = ((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5
                candidates.append((d, fidx, cx, cy, confs[best]))
            except Exception:
                continue
            if n % 100 == 0:
                print(f"  采样 {n} 帧 | 候选 {len(candidates)}")
    finally:
        reader.close()

    print(f"\n采样完成: {n} 帧 | 检测到球 {len(candidates)} 帧")

    if not candidates:
        print("❌ 没检测到任何球")
        return

    # 按距离排序，取最近的 12 帧导出
    candidates.sort(key=lambda x: x[0])
    top = candidates[:12]
    print(f"\n球离篮筐最近的 12 帧（导出图片）:")
    for i, (d, fidx, cx, cy, conf) in enumerate(top):
        print(f"  [{i+1}] 帧 {fidx} | 球({cx:.0f},{cy:.0f}) 置信度{conf:.2f} | 离篮筐中心 {d:.0f}px")

    # 重新读取这些帧并画图
    for i, (d, fidx, cx, cy, conf) in enumerate(top):
        from video_io import read_frame
        frame = read_frame(VIDEO, fidx, total=info["total"], fps=info["fps"])
        if frame is None:
            continue
        out = frame.copy()
        # 画篮筐框（绿）+ 上沿/下沿线
        x1, y1, x2, y2 = HOOP
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.line(out, (x1 - 40, y1), (x2 + 40, y1), (0, 255, 255), 2)  # 上沿 黄
        cv2.line(out, (x1 - 40, y2), (x2 + 40, y2), (255, 0, 255), 2)  # 下沿 紫
        cv2.putText(out, "HOOP", (x1, max(y1 - 12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(out, f"top y={y1}", (x2 + 10, y1 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(out, f"bot y={y2}", (x2 + 10, y2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        # 画球（红）
        cv2.circle(out, (int(cx), int(cy)), 8, (0, 0, 255), -1)
        cv2.putText(out, f"BALL conf={conf:.2f}", (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        ts = fidx / info["fps"]
        cv2.putText(out, f"Frame {fidx} ({ts:.1f}s) dist={d:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        path = str(out_dir / f"diag_{i+1:02d}_frame{fidx}.jpg")
        cv2.imwrite(path, out)
        print(f"  已保存: {path}")

    # 额外：把球 y 坐标分布画成统计图
    all_ys = [c[3] for c in candidates]
    all_xs = [c[2] for c in candidates]
    stat = np.zeros((info["height"], info["width"], 3), dtype=np.uint8) + 30
    # 画篮筐
    cv2.rectangle(stat, (HOOP[0], HOOP[1]), (HOOP[2], HOOP[3]), (0, 255, 0), 2)
    cv2.line(stat, (HOOP[0] - 40, HOOP[1]), (HOOP[2] + 40, HOOP[1]), (0, 255, 255), 1)
    cv2.line(stat, (HOOP[0] - 40, HOOP[3]), (HOOP[2] + 40, HOOP[3]), (255, 0, 255), 1)
    # 画所有球位置（红点）
    for x, y in zip(all_xs, all_ys):
        cv2.circle(stat, (int(x), int(y)), 2, (0, 0, 255), -1)
    cv2.putText(stat, f"ALL BALL POSITIONS ({len(all_ys)} pts) + HOOP BOX",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(stat, f"Hoop: x[{HOOP[0]},{HOOP[2]}] y[{HOOP[1]},{HOOP[3]}]",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    stat_path = str(out_dir / "00_ball_positions_map.jpg")
    cv2.imwrite(stat_path, stat)
    print(f"\n球位置分布图: {stat_path}")
    print(f"\n所有图片在: {out_dir}")
    print("请查看这些图片，确认篮筐框是否画在真实篮筐位置。")


if __name__ == "__main__":
    main()
