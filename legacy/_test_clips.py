"""模拟 UI 检测流程，验证每球独立片段生成是否正常。"""
import sys
import os
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# 缓存目录
_CACHE_ROOT = r"E:\basketball-project\cache"
os.environ["GRADIO_TEMP_DIR"] = os.path.join(_CACHE_ROOT, "gradio")

import cv2
from video_io import get_video_info, read_frame, VideoReader
from app import get_ball_model
from tracker import GoalDetector
from demo_interactive import draw_frame_diff

# 测试视频
VIDEO = r"D:\Downloads\2026.07.054th.mp4"
# 篮筐框（从之前的检测结果复制）
HOOP = (309, 154, 363, 234)
# 检测范围（前 60 秒，和 UI 测试一致）
START_FRAME = 0
END_FRAME = 1800  # 60s @ 30fps

print(f"=== 测试视频: {VIDEO} ===")
print(f"篮筐框: {HOOP}")
print(f"检测范围: {START_FRAME}-{END_FRAME} 帧")

# 1. 读取视频信息
info = get_video_info(VIDEO)
print(f"视频信息: {info['total']} 帧 | {info['fps']:.1f} fps | {info['width']}x{info['height']} | {info['codec']}")

fps = info["fps"]

# 2. 读取基准帧（用第 0 帧作为无球基准帧）
print("\n读取基准帧...")
baseline_frame = read_frame(VIDEO, 0, total=info["total"], fps=fps)
print(f"基准帧形状: {baseline_frame.shape}")

# 3. 初始化检测器（loose_mode=True 宽松模式）
detector = GoalDetector(HOOP, baseline_frame=baseline_frame,
                        min_gap_sec=5.0,
                        fusion_mode="visual_only",
                        loose_mode=True)
print(f"检测器初始化完成 (loose_mode={detector.loose_mode})")

# 4. 加载 YOLO
print("\n加载 YOLO 模型...")
model, weights_path = get_ball_model()
print(f"YOLO 权重: {weights_path}")

# 5. 检测（第一阶段：纯检测，不写视频）
print(f"\n=== 第一阶段：检测 {END_FRAME - START_FRAME} 帧 ===")
t0 = time.time()
processed = 0

reader = VideoReader(VIDEO)
try:
    for fidx, frame in reader.iter_frames(start=START_FRAME, end=END_FRAME, batch=1):
        # 喂入检测器（diff 模式不需要 YOLO 球位置）
        detector.feed(None, fidx, fps, frame=frame)
        processed += 1
        if processed % 300 == 0:
            print(f"  进度: {processed}/{END_FRAME - START_FRAME} | 进球: {len(detector.goals)}")
finally:
    reader.close()

detector.finalize()
elapsed = time.time() - t0
print(f"\n第一阶段完成: {processed} 帧 | 耗时: {elapsed:.1f}s | 检测到进球: {len(detector.goals)} 个")
print(f"进球时间: {detector.goals}")

# 6. 第二阶段：为每个进球生成独立片段
print(f"\n=== 第二阶段：生成 {len(detector.goals)} 个独立片段 ===")
out_dir = Path(_CACHE_ROOT) / "demo_output"
out_dir.mkdir(parents=True, exist_ok=True)
_stamp = int(time.time())
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
clip_half = int(fps * 3)  # 进球前后各 3 秒

import imageio_ffmpeg
import subprocess as _sp
ff = imageio_ffmpeg.get_ffmpeg_exe()

goal_clips = []
goals = sorted(detector.goals)
for gi, gts in enumerate(goals):
    gframe = int(gts * fps)
    seg_start = max(START_FRAME, gframe - clip_half)
    seg_end = min(END_FRAME, gframe + clip_half)
    print(f"\n  [{gi+1}/{len(goals)}] 进球 @ {gts:.1f}s (帧 {gframe})")
    print(f"      片段范围: 帧 {seg_start}-{seg_end} ({seg_end - seg_start + 1} 帧)")

    # 写 mp4v 中间文件
    clip_mp4v = str(out_dir / f"goal_{gi}_{int(gts)}s_{_stamp}.mp4")
    writer = cv2.VideoWriter(clip_mp4v, fourcc, fps,
                             (info["width"], info["height"]))
    written = 0
    try:
        for fidx in range(seg_start, seg_end + 1):
            frame = read_frame(VIDEO, fidx, total=info["total"], fps=fps)
            if frame is None:
                continue
            blob = detector.last_blob_box
            annotated = draw_frame_diff(frame, blob, HOOP, detector, fidx, fps)
            writer.write(annotated)
            written += 1
    finally:
        writer.release()
    print(f"      写入 {written} 帧 → {clip_mp4v}")

    # 转码为 H.264
    clip_h264 = str(out_dir / f"goal_{gi}_{int(gts)}s_{_stamp}_h264.mp4")
    try:
        r = _sp.run([ff, "-y", "-i", clip_mp4v, "-c:v", "libx264",
                     "-crf", "23", "-movflags", "+faststart", clip_h264],
                    creationflags=0x08000000, capture_output=True, timeout=60)
        if os.path.exists(clip_h264) and os.path.getsize(clip_h264) > 0:
            os.remove(clip_mp4v)
            final_path = clip_h264
            print(f"      转码成功 → {clip_h264} ({os.path.getsize(clip_h264)/1024:.0f} KB)")
        else:
            final_path = clip_mp4v
            print(f"      ⚠️ 转码失败，用 mp4v: {r.stderr.decode('utf-8', errors='ignore')[:200]}")
    except Exception as e:
        final_path = clip_mp4v
        print(f"      ⚠️ 转码异常: {e}")

    goal_clips.append({"ts": gts, "path": final_path, "idx": gi})

# 7. 验证片段文件
print(f"\n=== 验证片段文件 ===")
for i, clip in enumerate(goal_clips):
    p = clip["path"]
    exists = os.path.exists(p)
    size = os.path.getsize(p) / 1024 if exists else 0
    print(f"  [{i+1}] {clip['ts']:.1f}s | {'✅' if exists else '❌'} | {size:.0f} KB | {os.path.basename(p)}")

print(f"\n=== 测试完成 ===")
print(f"共生成 {len(goal_clips)} 个片段")
print(f"总大小: {sum(os.path.getsize(c['path'])/1024/1024 for c in goal_clips if os.path.exists(c['path'])):.1f} MB")
