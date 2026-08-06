"""ffmpeg 剪辑模块：根据进球时间戳切片并拼接成集锦。"""
import os
import subprocess

# 优先用 imageio-ffmpeg 自带版本（含 libx264 + nvenc），conda 版无 libx264
try:
    import imageio_ffmpeg
    _DEFAULT_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _DEFAULT_FFMPEG = "ffmpeg"


def cut_clips(video_path, timestamps, pre_roll=5, post_roll=5, min_gap=8,
              output_path="highlights.mp4", codec="libx264", ffmpeg_path=""):
    """根据进球时间戳剪辑集锦。

    timestamps: 进球时刻列表（秒，浮点）
    min_gap: 两个片段间隔小于此值则合并，避免连续得分重复切。
    ffmpeg_path: ffmpeg 可执行文件路径，留空则用 PATH 中的 ffmpeg。
    """
    if not timestamps:
        print("没有进球时间戳，跳过剪辑")
        return None

    ffmpeg = ffmpeg_path or _DEFAULT_FFMPEG
    timestamps = sorted(timestamps)

    # 合并过近的片段
    segments = []
    for ts in timestamps:
        start = max(0.0, ts - pre_roll)
        end = ts + post_roll
        if segments and start - segments[-1][1] < min_gap:
            segments[-1][1] = end
        else:
            segments.append([start, end])

    tmp_dir = "tmp_clips"
    os.makedirs(tmp_dir, exist_ok=True)
    clip_files = []

    # 切片：用 -ss 在输入前做快速 seek，-t 控制时长
    for i, (start, end) in enumerate(segments):
        clip_path = os.path.join(tmp_dir, f"clip_{i:03d}.mp4")
        duration = end - start
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", codec, "-c:a", "aac",
            clip_path,
        ]
        subprocess.run(cmd, check=True)
        clip_files.append(clip_path)

    # 拼接
    list_path = os.path.join(tmp_dir, "concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clip_files:
            f.write(f"file '{os.path.abspath(p)}'\n")

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-c", "copy", output_path,
    ]
    subprocess.run(cmd, check=True)

    # 清理临时文件
    for p in clip_files:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.remove(list_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"集锦已生成: {output_path}（共 {len(segments)} 段）")
    return output_path
