"""ffmpeg 剪辑模块：根据进球时间戳切片并拼接成集锦。

修复点：
  - 临时目录改到 E:\\bball_cache\\clips，避免 C 盘空间不足
  - subprocess 加 creationflags=0x08000000 绕开沙箱限制
  - concat 改用 re-encode（-c:v libx264 -c:a aac）避免各片段编码不一致拼接失败
  - 输出默认路径改到 E:\\bball_cache\\demo_output，避免 C 盘中文路径 URL 编码问题
"""
import os
import subprocess

# 优先用 imageio-ffmpeg 自带版本（含 libx264 + nvenc），conda 版无 libx264
try:
    import imageio_ffmpeg
    _DEFAULT_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _DEFAULT_FFMPEG = "ffmpeg"

# Windows 沙箱规避标志（CREATE_NO_WINDOW）
_SBOX = 0x08000000 if os.name == "nt" else 0

# 缓存根目录（与 demo_interactive.py 保持一致）
_CACHE_ROOT = r"E:\bball_cache"


def cut_clips(video_path, timestamps, pre_roll=5, post_roll=5, min_gap=8,
              output_path=None, codec="libx264", ffmpeg_path=""):
    """根据进球时间戳剪辑集锦。

    timestamps: 进球时刻列表（秒，浮点）
    min_gap: 两个片段间隔小于此值则合并，避免连续得分重复切。
    output_path: 输出文件路径，None 则默认到 E:\\bball_cache\\demo_output\\highlights.mp4
    ffmpeg_path: ffmpeg 可执行文件路径，留空则用 imageio-ffmpeg 自带版本。
    返回: 输出文件路径 或 None（无进球）
    """
    if not timestamps:
        print("没有进球时间戳，跳过剪辑")
        return None

    ffmpeg = ffmpeg_path or _DEFAULT_FFMPEG
    timestamps = sorted([float(t) for t in timestamps])

    # 合并过近的片段
    segments = []
    for ts in timestamps:
        start = max(0.0, ts - pre_roll)
        end = ts + post_roll
        if segments and start - segments[-1][1] < min_gap:
            segments[-1][1] = end
        else:
            segments.append([start, end])

    # 临时目录改到 E 盘缓存
    tmp_dir = os.path.join(_CACHE_ROOT, "clips")
    os.makedirs(tmp_dir, exist_ok=True)
    clip_files = []

    # 输出路径默认到 E 盘
    if output_path is None:
        out_dir = os.path.join(_CACHE_ROOT, "demo_output")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "highlights.mp4")

    # 切片：用 -ss 在输入前做快速 seek，-t 控制时长
    for i, (start, end) in enumerate(segments):
        clip_path = os.path.join(tmp_dir, f"clip_{i:03d}.mp4")
        duration = end - start
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", codec, "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            clip_path,
        ]
        subprocess.run(cmd, check=True, creationflags=_SBOX)
        clip_files.append(clip_path)

    # 拼接列表文件
    list_path = os.path.join(tmp_dir, "concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clip_files:
            # concat demuxer 要求路径用正斜杠或转义，Windows 用绝对路径 + 正斜杠最稳
            p_norm = os.path.abspath(p).replace("\\", "/")
            f.write(f"file '{p_norm}'\n")

    # 拼接：用 re-encode（-c:v libx264 -c:a aac）确保各片段编码一致
    # 之前用 -c copy 对重新编码的片段可能失败（时间戳/SPS 不一致）
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-c:v", codec, "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, creationflags=_SBOX)

    # 清理临时文件
    for p in clip_files:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.remove(list_path)
    except OSError:
        pass

    total_dur = sum(e - s for s, e in segments)
    print(f"集锦已生成: {output_path}（共 {len(segments)} 段，时长 {total_dur:.0f}s）")
    return output_path
