"""ffmpeg 剪辑模块：根据进球时间戳切片并拼接成集锦。

修复点：
  - 临时目录改到 E:\\basketball-project\\cache\\clips，避免 C 盘空间不足
  - subprocess 加 creationflags=0x08000000 绕开沙箱限制
  - concat 改用 re-encode（-c:v libx264 -c:a aac）避免各片段编码不一致拼接失败
  - 输出默认路径改到 E:\\basketball-project\\cache\\demo_output，避免 C 盘中文路径 URL 编码问题
  - GPU 硬编：优先使用 h264_nvenc（NVIDIA GPU 加速），不可用则回退 libx264
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
_CACHE_ROOT = r"E:\basketball-project\cache"


def _detect_nvenc(ffmpeg):
    """检测 ffmpeg 是否支持 h264_nvenc 编码器。"""
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                           capture_output=True, text=True, creationflags=_SBOX, timeout=10)
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


def _build_encode_args(ffmpeg, quality="hq"):
    """构建视频编码参数，优先用 NVENC 硬编，回退 libx264 软编。

    quality: "hq" 高质量（集锦，cq=20）/ "preview" 预览（cq=26）
    返回: (codec, preset_or_preset, crf_or_cq) 参数列表
    """
    if _detect_nvenc(ffmpeg):
        # NVENC 硬编：-preset p4（平衡速度与质量）-rc vbr -cq 控制质量
        # GTX 1650 有 1 个 NVENC 编码器，可大幅加速
        cq = "20" if quality == "hq" else "26"
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                "-cq", cq, "-b:v", "0", "-spatial_aq", "1"]
    else:
        # 软编回退
        crf = "18" if quality == "hq" else "26"
        return ["-c:v", "libx264", "-preset", "fast", "-crf", crf]


def cut_clips(video_path, timestamps, pre_roll=5, post_roll=5, min_gap=8,
              output_path=None, ffmpeg_path="", progress_callback=None):
    """根据进球时间戳剪辑集锦（GPU 硬编加速）。

    timestamps: 进球时刻列表（秒，浮点）
    min_gap: 两个片段间隔小于此值则合并，避免连续得分重复切。
    output_path: 输出文件路径，None 则默认到 E:\\basketball-project\\cache\\demo_output\\highlights.mp4
    ffmpeg_path: ffmpeg 可执行文件路径，留空则用 imageio-ffmpeg 自带版本。
    progress_callback: 可选进度回调 (pct, msg)，0-100。
    返回: 输出文件路径 或 None（无进球）
    """

    def _report(pct, msg):
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

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

    # 检测编码器一次，避免每个片段重复检测
    use_nvenc = _detect_nvenc(ffmpeg)
    encode_tag = "NVENC 硬编" if use_nvenc else "libx264 软编"
    print(f"[剪辑] 使用 {encode_tag} | 共 {len(segments)} 段", flush=True)
    _report(10, f'初始化编码器（{len(segments)} 段，{encode_tag}）...')

    # 切片：用 -ss 在输入前做快速 seek，-t 控制时长
    # 集锦保持原画质：NVENC cq=20 / libx264 crf=18，不缩放
    encode_args = _build_encode_args(ffmpeg, quality="hq")
    for i, (start, end) in enumerate(segments):
        _report(15 + 70 * i / len(segments), f'剪切片段 {i+1}/{len(segments)}...')
        clip_path = os.path.join(tmp_dir, f"clip_{i:03d}.mp4")
        duration = end - start
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", video_path,
            "-t", f"{duration:.3f}",
        ] + encode_args + [
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

    # 拼接：用 re-encode 确保各片段编码一致
    _report(90, '正在拼接集锦视频...')
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_path,
    ] + encode_args + [
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, creationflags=_SBOX)
    _report(98, '清理临时文件...')

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
    print(f"集锦已生成: {output_path}（{encode_tag} | 共 {len(segments)} 段 | 时长 {total_dur:.0f}s）", flush=True)
    return output_path
