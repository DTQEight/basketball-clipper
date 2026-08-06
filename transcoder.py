"""视频转码模块：将 HEVC 等浏览器不支持的编码转为 H.264。

用 imageio-ffmpeg 自带的 FFmpeg（含 libx264 + nvenc），
支持 GPU 硬解硬编（hevc_cuvid → h264_nvenc），失败回退 CPU 软编。

关键修复：
  1. 用参数列表（subprocess.run）而非字符串拼接，避免文件名空格导致解析错误
  2. 转码输出到 E 盘（C 盘空间不足）
  3. creationflags=CREATE_NO_WINDOW 避免 Windows 弹窗
"""
import os
import subprocess
import tempfile

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"

# 转码输出目录（优先 E 盘，C 盘空间不足）
_CACHE_ROOT = r"E:\bball_cache"
_PREVIEW_DIR = os.path.join(_CACHE_ROOT, "bball_tmp")


def _ensure_preview_dir():
    os.makedirs(_PREVIEW_DIR, exist_ok=True)
    return _PREVIEW_DIR


def probe_codec(video_path, ffprobe_path=None):
    """用 ffprobe 探测视频编码。

    ffprobe_path: 指定 ffprobe 路径，None 则用 PATH 中的。
    返回编码名（小写），失败返回 'unknown'。
    """
    ffprobe = ffprobe_path or "ffprobe"
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, check=True, timeout=30,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
        return r.stdout.strip().lower()
    except Exception:
        return "unknown"


def transcode_to_h264(video_path, ffprobe_path=None, use_gpu=True):
    """将视频转码为 H.264（浏览器可播放）。

    返回 (output_path_or_None, status_message)。
    - 若原编码已是 H.264，返回原路径
    - 转码失败返回 (None, error_message)
    """
    codec = probe_codec(video_path, ffprobe_path)

    # H.264 系列浏览器可直接播放
    if codec in ("h264", "avc", "avc1", ""):
        return video_path, f"编码 {codec or 'h264'}，浏览器可直接播放"

    out_dir = _ensure_preview_dir()
    out_path = os.path.join(out_dir, "bball_preview.mp4")
    # 清理旧文件
    try:
        os.remove(out_path)
    except OSError:
        pass

    # 关键：用参数列表传给 subprocess，文件名带空格也能正确处理
    def _run(cmd, timeout):
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )

    # GPU 路径：hevc_cuvid 硬解 + h264_nvenc 硬编
    if use_gpu:
        cmd_gpu = [
            FFMPEG, "-y", "-loglevel", "error",
            "-hwaccel", "cuda", "-c:v", "hevc_cuvid",
            "-i", video_path,
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "aac",
            out_path,
        ]
        try:
            r = _run(cmd_gpu, 1800)
            if r.returncode == 0 and os.path.exists(out_path):
                return out_path, f"原编码 {codec}，已转 H.264（GPU 加速）"
        except subprocess.TimeoutExpired:
            pass

    # CPU 软编回退
    cmd_cpu = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", video_path,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac",
        out_path,
    ]
    try:
        r2 = _run(cmd_cpu, 7200)
        if r2.returncode == 0 and os.path.exists(out_path):
            return out_path, f"原编码 {codec}，已转 H.264（CPU 软编）"
        err = (r2.stderr or "")[:200]
        return None, f"转码失败（{codec}）: {err}"
    except subprocess.TimeoutExpired:
        return None, f"转码超时（{codec}），CPU 软编耗时过长"
    except Exception as e:
        return None, f"转码异常: {str(e)[:100]}"
