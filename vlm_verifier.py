"""VLM 进球验证模块：用大模型对 CV 检测的候选进球做二次确认。

工作原理（CV + VLM 融合）：
  1. CV（diff 基准帧差法）放宽阈值扫描整段视频，找出所有"疑似进球"候选时段
  2. 对每个候选时刻，截取前后 window_sec 秒的视频片段
  3. 压缩片段后 base64 编码，发送给 Mimo v2.5 VLM API
  4. VLM 判断"球是否从上方穿过篮筐进入网中"
  5. VLM 确认 = 真进球，VLM 否决 = 误报

优势：
  - VLM 只处理少量短片段（5-20 个 × 6 秒），不是整段视频，结果稳定
  - CV 负责高召回（不漏球），VLM 负责高精度（不误报）
  - 规避纯 VLM 的非确定性问题（项目记忆记录的教训）
"""
import os
import base64
import subprocess
import tempfile
from pathlib import Path

# OpenAI 兼容 SDK（Mimo API 用此格式）
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# imageio-ffmpeg 自带 ffmpeg（含 libx264）
try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _FFMPEG = "ffmpeg"

_SBOX = 0x08000000 if os.name == "nt" else 0  # Windows 沙箱规避

MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5"

# 缓存目录
_CACHE_ROOT = r"E:\bball_cache"

# VLM 验证 prompt：只判断是否进球，要求严格
VERIFY_PROMPT = (
    "请仔细观察这段篮球视频片段，判断其中是否有进球事件。\n\n"
    "进球定义：篮球从篮筐上方穿过篮圈进入网中（包括空心球、打板进球、扣篮）。"
    "球只碰到篮筐没进、球弹出去、球从下方飞过都不算进球。\n\n"
    "请严格判断，只在明确看到球进入篮筐时才回答是。如果不确定，回答否。\n\n"
    "请用以下 JSON 格式返回（不要包含其他文字）：\n"
    '{"is_goal": true或false, "confidence": 0.0到1.0, "reason": "简要说明判断依据"}'
)


def _extract_clip(video_path, start_sec, duration_sec, output_path):
    """用 ffmpeg 截取视频片段。

    start_sec: 起始时间（秒）
    duration_sec: 时长（秒）
    output_path: 输出文件路径
    返回: True/False
    """
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-ss", f"{max(0, start_sec):.3f}", "-i", str(video_path),
        "-t", f"{duration_sec:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "28",
        "-vf", "scale='min(720,iw)':'min(720,ih)':force_original_aspect_ratio=decrease,fps=3",
        "-an",  # 丢音频减小体积
        "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60,
                           creationflags=_SBOX)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def _encode_video_base64(path):
    """读取视频文件并编码为 base64 data URL。"""
    with open(path, "rb") as f:
        data = f.read()
    return f"data:video/mp4;base64,{base64.b64encode(data).decode('utf-8')}"


def verify_goal(video_path, timestamp, api_key,
                window_sec=3.0, fps=3, timeout=30):
    """对单个候选时刻进行 VLM 验证。

    video_path: 视频文件路径
    timestamp: 候选进球时间戳（秒）
    api_key: Mimo API key
    window_sec: 截取窗口（前后各 window_sec 秒，总 2*window_sec 秒）
    fps: 发送给 VLM 的帧率（3 fps 足够）
    timeout: API 超时（秒）
    返回: dict {
        "is_goal": bool,        # VLM 判断是否进球
        "confidence": float,    # 置信度 0-1
        "reason": str,          # 判断依据
        "elapsed_s": float,     # API 耗时
        "error": str or None,   # 错误信息
    }
    """
    if OpenAI is None:
        return {"is_goal": False, "confidence": 0, "reason": "",
                "elapsed_s": 0, "error": "openai 库未安装"}

    # 截取候选时段片段
    start = max(0, timestamp - window_sec)
    duration = window_sec * 2

    tmp_dir = Path(_CACHE_ROOT) / "vlm_clips"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    clip_path = tmp_dir / f"clip_{int(timestamp * 1000)}.mp4"

    try:
        if not _extract_clip(video_path, start, duration, clip_path):
            return {"is_goal": False, "confidence": 0, "reason": "",
                    "elapsed_s": 0, "error": "截取视频片段失败"}

        # 检查片段大小（超过 50MB 会撑爆内存）
        clip_size_mb = clip_path.stat().st_size / 1024 / 1024
        if clip_size_mb > 50:
            return {"is_goal": False, "confidence": 0, "reason": "",
                    "elapsed_s": 0, "error": f"片段过大 {clip_size_mb:.0f}MB"}

        # base64 编码
        data_url = _encode_video_base64(clip_path)

        # 调用 Mimo API
        import time
        t0 = time.time()
        client = OpenAI(api_key=api_key, base_url=MIMO_BASE_URL)

        response = client.chat.completions.create(
            model=MIMO_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": data_url},
                     "fps": fps, "media_resolution": "default"},
                    {"type": "text", "text": VERIFY_PROMPT},
                ],
            }],
            max_tokens=512,
            timeout=timeout,
            extra_body={"thinking": {"type": "disabled"}},
        )
        elapsed = time.time() - t0

        content = response.choices[0].message.content.strip()

        # 解析 JSON（容错处理）
        import json
        import re
        result = {"is_goal": False, "confidence": 0.0, "reason": "",
                  "elapsed_s": round(elapsed, 1), "error": None}

        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                result["is_goal"] = bool(data.get("is_goal", False))
                result["confidence"] = float(data.get("confidence", 0))
                result["reason"] = str(data.get("reason", ""))
            else:
                # 没有 JSON，尝试从文本判断
                lower = content.lower()
                if '"is_goal": true' in lower or '"is_goal":true' in lower:
                    result["is_goal"] = True
                result["reason"] = content[:200]
        except (json.JSONDecodeError, ValueError) as e:
            result["error"] = f"JSON 解析失败: {e}"
            result["reason"] = content[:200]

        return result

    except Exception as e:
        return {"is_goal": False, "confidence": 0, "reason": "",
                "elapsed_s": 0, "error": str(e)}
    finally:
        # 清理临时片段
        try:
            clip_path.unlink(missing_ok=True)
        except Exception:
            pass


def verify_goals_batch(video_path, timestamps, api_key,
                       window_sec=3.0, fps=3, progress_callback=None):
    """批量验证多个候选进球。

    timestamps: 候选进球时间戳列表（秒）
    progress_callback: 可选回调 fn(current, total, result)
    返回: list of dict，每个 dict 包含原时间戳和验证结果
        {
            "timestamp": float,      # 原时间戳
            "is_goal": bool,         # VLM 判断
            "confidence": float,
            "reason": str,
            "elapsed_s": float,
            "error": str or None,
        }
    """
    results = []
    total = len(timestamps)
    for i, ts in enumerate(timestamps):
        r = verify_goal(video_path, ts, api_key, window_sec=window_sec, fps=fps)
        r["timestamp"] = ts
        results.append(r)
        if progress_callback:
            progress_callback(i + 1, total, r)
    return results


def filter_confirmed_goals(verify_results, min_confidence=0.5):
    """从验证结果中筛选 VLM 确认的进球。

    min_confidence: 最低置信度阈值
    返回: 确认进球的时间戳列表
    """
    return [r["timestamp"] for r in verify_results
            if r["is_goal"] and r["confidence"] >= min_confidence]
