"""调试 VLM 全否决问题：直接打印 VLM 对候选片段的完整回复。

用法:
    E:\\bball-env\\python.exe _debug_vlm.py <视频路径> <候选时间戳1> [候选时间戳2 ...]
    E:\\bball-env\\python.exe _debug_vlm.py D:\\Downloads\\highlights.mp4 15.0 30.0 45.0
"""
import os
import sys
import json
import base64
import subprocess
from pathlib import Path
from openai import OpenAI

# 复用 vlm_verifier 的工具（从项目根目录向上查找）
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from vlm_verifier import (
    _extract_clip, _encode_video_base64, _FFMPEG, _SBOX,
    _CACHE_ROOT, MIMO_BASE_URL, MIMO_MODEL, VERIFY_PROMPT,
)

# 优先从 .env 读 API key
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

API_KEY = os.environ.get("MIMO_API_KEY", "")

# 调试用：放宽版 prompt，让 VLM 更详细描述看到了什么
DEBUG_PROMPT = (
    "这是一段篮球视频片段。请详细描述你看到的画面内容，重点关注：\n"
    "1. 场景（室内/室外、机位角度）\n"
    "2. 球员动作（谁在投篮、防守、抢篮板）\n"
    "3. 篮球的位置和运动轨迹（能否看清球？球在飞吗？球是否接近篮筐？）\n"
    "4. 是否有进球事件（球是否进入篮筐？如果没看到进球，说明为什么——是没拍到、被遮挡、还是球没进？）\n\n"
    "请如实描述，不要臆测。最后用 JSON 返回判断：\n"
    '{"is_goal": true或false, "confidence": 0.0到1.0, "reason": "详细说明", '
    '"ball_visible": true或false, "hoop_visible": true或false}'
)


def debug_one(video_path, timestamp, window_sec=4.5, use_strict_prompt=False):
    """对单个时间戳做调试，返回完整 VLM 回复。"""
    print(f"\n{'='*60}")
    print(f"调试时刻: {timestamp}s (窗口 ±{window_sec}s = {window_sec*2}s 片段)")
    print(f"{'='*60}")

    tmp_dir = Path(_CACHE_ROOT) / "vlm_debug"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    clip_path = tmp_dir / f"dbg_{int(timestamp*1000)}.mp4"

    start = max(0, timestamp - window_sec)
    duration = window_sec * 2

    print(f"截取片段: {start:.1f}s - {start+duration:.1f}s ...")
    if not _extract_clip(video_path, start, duration, clip_path):
        print("❌ 截取失败")
        return

    size_kb = clip_path.stat().st_size / 1024
    print(f"片段大小: {size_kb:.0f} KB")

    # 顺便把片段复制到桌面方便人工查看
    import shutil
    desktop_clip = Path(r"C:\Users\desktop\Desktop") / f"vlm_dbg_{int(timestamp)}s.mp4"
    try:
        shutil.copy2(clip_path, desktop_clip)
        print(f"片段已复制到桌面: {desktop_clip.name}")
    except Exception:
        pass

    print(f"base64 编码...")
    data_url = _encode_video_base64(clip_path)

    prompt = VERIFY_PROMPT if use_strict_prompt else DEBUG_PROMPT
    prompt_name = "严格版" if use_strict_prompt else "调试版（详细描述）"
    print(f"调用 Mimo v2.5 (prompt: {prompt_name})...")

    import time
    t0 = time.time()
    client = OpenAI(api_key=API_KEY, base_url=MIMO_BASE_URL)
    try:
        resp = client.chat.completions.create(
            model=MIMO_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": data_url},
                     "fps": 3, "media_resolution": "default"},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=800,
            timeout=60,
            extra_body={"thinking": {"type": "disabled"}},
        )
        elapsed = time.time() - t0
        content = resp.choices[0].message.content
        usage = resp.usage

        print(f"\n⏱ 耗时: {elapsed:.1f}s")
        if usage:
            print(f"📊 tokens: prompt={usage.prompt_tokens} "
                  f"completion={usage.completion_tokens} "
                  f"total={usage.total_tokens}")
        print(f"\n━━━ VLM 完整回复 ━━━")
        print(content)
        print(f"━━━ 回复结束 ━━━")

        # 尝试解析 JSON
        try:
            import re
            m = re.search(r'\{[\s\S]*\}', content)
            if m:
                data = json.loads(m.group())
                print(f"\n解析 JSON: {json.dumps(data, ensure_ascii=False, indent=2)}")
        except Exception as e:
            print(f"\n⚠️ JSON 解析失败: {e}")

    except Exception as e:
        print(f"❌ API 调用失败: {e}")
    finally:
        clip_path.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python _debug_vlm.py <视频路径> <时间戳1> [时间戳2 ...]")
        print("示例: python _debug_vlm.py D:\\Downloads\\highlights.mp4 15.0 30.0")
        sys.exit(1)

    video = sys.argv[1]
    timestamps = [float(t) for t in sys.argv[2:]]

    if not API_KEY:
        print("❌ 未找到 MIMO_API_KEY，请检查 .env 文件")
        sys.exit(1)

    print(f"视频: {video}")
    print(f"候选时刻: {timestamps}")
    print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")

    for ts in timestamps:
        debug_one(video, ts, window_sec=4.5, use_strict_prompt=True)
