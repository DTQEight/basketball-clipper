"""测试 Mimo v2.5 视频理解能力。"""
import os
import base64
import subprocess
import imageio_ffmpeg
from openai import OpenAI
from pathlib import Path

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SBOX = 0x08000000

# 1. 截取 5 秒片段
src = r"D:\Downloads\highlights.mp4"
clip = r"E:\basketball-project\cache\vlm_test.mp4"
subprocess.run([
    FFMPEG, "-y", "-loglevel", "error",
    "-ss", "10", "-i", src, "-t", "5",
    "-c:v", "libx264", "-preset", "fast", "-crf", "28",
    "-vf", "scale='min(640,iw)':'min(640,ih)':force_original_aspect_ratio=decrease,fps=2",
    "-an", "-movflags", "+faststart", clip,
], creationflags=SBOX)
print(f"片段大小: {Path(clip).stat().st_size / 1024:.0f} KB")

# 2. base64 编码
with open(clip, "rb") as f:
    data = f.read()
data_url = f"data:video/mp4;base64,{base64.b64encode(data).decode()}"
print(f"base64 长度: {len(data_url) / 1024:.0f} KB")

# 3. 调用 Mimo v2.5 视频理解
client = OpenAI(
    api_key="tp-c4ljxg7lgen91djwqnbl3l0sa1pju6ptdl0umriv54naf6s6",
    base_url="https://token-plan-cn.xiaomimimo.com/v1",
)
print("调用 Mimo v2.5 视频理解...")
try:
    resp = client.chat.completions.create(
        model="mimo-v2.5",
        messages=[{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": data_url},
                 "fps": 2, "media_resolution": "default"},
                {"type": "text", "text": "请描述这个视频片段中的内容，是否有篮球进球？"},
            ],
        }],
        max_tokens=512,
        extra_body={"thinking": {"type": "disabled"}},
    )
    print("=== VLM 回复 ===")
    print(resp.choices[0].message.content)
except Exception as e:
    print(f"调用失败: {e}")
