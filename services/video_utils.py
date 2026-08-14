"""视频工具函数：帧编码、文件扫描。"""
import os
import re
import base64

import cv2

from . import state


def frame_to_base64(frame):
    """将 cv2 帧转为 base64 PNG data URI。

    输入约定为 RGB（与 load_video / preview_frame / click_calibrate
    等返回值一致），而 cv2.imencode 按 BGR 处理，
    因此先转回 BGR 再编码，避免预览画面红蓝通道互换。
    """
    if frame is None:
        return None
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.png', frame)
    b64 = base64.b64encode(buf).decode('utf-8')
    return f'data:image/png;base64,{b64}'


def scan_video_files(folder):
    """扫描文件夹内的视频文件，按自然顺序排序。"""
    folder = folder.strip().strip('"').strip("'")
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in state.VIDEO_EXTS:
            files.append(os.path.join(folder, name))

    def _natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    files.sort(key=_natural_key)
    return files
