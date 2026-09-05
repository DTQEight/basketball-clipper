"""视频工具函数：帧编码、文件扫描。"""
import os
import re
import base64

import cv2

from . import state


def frame_to_base64(frame) -> "str | None":
    """将 cv2 帧转为 base64 JPEG data URI（质量 85）。

    输入约定为 RGB（与 load_video / preview_frame / click_calibrate
    等返回值一致），而 cv2.imencode 按 BGR 处理，
    因此先转回 BGR 再编码，避免预览画面红蓝通道互换。
    JPEG q85 替代 PNG 无损：标定预览不需要无损，1080p 下
    体积/编码耗时降一个数量级（~5MB/数百 ms → ~300KB/几十 ms），
    不影响 image_x/y 坐标换算（坐标基于元素尺寸，与压缩无关）。
    """
    if frame is None:
        return None
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', frame,
                          [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64 = base64.b64encode(buf).decode('utf-8')
    return f'data:image/jpeg;base64,{b64}'


def scan_video_files(folder) -> list:
    """扫描文件夹内的视频文件，按自然顺序排序。"""
    if not isinstance(folder, str):
        return []
    folder = folder.strip().strip('"').strip("'")
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        # isfile 过滤：扩展名形如 *.mp4 的子目录不应被当作视频返回
        if ext in state.VIDEO_EXTS:
            fpath = os.path.join(folder, name)
            if os.path.isfile(fpath):
                files.append(fpath)

    def _natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    files.sort(key=_natural_key)
    return files
