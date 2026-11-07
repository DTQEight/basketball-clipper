"""片段缩略图：切片阶段顺手生成 + 磁盘缓存（UI 与检测共用）。

片段列表首次渲染时逐张解码会把列表卡住数秒（40 个约 15 秒）。这里把
「解码片段中点帧 → 96×54 JPEG → 落盘」收成一个可在检测切片线程里调用的函数，
UI 侧只读缓存；未命中时再退回后台补图（P1-4）。
"""
import base64
import hashlib
import os

from . import state

THUMB_W, THUMB_H = 96, 54   # 行内 46×26 的 2x，兼顾高分屏清晰度与体积
# 1×1 透明 GIF：占位期间不显示浏览器的「图片损坏」图标
BLANK_IMG = ('data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAE'
             'AAAIBRAA7')


def _thumb_dir():
    return os.path.join(state.CACHE_ROOT, "thumbs")


def cache_path(clip_path):
    """按「绝对路径 + mtime + 大小」派生缓存文件名（片段重切后自动失效）。"""
    try:
        st = os.stat(clip_path)
        key = f'{os.path.abspath(clip_path)}|{st.st_mtime_ns}|{st.st_size}'
    except OSError:
        return None
    h = hashlib.md5(key.encode('utf-8')).hexdigest()
    return os.path.join(_thumb_dir(), h + '.jpg')


def uri(clip_path):
    """读缓存并转 data URI；未命中返回 None（不触发解码）。"""
    fp = cache_path(clip_path)
    if not fp or not os.path.exists(fp):
        return None
    try:
        with open(fp, 'rb') as f:
            return 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode()
    except OSError:
        return None


def ensure(clip_path):
    """确保缩略图已落盘（未命中则解码中点帧并写入），返回 data URI 或 None。

    阻塞函数：UI 调用方应放进 io_bound；检测切片线程可直接调用。
    """
    got = uri(clip_path)
    if got:
        return got
    import cv2
    cap = cv2.VideoCapture(clip_path)
    try:
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
        ok, frame = cap.read()
        if not ok:  # 中部不可解时退回首帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        frame = cv2.resize(frame, (THUMB_W, THUMB_H))
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            return None
    finally:
        cap.release()
    fp = cache_path(clip_path)
    if fp:
        try:
            os.makedirs(_thumb_dir(), exist_ok=True)
            tmp = fp + '.tmp'
            with open(tmp, 'wb') as f:
                f.write(buf.tobytes())
            os.replace(tmp, fp)  # 原子落盘：多标签页并发写同一片段也不会读到半截文件
        except OSError:
            pass
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()
