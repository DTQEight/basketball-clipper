"""统一视频读取模块（基于 PyAV，兼容 HEVC/H.264/moov 后置 mp4）。

OpenCV 5.0 自带 FFmpeg 后端不完整，无法解码 HEVC，
且对 moov atom 在文件末尾的 mp4 会失败。
本模块用 PyAV 统一解决这些问题。
"""
import logging
import os
import av
import numpy as np

_log = logging.getLogger("video_io")


def av_open(path: str):
    """打开视频容器，兼容 moov atom 在末尾的 mp4（非 faststart）。

    PyAV 默认探测对 moov 后置的大文件会报 InvalidData，指定 format 即可。
    """
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    fmt_map = {
        "mp4": "mp4", "mov": "mov", "m4v": "mp4",
        "mkv": "matroska", "webm": "webm", "avi": "avi",
        "ts": "mpegts", "flv": "flv",
    }
    fmt = fmt_map.get(ext)
    if fmt:
        try:
            return av.open(path, format=fmt)
        except av.error.InvalidDataError:
            pass
    return av.open(path)


def get_video_info(path: str) -> dict:
    """获取视频基本信息。

    返回 dict: {total, fps, width, height, codec, duration}
    """
    c = av_open(path)
    s = c.streams.video[0]
    # s.frames 可能为 None/0（部分视频头信息不全），or 0 兜底；
    # 不能写 int(s.frames)，None 会抛 TypeError
    total = int(s.frames or 0)
    fps = float(s.average_rate) if s.average_rate else 30.0
    width = s.width or 0
    height = s.height or 0
    codec = s.codec_context.name or "unknown"
    duration = 0.0
    if s.duration and s.time_base:
        duration = float(s.duration) * float(s.time_base)
    # 某些视频 frames 为 0，按 duration*rate 估算
    if total == 0 and duration > 0 and fps > 0:
        total = int(duration * fps)
    c.close()
    return {
        "total": max(total, 1),
        "fps": fps,
        "width": width,
        "height": height,
        "codec": codec,
        "duration": duration,
    }


def read_frame(path: str, idx: int, total: int = 0, fps: float = 30.0):
    """读取指定帧（按帧号 seek），返回 BGR ndarray 或 None。

    path:  视频文件路径
    idx:   帧号（0-based）
    total: 总帧数（用于 clamp）
    fps:   帧率（用于 pts 转换，0 则自动读取）
    """
    if total > 0:
        idx = min(int(idx), total - 1)
    else:
        idx = int(idx)
    if idx < 0:
        return None

    container = None
    try:
        container = av_open(path)
        stream = container.streams.video[0]
        if fps <= 0:
            fps = float(stream.average_rate) if stream.average_rate else 30.0
        tb = float(stream.time_base) if stream.time_base else 1.0 / fps
        target_pts = int(idx / fps / tb) if tb > 0 else 0
        # 提前约 2 帧 seek：关键帧对齐可能使 seek 落在目标帧之后，
        # 提前解码再跳到目标帧，保证返回的是目标帧而非更晚的帧
        pre_pts = int(2.0 / fps / tb) if tb > 0 else 0
        container.seek(max(0, target_pts - pre_pts), stream=stream)
        for f in container.decode(stream):
            cur_pts = f.pts if f.pts is not None else 0
            cur_idx = int(round(cur_pts * tb * fps))
            if cur_idx >= idx:
                return f.to_ndarray(format="bgr24")
        return None
    except Exception as e:
        # 旧实现静默返回 None，解码问题（容器损坏/编码不支持）无从排查
        _log.warning(f"[WARN] read_frame 失败: {path} @帧 {idx}: {e}")
        return None
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass


class VideoReader:
    """顺序解码读取器（用于批量处理，避免反复 open/seek）。

    用法:
        reader = VideoReader(path)
        for idx, frame in reader.iter_frames(start=1000, end=5000, batch=3):
            # 处理 frame
        reader.close()
    """

    def __init__(self, path):
        self.container = av_open(path)
        self.stream = self.container.streams.video[0]
        # 直接复用已打开的容器读取信息，避免再 open 一次（get_video_info 会重开）
        s = self.stream
        self.total = int(s.frames or 0) or 0
        self.fps = float(s.average_rate) if s.average_rate else 30.0
        if self.total == 0 and s.duration and s.time_base:
            duration = float(s.duration) * float(s.time_base)
            if duration > 0:
                self.total = int(duration * self.fps)
        self.total = max(self.total, 1)
        self.tb = float(self.stream.time_base) if self.stream.time_base else 1.0 / self.fps

    def seek(self, frame_idx):
        """seek 到指定帧号（提前约 2 帧，确保 decode 覆盖目标帧）。"""
        pts = int(frame_idx / self.fps / self.tb) if self.tb > 0 else 0
        pre = int(2.0 / self.fps / self.tb) if self.tb > 0 else 0
        self.container.seek(max(0, pts - pre), stream=self.stream)

    def iter_frames(self, start=0, end=None, batch=1):
        """迭代帧序列。

        start: 起始帧号
        end:   结束帧号（不包含），None 则到视频末尾
        batch: 抽帧间隔，每 batch 帧取 1 帧
        yield: (frame_idx, frame_bgr)
        """
        if end is None:
            end = self.total
        self.seek(start)
        for f in self.container.decode(self.stream):
            cur_pts = f.pts if f.pts is not None else 0
            fidx = int(round(cur_pts * self.tb * self.fps))
            if fidx < start:
                continue
            if fidx >= end:
                break
            if (fidx - start) % batch != 0:
                continue
            yield fidx, f.to_ndarray(format="bgr24")

    def close(self):
        if self.container:
            try:
                self.container.close()
            except Exception:
                pass
            self.container = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
