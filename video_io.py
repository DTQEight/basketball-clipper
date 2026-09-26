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


def _enable_frame_threading(container):
    """开启帧级多线程解码。

    PyAV 默认 thread_type=SLICE（帧内切片并行），1080p 下可用并行度有限；
    改成 FRAME（帧间流水线）后解码可摊到多核：实测 130 -> 285 帧/秒。
    两种模式解出的帧在帧号与像素上逐帧一致，含 seek 路径（详见
    cache/_thread_check.py 的对照测试）。代价是每线程缓冲一帧，
    1080p yuv420p 每帧约 3.1MB，8 线程多占约 25MB。

    失败时保持 PyAV 默认，不影响可用性。
    """
    try:
        container.streams.video[0].codec_context.thread_type = "FRAME"
    except Exception as e:
        _log.warning(f"[WARN] 开启 FRAME 线程解码失败，回退默认: {e}")
    return container


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
            return _enable_frame_threading(av.open(path, format=fmt))
        except av.error.InvalidDataError:
            pass
    return _enable_frame_threading(av.open(path))


def _stream_start_pts(stream, fps: float) -> int:
    """视频流起始 pts（帧号换算基准）。

    .ts/.flv 等容器的首帧 pts 常从非零开始（录制起点偏移），
    不减去起始偏移会导致所有帧号/进球时间戳整体平移。

    忽略阈值必须以**帧**为尺度（旧实现按 0.5 秒，会漏修半秒内的
    真实偏移）：0.4s 的偏移在 30fps 下就是 12 帧——不修则 seek
    整体错位、开头 12 帧读不到（截尾）且帧号平移。只有不足半帧的
    微头部偏移（如 mp4 的半个 pts tick）才忽略，避免换算取整抖动
    把目标帧算偏 1 帧。fps 参与换算：同样的 0.02s 在 15fps 视频
    上不足半帧、在 60fps 视频上已超过 1 帧。
    """
    try:
        if stream.start_time is not None and stream.time_base:
            tb = float(stream.time_base)
            if tb > 0:
                start_sec = float(stream.start_time) * tb
                # 起始 pts 为负/0 不需要修正；换算成帧后 <0.5 帧的偏移忽略
                if start_sec > 0 and start_sec * max(float(fps) or 1.0, 1.0) >= 0.5:
                    # round 而非 int：float 除法可能把整数 pts 算成 x.9999，截断丢 1
                    return int(round(start_sec / tb))
    except Exception:
        pass
    return 0


def get_video_info(path: str) -> dict:
    """获取视频基本信息。

    返回 dict: {total, fps, width, height, codec, duration}
    """
    c = av_open(path)
    try:
        # 显式检查无视频流（纯音频/损坏文件）：直接 streams.video[0] 抛 IndexError，
        # 调用方只能看到裸下标错误；给出明确错误信息便于区分"文件坏了"和"代码 bug"
        if not c.streams.video:
            raise ValueError(f"文件没有视频流: {path}")
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
    finally:
        # 无视频流（streams.video[0] IndexError）时也要释放容器，
        # Windows 下泄漏句柄会锁定文件妨碍后续删除/覆盖
        c.close()
    return {
        "total": max(total, 1),
        "fps": fps,
        "width": width,
        "height": height,
        "codec": codec,
        "duration": duration,
    }


# PyAV 的 color_trc 取值（AVColorTransferCharacteristic）里的两种 HDR 传递特性
#   16 = SMPTE 2084（PQ）  ·  18 = ARIB STD B67（HLG）
_HDR_COLOR_TRC = (16, 18)


def is_hdr_source(path: str) -> bool:
    """源视频是不是 HDR（HLG / PQ）。

    只给「产出一个给人看的文件」的路径用（预览片段 / 集锦导出）——这类文件要能在
    浏览器里正常显示，得把 HDR 先压到 BT.709 SDR。**检测与特征抽取一律照旧直接
    读源文件**，不做任何转换（口径一动，历史分数/缓存就全失去可比性）。

    老录像都是 8-bit BT.709（color_trc=1）→ False。读不到/未知/异常一律当 SDR：
    误判成 False 只是画面偏淡（HLG 向后兼容，能看），误判成 True 会把 SDR 当 HDR
    压暗，所以异常侧保守。
    """
    try:
        c = av_open(path)
    except Exception:
        return False
    try:
        if not c.streams.video:
            return False
        trc = c.streams.video[0].codec_context.color_trc
        return trc is not None and int(trc) in _HDR_COLOR_TRC
    except Exception:
        return False
    finally:
        c.close()


def read_frame(path: str, idx: int, total: int = 0, fps: float = 30.0):
    """读取指定帧（按帧号 seek），返回 BGR ndarray 或 None。

    path:  视频文件路径
    idx:   帧号（0-based）
    total: 总帧数（用于 clamp，0/None 均视为不 clamp）
    fps:   帧率（用于 pts 转换，0/None 则自动读取）
    """
    # None 防御：显式传入 None 时不抛 TypeError，与 0 同语义（不 clamp）
    if total and total > 0:
        idx = min(int(idx), total - 1)
    else:
        idx = int(idx)
    if idx < 0:
        return None

    container = None
    try:
        container = av_open(path)
        stream = container.streams.video[0]
        # 关闭帧级多线程解码：这是一个必现死锁的规避（2026.09.22 定位）。
        # 现象：同一进程里先跑过 torch/CUDA 的 YOLO 推理后，再解某些 .mov
        # （实测 E:\ball\*.mov，HEVC/QuickTime 1920x1080），会永久卡在 avcodec
        # 帧线程池的条件变量上（原生栈 SleepConditionVariableSRW；服务端表现为
        # 「加载视频」永远转圈、TCP 可连但 HTTP 不响应、CPU 0%）。
        # 实测（同序列各重复 4 次）：默认帧线程 4/4 卡死；thread_count=1 → 0/4。
        # 单帧解码本身是毫秒级，关掉线程无可见代价；批量解码走 VideoReader
        # （检测主路径，285 帧/秒），不在此处，不受影响。
        stream.codec_context.thread_count = 1
        if not fps or fps <= 0:
            fps = float(stream.average_rate) if stream.average_rate else 30.0
        tb = float(stream.time_base) if stream.time_base else 1.0 / fps
        start_pts = _stream_start_pts(stream, fps)
        target_pts = start_pts + int(idx / fps / tb) if tb > 0 else 0
        # 提前约 2 帧 seek：关键帧对齐可能使 seek 落在目标帧之后，
        # 提前解码再跳到目标帧，保证返回的是目标帧而非更晚的帧
        pre_pts = int(2.0 / fps / tb) if tb > 0 else 0
        container.seek(max(0, target_pts - pre_pts), stream=stream)
        # demux + codec_context.decode 替代 container.decode：逐 packet try/except
        # 跳过损坏的 NAL（兼容开头损坏但后续数据完好的视频，如断电录制）。
        # 正常视频无损坏 packet，try/except 不触发，零开销。
        cc = stream.codec_context
        for p in container.demux(stream):
            try:
                for f in cc.decode(p):
                    cur_pts = f.pts if f.pts is not None else 0
                    cur_idx = int(round((cur_pts - start_pts) * tb * fps))
                    if cur_idx >= idx:
                        return f.to_ndarray(format="bgr24")
            except Exception:
                continue
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

    def __init__(self, path, single_thread=False):
        """single_thread=True 关闭帧级多线程解码（thread_count=1）。

        与 read_frame 同一规避：进程内跑过 torch/CUDA 推理后再解码，avcodec 帧
        线程池会偶发永久卡在条件变量上（原生栈 SleepConditionVariableSRW，
        表现为 CPU 0% 假死）。检测主路径（解码与 YOLO 交替但码率/节奏稳定）
        实测未复现，保持多线程以吃到 285 帧/秒；**复核路径**已复现
        （goal_verifier A 臂 extract()：批 3/11 卡死），故按调用方显式开启。
        """
        self.container = av_open(path)
        try:
            self.stream = self.container.streams.video[0]
            if single_thread:
                # 必须在 seek/decode 之前设置，线程池建好后再改无效
                self.stream.codec_context.thread_count = 1
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
            # 起始 pts 偏移（.ts/.flv 非零起始），帧号换算需减去
            self.start_pts = _stream_start_pts(self.stream, self.fps)
            # 容错解码跳过的损坏 packet/帧计数：为 0 表示零丢帧；
            # >0 表示视频存在坏 NAL（decode 层已跳过）。调用方可在
            # 迭代结束后读取——processed==0 且此处>0 即可判定"全程解码失败"，
            # 而不是把空结果当"正常检测完成"（B2）
            self.decode_errors = 0
        except Exception:
            # 构造中途失败（无视频流等）也要释放容器，避免句柄泄漏锁文件
            self.close()
            raise

    def seek(self, frame_idx):
        """seek 到指定帧号（提前约 2 帧，确保 decode 覆盖目标帧）。"""
        pts = self.start_pts + int(frame_idx / self.fps / self.tb) if self.tb > 0 else 0
        pre = int(2.0 / self.fps / self.tb) if self.tb > 0 else 0
        self.container.seek(max(0, pts - pre), stream=self.stream)

    def iter_frames(self, start=0, end=None, batch=1):
        """迭代帧序列。

        start: 起始帧号
        end:   结束帧号（不包含），None 则到视频末尾
        batch: 抽帧间隔，每 batch 帧取 1 帧（<1 视为 1，防取模除零）
        yield: (frame_idx, frame_bgr)

        容错范围：decode 逐包与 demux 容器层都做异常隔离——损坏 NAL 跳过
        继续，截断/读取到 EOF 边界（容器层错误）则停止迭代并计入
        decode_errors，不再把异常穿给调用方导致整次检测崩溃
        （断电录制/拷贝中断的 mp4 尾部常缺数据）。
        """
        batch = max(int(batch), 1)
        if end is None:
            end = self.total
        self.seek(start)
        # demux + codec_context.decode 替代 container.decode：逐 packet try/except
        # 跳过损坏的 NAL（与 read_frame 同一容错策略，保证检测流程也能处理损坏视频）
        cc = self.stream.codec_context
        try:
            for p in self.container.demux(self.stream):
                try:
                    for f in cc.decode(p):
                        cur_pts = f.pts if f.pts is not None else 0
                        fidx = int(round((cur_pts - self.start_pts) * self.tb * self.fps))
                        if fidx < start:
                            continue
                        if fidx >= end:
                            return
                        if (fidx - start) % batch != 0:
                            continue
                        yield fidx, f.to_ndarray(format="bgr24")
                except Exception:
                    # 跳过损坏 NAL（兼容开头损坏但后续数据完好的视频）。
                    # 计数暴露给调用方：全程 0 帧 + 全量错误 → 判定解码失败
                    self.decode_errors += 1
                    continue
        except Exception:
            # 容器层 demux 错误：多为截断文件读越过真实数据末尾
            # （如 "partial file"）。没有更多可解数据，计数后停止迭代——
            # 已解出的帧仍然有效，由调用方按 decode_errors>0 决定如何告警。
            # 旧实现此处异常直接穿出迭代器，打崩 run_detect 主循环
            # （断电录制/拷贝中断的 mp4 尾部常缺数据）。
            self.decode_errors += 1
            return

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

    def __del__(self):
        # 兜底释放：调用方漏掉 with/close（尤其异常路径）时避免容器句柄泄漏
        # （Windows 下会锁定文件，妨碍后续删除/覆盖）。
        # 解释器退出阶段异常一律吞掉，不能因清理失败抛 __del__ 警告。
        try:
            self.close()
        except Exception:
            pass
