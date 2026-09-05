"""video_io 契约测试：read_frame 参数边界（P0-1 回归防护）。"""
import pytest
from fractions import Fraction

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")
import numpy as np

from video_io import get_video_info, read_frame, VideoReader, _stream_start_pts


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    """合成 30 帧 30fps mp4（批量标定读取路径的最小复现）。"""
    p = tmp_path_factory.mktemp("vio") / "sample.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    if not w.isOpened():
        pytest.skip("cv2 无 mp4v 编码器")
    for i in range(30):
        f = np.full((240, 320, 3), 100, dtype=np.uint8)
        cv2.circle(f, (160, 120), 8 + (i % 5), (255, 255, 255), -1)
        w.write(f)
    w.release()
    return str(p)


class TestReadFrameContract:
    def test_total_and_fps_none_tolerated(self, sample_video):
        """total=None 与 0 同语义（不 clamp），fps=None 与 0 同语义（自动读流）：
        显式传 None 不抛 TypeError（旧契约钉住 None 必抛 TypeError，
        P2 防御修复后改为宽容语义——调用方从 JSON/配置读取未校验值不崩溃）。"""
        frame = read_frame(sample_video, 0, total=None, fps=0)
        assert frame is not None
        frame2 = read_frame(sample_video, 0, total=0, fps=None)
        assert frame2 is not None

    def test_out_of_range_frame_returns_none(self, sample_video):
        """超界帧号由 decode 循环自然返回 None（不抛异常）。"""
        info = get_video_info(sample_video)
        assert read_frame(sample_video, info["total"] + 5000, total=0, fps=0) is None

    def test_videoreader_frames_aligned(self, sample_video):
        """VideoReader 帧号从 0 连续对齐（pts 偏移修正回归防护）。"""
        with VideoReader(sample_video) as r:
            idxs = [i for i, _ in r.iter_frames(batch=1)]
        assert idxs == list(range(len(idxs)))     # 0-based 连续
        assert len(idxs) >= 28                    # 30 帧允许容器级 ±2

    def test_videoreader_batch_floor(self, sample_video):
        """iter_frames batch<1 视为 1（防取模除零，P2 防御回归）。"""
        with VideoReader(sample_video) as r:
            idxs = [i for i, _ in r.iter_frames(batch=0)]
        assert idxs                               # 不抛 ZeroDivisionError 且能出帧

    def test_videoreader_del_releases_container(self, sample_video):
        """漏掉 with/close 时 __del__ 兜底释放容器（句柄泄漏回归防护）。"""
        r = VideoReader(sample_video)
        assert r.container is not None
        del r                                     # 触发 __del__ → close()
        import gc
        gc.collect()

    def test_videoreader_decode_errors_zero_on_clean_video(self, sample_video):
        """B2: 正常视频 decode_errors 恒为 0（容错计数不误报）。"""
        with VideoReader(sample_video) as r:
            assert r.decode_errors == 0
            idxs = [i for i, _ in r.iter_frames(batch=1)]
        assert idxs
        assert r.decode_errors == 0


class _FakeStream:
    def __init__(self, start_time, time_base):
        self.start_time = start_time
        self.time_base = time_base


class TestStreamStartPts:
    """B1: 起始 pts 修正阈值以帧为单位（旧版按 0.5 秒漏修真实偏移）。"""

    def test_offset_below_half_second_is_corrected(self):
        """0.4s@30fps = 12 帧：旧版 0.4 < 0.5 秒忽略 → 帧号整体平移 + 开头截尾。"""
        tb = Fraction(1, 90000)
        pts = int(0.4 * 90000)                    # 36000
        assert _stream_start_pts(_FakeStream(pts, tb), 30) == pts

    def test_sub_half_frame_offset_ignored(self):
        """0.01s@30fps = 0.3 帧 < 0.5 帧：亚帧微头部偏移忽略，防取整抖动。"""
        tb = Fraction(1, 90000)
        pts = int(0.01 * 90000)
        assert _stream_start_pts(_FakeStream(pts, tb), 30) == 0

    def test_same_offset_scaled_by_fps(self):
        """同一偏移是否修正由 fps 决定：60fps 超半帧修正，15fps 不足半帧忽略。"""
        tb = Fraction(1, 90000)
        pts = int(0.02 * 90000)                   # 1800
        assert _stream_start_pts(_FakeStream(pts, tb), 60) == pts
        assert _stream_start_pts(_FakeStream(pts, tb), 15) == 0

    def test_none_zero_or_negative_start_ignored(self):
        tb = Fraction(1, 90000)
        assert _stream_start_pts(_FakeStream(None, tb), 30) == 0
        assert _stream_start_pts(_FakeStream(0, tb), 30) == 0
        assert _stream_start_pts(_FakeStream(-5, tb), 30) == 0


class TestScanVideoFiles:
    def test_scan_ignores_fake_dir_and_non_str(self, tmp_path):
        """scan_video_files：伪视频（*.mp4 子目录）不返回；非 str/None 不抛。"""
        import os
        from services import video_utils
        (tmp_path / "a.mp4").write_bytes(b"x")
        (tmp_path / "fake.mp4").mkdir()
        (tmp_path / "b.mov").write_bytes(b"x")
        got = [os.path.basename(p) for p in video_utils.scan_video_files(str(tmp_path))]
        assert got == ["a.mp4", "b.mov"]
        assert video_utils.scan_video_files(None) == []
        assert video_utils.scan_video_files(123) == []
