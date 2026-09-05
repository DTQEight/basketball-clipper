"""video_io 契约测试：read_frame 参数边界（P0-1 回归防护）。"""
import pytest

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")
import numpy as np

from video_io import get_video_info, read_frame, VideoReader


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
