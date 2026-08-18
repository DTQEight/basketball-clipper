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
    def test_total_must_be_int_not_none(self, sample_video):
        """total=None 是类型违规（on_batch_load_video 曾因此崩溃）：
        合法边界是 total=0（不 clamp）与正整数。此测试钉住契约。"""
        frame = read_frame(sample_video, 0, total=0, fps=0)
        assert frame is not None
        with pytest.raises(TypeError):
            read_frame(sample_video, 0, total=None, fps=0)

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
