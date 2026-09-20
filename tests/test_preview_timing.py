"""预览切片打点回归：汇总行格式 + 打点接线（等锁/墙钟/回退是否真被记下）。

背景：预览每片段成本 6~9s 且与视频长短无关，需要先量出钱花在等锁/IO/解码/编码
哪一段。这些断言锁住的是「汇总行别崩 + _run_cut 的 (wait, wall) 真被 _cut_one 收走」，
避免以后重构把打点悄悄摘掉（摘掉不会有任何功能报错，只会让优化失去依据）。
"""
import subprocess
import sys

import pytest

import services.detection as detection
from services import state


class _FakeProc:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


class TestPreviewTimingSummary:
    def test_empty_records(self):
        assert detection._preview_timing_summary([]) == '预览打点: 无片段'

    def test_normal_records(self):
        recs = [{"wait": 1.0, "wall": 3.0, "probe": None, "fallback": False},
                {"wait": 2.0, "wall": 5.0, "probe": None, "fallback": False}]
        line = detection._preview_timing_summary(recs)
        assert '2 片段' in line
        assert 'ffmpeg 合计 8.0s' in line
        assert '等锁合计 3.0s' in line
        # 中位/最大：wait [1,2] → 2/2；wall [3,5] → 5/5（_med 取上中位）
        assert '等锁 中位/最大 2.00/2.00s' in line
        assert 'ffmpeg 中位/最大 5.00/5.00s' in line

    def test_probe_share(self):
        recs = [{"wait": 0.0, "wall": 4.0, "probe": 1.0, "fallback": False},
                {"wait": 0.0, "wall": 6.0, "probe": 3.0, "fallback": False}]
        line = detection._preview_timing_summary(recs)
        # 探针中位 3.0 → 编码约占 (中位 wall 6.0 - 3.0)/6.0 = 50%
        assert '探针(只读+解码) 中位 3.00s' in line
        assert '编码约占 50%' in line

    def test_fallback_and_all_failed(self):
        recs = [{"wait": 0.0, "wall": 0.0, "probe": None, "fallback": True},
                {"wait": 0.0, "wall": 0.0, "probe": None, "fallback": False}]
        line = detection._preview_timing_summary(recs)
        assert 'NVENC 回退 1 次' in line
        # 全 0（全失败/全取消）不得崩，也不该冒出一条假的 ffmpeg 中位行
        assert 'ffmpeg 中位' not in line


@pytest.fixture
def _preview_env(tmp_path, monkeypatch):
    """把 _generate_preview_clips 的外部依赖全部替成假件（不跑真 ffmpeg）。"""
    monkeypatch.setattr(state, "CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(detection, "build_encode_args",
                        lambda ff, quality="preview", **kw: (
                            ["-c:v", "libx264"] if kw.get("use_nvenc") is False
                            else ["-c:v", "h264_nvenc"]))
    monkeypatch.setattr(detection, "PREVIEW_PROBE_N", 0)
    import imageio_ffmpeg
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: sys.executable)
    return tmp_path


def _fake_run_factory(monkeypatch, fail_first_nvenc=0):
    """假 ffmpeg：把输出路径（命令末元素）写成非空文件，返回成功。"""
    calls = []

    def _run(cmd, **kw):
        calls.append(cmd)
        if "h264_nvenc" in cmd and len(calls) <= fail_first_nvenc:
            return _FakeProc(returncode=1, stderr="nvenc session limit")
        out = cmd[-1]
        if out != "-":                       # "-" 是探针的 null muxer 输出
            with open(out, "wb") as f:
                f.write(b"fake mp4 bytes")
        return _FakeProc()

    monkeypatch.setattr(subprocess, "run", _run)
    return calls


class TestPreviewTimingWiring:
    def test_summary_logged_and_records_collected(self, _preview_env, monkeypatch, caplog):
        _fake_run_factory(monkeypatch)
        with caplog.at_level("INFO"):
            clips = detection._generate_preview_clips(
                "src.mp4", [1.0, 2.0, 3.0], 0, 10, 30.0, 300, "stamp")
        assert len(clips) == 3
        lines = [r.message for r in caplog.records if r.message.startswith('预览打点:')]
        assert len(lines) == 1
        assert '3 片段' in lines[0]

    def test_nvenc_fallback_counted(self, _preview_env, monkeypatch, caplog):
        calls = _fake_run_factory(monkeypatch, fail_first_nvenc=1)
        with caplog.at_level("INFO"):
            clips = detection._generate_preview_clips(
                "src.mp4", [1.0], 0, 10, 30.0, 300, "stamp")
        assert len(clips) == 1
        assert any("libx264" in c for c in calls)          # 确实回退软编重切了
        line = [r.message for r in caplog.records
                if r.message.startswith('预览打点:')][0]
        assert 'NVENC 回退 1 次' in line

    def test_probe_runs_for_first_n(self, _preview_env, monkeypatch, caplog):
        monkeypatch.setattr(detection, "PREVIEW_PROBE_N", 1)
        calls = _fake_run_factory(monkeypatch)
        with caplog.at_level("INFO"):
            detection._generate_preview_clips(
                "src.mp4", [1.0, 2.0], 0, 10, 30.0, 300, "stamp")
        probes = [c for c in calls if c[-1] == "-"]        # null muxer = 探针
        assert len(probes) == 1                            # 只对前 1 个片段探针
        line = [r.message for r in caplog.records
                if r.message.startswith('预览打点:')][0]
        assert '探针(只读+解码)' in line

    def test_per_clip_line_when_profile_on(self, _preview_env, monkeypatch, caplog):
        monkeypatch.setattr(detection, "PREVIEW_PROFILE", True)
        _fake_run_factory(monkeypatch)
        with caplog.at_level("INFO"):
            detection._generate_preview_clips(
                "src.mp4", [1.0], 0, 10, 30.0, 300, "stamp")
        assert any(r.message.startswith('[打点] 片段 0') for r in caplog.records)
