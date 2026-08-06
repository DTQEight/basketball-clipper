"""音频峰值检测模块：从视频提取音频，检测进球时的欢呼声/swish 声峰值。

实现思路（轻量，不依赖 librosa）：
  1. 用 ffmpeg 提取音频为 16-bit PCM mono（采样率 8kHz，足够检测能量峰值）
  2. 用 numpy 读取 PCM 数据，计算短时 RMS 能量
  3. 用动态阈值 + 最小间隔检测峰值
  4. 返回峰值时间戳列表（秒）

参考：basketball-highlights 项目 P0 TODO 中的"音频峰值检测"方案。
"""

import os
import subprocess
import tempfile
import numpy as np

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"

# 音频缓存目录（与转码模块一致，放 E 盘避免 C 盘空间不足）
_CACHE_ROOT = r"E:\bball_cache"
_AUDIO_DIR = os.path.join(_CACHE_ROOT, "bball_audio")


def _ensure_audio_dir():
    os.makedirs(_AUDIO_DIR, exist_ok=True)
    return _AUDIO_DIR


def probe_audio(video_path, ffprobe_path=None):
    """探测视频是否包含音频流。

    返回: (has_audio, sample_rate, channels) 或 (False, 0, 0)
    """
    ffprobe = ffprobe_path or "ffprobe"
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, check=True, timeout=30,
            creationflags=0x08000000,
        )
        out = r.stdout.strip()
        if not out:
            return (False, 0, 0)
        parts = out.split(",")
        sr = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
        ch = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return (sr > 0, sr, ch)
    except Exception:
        return (False, 0, 0)


def extract_audio_pcm(video_path, sample_rate=8000, mono=True, timeout=600):
    """从视频提取音频为 16-bit PCM 数据。

    sample_rate: 重采样率（8000Hz 足够检测能量峰值，且数据量小）
    mono: True 单声道
    返回: (np.int16 数组, 实际采样率) 或 (None, 0)
    """
    _ensure_audio_dir()
    # 用管道直接读 PCM，避免写临时文件
    cmd = [FFMPEG, "-v", "error", "-i", video_path,
           "-vn",                 # 不要视频
           "-ar", str(sample_rate),
           "-ac", "1" if mono else "2",
           "-f", "s16le",         # 16-bit little-endian PCM
           "-"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout,
            creationflags=0x08000000,
        )
        raw = proc.stdout
        if len(raw) < 2:
            return (None, 0)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        return (audio, sample_rate)
    except subprocess.TimeoutExpired:
        return (None, 0)
    except Exception:
        return (None, 0)


def compute_rms_energy(audio, sample_rate, frame_ms=50, hop_ms=25):
    """计算短时 RMS 能量。

    frame_ms: 窗长（毫秒）
    hop_ms: 步长（毫秒）
    返回: (energy 数组, 每个能量点对应的时间戳数组)
    """
    if audio is None or len(audio) == 0:
        return (np.array([]), np.array([]))

    frame_len = int(sample_rate * frame_ms / 1000)
    hop_len = int(sample_rate * hop_ms / 1000)
    if frame_len < 2 or hop_len < 1:
        return (np.array([]), np.array([]))

    n = len(audio)
    n_frames = max(0, (n - frame_len) // hop_len + 1)
    if n_frames == 0:
        # 音频太短，整体算一帧
        rms = float(np.sqrt(np.mean(audio ** 2))) if n > 0 else 0.0
        return (np.array([rms]), np.array([0.0]))

    energy = np.zeros(n_frames, dtype=np.float32)
    times = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        seg = audio[start:end]
        energy[i] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) > 0 else 0.0
        times[i] = (start + frame_len / 2) / sample_rate

    return (energy, times)


def detect_peaks(energy, times,
                 threshold_factor=4.0,
                 min_peak_ratio=0.15,
                 min_gap_sec=3.0,
                 smooth_window=5):
    """基于动态阈值检测能量峰值。

    threshold_factor: 峰值需为背景噪声 RMS 的多少倍（4 倍较保守）
    min_peak_ratio: 峰值至少为最大能量的多少比例（过滤小峰）
    min_gap_sec: 峰值间最小间隔（秒）
    smooth_window: 能量平滑窗（帧数）
    返回: 峰值时间戳列表（秒）
    """
    if len(energy) == 0:
        return []

    # 中值滤波平滑能量曲线（去抖动）
    if smooth_window > 1 and len(energy) >= smooth_window:
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        energy_s = np.convolve(energy, kernel, mode="same")
    else:
        energy_s = energy

    # 估计背景噪声：用能量分布的低百分位（避免被峰值拉高）
    bg = float(np.percentile(energy_s, 30))
    bg = max(bg, 1.0)  # 防止除零
    peak_thresh = bg * threshold_factor
    max_energy = float(np.max(energy_s))
    abs_thresh = max_energy * min_peak_ratio

    peaks = []
    last_peak_t = -1e9
    for i in range(len(energy_s)):
        e = energy_s[i]
        if e < peak_thresh:
            continue
        if e < abs_thresh:
            continue
        t = float(times[i])
        # 最小间隔过滤
        if t - last_peak_t < min_gap_sec:
            # 取较大的峰
            if peaks and e > energy_s[int((last_peak_t - times[0]) / (times[1] - times[0] + 1e-9))]:
                peaks[-1] = t
                last_peak_t = t
            continue
        peaks.append(t)
        last_peak_t = t

    return peaks


class AudioPeakDetector:
    """音频峰值检测器：封装从视频到峰值时间戳的完整流程。"""

    def __init__(self, sample_rate=8000, frame_ms=50, hop_ms=25,
                 threshold_factor=4.0, min_peak_ratio=0.15,
                 min_gap_sec=3.0, smooth_window=5):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.threshold_factor = threshold_factor
        self.min_peak_ratio = min_peak_ratio
        self.min_gap_sec = min_gap_sec
        self.smooth_window = smooth_window

        # 结果缓存
        self.audio = None
        self.energy = None
        self.energy_times = None
        self.peaks = []

    def analyze(self, video_path, ffprobe_path=None):
        """分析视频音频，返回峰值时间戳列表。"""
        has_audio, sr, ch = probe_audio(video_path, ffprobe_path)
        if not has_audio:
            self.peaks = []
            self.audio = None
            self.energy = None
            self.energy_times = None
            return []

        self.audio, self.sample_rate = extract_audio_pcm(
            video_path, sample_rate=self.sample_rate, mono=True)
        if self.audio is None or len(self.audio) == 0:
            self.peaks = []
            return []

        self.energy, self.energy_times = compute_rms_energy(
            self.audio, self.sample_rate,
            frame_ms=self.frame_ms, hop_ms=self.hop_ms)

        self.peaks = detect_peaks(
            self.energy, self.energy_times,
            threshold_factor=self.threshold_factor,
            min_peak_ratio=self.min_peak_ratio,
            min_gap_sec=self.min_gap_sec,
            smooth_window=self.smooth_window)

        return self.peaks

    def get_energy_curve(self, max_points=2000):
        """获取能量曲线（用于 UI 可视化），下采样到 max_points 点。"""
        if self.energy is None or len(self.energy) == 0:
            return np.array([]), np.array([])
        n = len(self.energy)
        if n <= max_points:
            return self.energy_times, self.energy
        step = n // max_points
        return self.energy_times[::step], self.energy[::step]

    def get_stats(self):
        """获取统计信息（用于 UI 展示）。"""
        if self.energy is None:
            return {"has_audio": False, "peaks": 0, "duration": 0.0,
                    "bg_noise": 0.0, "max_energy": 0.0}
        bg = float(np.percentile(self.energy, 30)) if len(self.energy) > 0 else 0.0
        return {
            "has_audio": True,
            "peaks": len(self.peaks),
            "duration": float(self.energy_times[-1]) if len(self.energy_times) > 0 else 0.0,
            "bg_noise": bg,
            "max_energy": float(np.max(self.energy)) if len(self.energy) > 0 else 0.0,
            "peak_threshold": bg * self.threshold_factor,
        }


def fuse_signals(visual_goals, audio_peaks, fusion_window=2.0):
    """融合视觉和音频信号。

    visual_goals: 视觉检测到的进球时间戳列表
    audio_peaks: 音频检测到的峰值时间戳列表
    fusion_window: 时间窗口（秒），视觉和音频在此窗口内同时出现视为融合确认

    返回: {
        "fused": [(ts, "fused"), ...] 高置信度（视觉+音频都触发）
        "visual_only": [(ts, "visual"), ...] 中置信度（仅视觉）
        "audio_only": [(ts, "audio"), ...] 低置信度候选（仅音频）
        "all": 完整列表（按时间排序）
    }
    """
    visual_used = set()
    audio_used = set()

    fused = []
    for i, v in enumerate(visual_goals):
        # 找最近的音频峰值
        best_j = -1
        best_dist = fusion_window
        for j, a in enumerate(audio_peaks):
            if j in audio_used:
                continue
            d = abs(a - v)
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j >= 0:
            fused.append((float(v), "fused", best_dist))
            visual_used.add(i)
            audio_used.add(best_j)

    visual_only = [(float(v), "visual", 0.0)
                   for i, v in enumerate(visual_goals) if i not in visual_used]
    audio_only = [(float(a), "audio", 0.0)
                  for j, a in enumerate(audio_peaks) if j not in audio_used]

    all_events = sorted(fused + visual_only + audio_only, key=lambda x: x[0])

    return {
        "fused": fused,
        "visual_only": visual_only,
        "audio_only": audio_only,
        "all": all_events,
    }
