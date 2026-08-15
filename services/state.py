"""全局状态与持久化（历史记录、片段缓存）。

所有运行时状态集中在此模块，业务函数和 UI 层通过 `from services import state`
访问/修改状态，避免全局变量散落各处。
"""
import os
import json
import time
from pathlib import Path

# ============ 路径常量 ============
# 项目根目录（扁平结构：代码直接在项目根下）
_ROOT = Path(__file__).parent.parent.resolve()

# 缓存目录：优先用环境变量，否则用项目内 cache 目录
if os.environ.get("BBALL_CACHE_ROOT"):
    CACHE_ROOT = os.environ["BBALL_CACHE_ROOT"]
else:
    CACHE_ROOT = str(_ROOT / "cache")

# Windows subprocess 屏蔽控制台窗口（Linux 下为 0）
SBOX = 0x08000000 if os.name == "nt" else 0

# 临时文件重定向到缓存目录
_tmp_root = os.path.join(CACHE_ROOT, "tmp")
os.makedirs(_tmp_root, exist_ok=True)
os.environ["TMPDIR"] = _tmp_root
os.environ["TEMP"] = _tmp_root
os.environ["TMP"] = _tmp_root

# 历史记录文件
HISTORY_FILE = os.path.join(CACHE_ROOT, "detection_history.json")
# 预览片段缓存索引（持久化，重启后可复用上一次生成的片段）
CLIP_CACHE_FILE = os.path.join(CACHE_ROOT, "clip_cache.json")

DEFAULT_VIDEO = ""
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ts"}

# 历史记录上限（防止 detection_history.json 无限增长）
MAX_HISTORY_RECORDS = 50

# 预览片段目录
DEMO_OUTPUT_DIR = os.path.join(CACHE_ROOT, "demo_output")


def _purge_old_clips(max_days=7):
    """启动时自动清理 demo_output/ 下超过 max_days 天的旧预览片段。"""
    out = DEMO_OUTPUT_DIR
    if not os.path.isdir(out):
        return
    cutoff = time.time() - max_days * 86400
    for fname in os.listdir(out):
        fpath = os.path.join(out, fname)
        try:
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
        except OSError:
            pass


_purge_old_clips()

# ============ 运行时状态 ============
video_state = {"path": None, "total": 0, "fps": 30.0, "codec": "unknown",
               "current_frame": 0, "width": 0, "height": 0}
calib = {
    "clicks": [],
    "hoop": None,
    "baseline_frame": None,
    "baseline_idx": -1,
}
last_goals = []
last_goal_clips = []
kept_goal_indices = set()

# 检测取消标志（UI 点击「取消」时置 True，检测循环轮询后中断）
cancel_requested = False

# 预览片段缓存：key=(视频路径, 进球时间戳元组) -> [片段dict, ...]
clip_cache = {}

# 文件夹批量模式状态
batch_files = []
batch_calibs = {}
batch_current_video = None


# ============ 历史记录 ============
def load_history():
    """加载历史记录列表。"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_history(records):
    """保存历史记录到磁盘。"""
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 保存历史记录失败: {e}", flush=True)


def add_history(video_path, hoop, goals, kept_goals, baseline_idx=-1,
                ball_conf=None, min_gap_sec=None, diff_threshold=None,
                auto_threshold=None, yolo_step=None, skip_yolo_no_motion=None,
                min_circularity=None, min_in_hoop_frames=None,
                min_blob_area=None, search_margin=None,
                elapsed_sec=None, batch_idx=None, batch_total=None,
                detect_start_time=None, detect_end_time=None,
                # ===== 视频元信息 =====
                video_fps=None, video_width=None, video_height=None,
                video_total_frames=None, video_duration_sec=None,
                # ===== 处理速度指标 =====
                processed_frames=None, proc_fps=None, speed_vs_realtime=None,
                # ===== YOLO 跳过/条件跳过 =====
                yolo_called=None, yolo_cond_skipped=None, yolo_skip_rate_pct=None,
                # ===== YOLO 路径确认/否决 =====
                yolo_confirmed=None, yolo_rejected=None, yolo_confirm_rate_pct=None,
                # ===== 进球路径细分 =====
                cross_above=None, cross_below=None, in_hoop=None, reject_cooldown=None,
                # ===== 自适应阈值详情 =====
                auto_threshold_value=None, warmup_p95_median=None, warmup_sample_count=None):
    """添加一条历史记录（同视频会覆盖旧记录）。"""
    records = load_history()
    records = [r for r in records if r.get("video") != video_path]
    rec = {
        "video": video_path,
        "video_name": os.path.basename(video_path),
        "hoop": list(hoop) if hoop else None,
        "baseline_idx": int(baseline_idx),
        "goals": [float(t) for t in goals],
        "kept_goals": [float(t) for t in kept_goals],
        "total": len(goals),
        "kept": len(kept_goals),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # 核心检测参数
    if ball_conf is not None:
        rec["ball_conf"] = float(ball_conf)
    if min_gap_sec is not None:
        rec["min_gap_sec"] = float(min_gap_sec)
    if diff_threshold is not None:
        rec["diff_threshold"] = diff_threshold  # 可能为 int 或 str 'auto'
    if auto_threshold is not None:
        rec["auto_threshold"] = bool(auto_threshold)
    if yolo_step is not None:
        rec["yolo_step"] = int(yolo_step)
    if skip_yolo_no_motion is not None:
        rec["skip_yolo_no_motion"] = bool(skip_yolo_no_motion)
    if min_circularity is not None:
        rec["min_circularity"] = float(min_circularity)
    if min_in_hoop_frames is not None:
        rec["min_in_hoop_frames"] = int(min_in_hoop_frames)
    if min_blob_area is not None:
        rec["min_blob_area"] = int(min_blob_area)
    if search_margin is not None:
        rec["search_margin"] = int(search_margin)
    # 时间/批量
    if elapsed_sec is not None:
        rec["elapsed_sec"] = round(float(elapsed_sec), 1)
    if batch_idx is not None:
        rec["batch_idx"] = int(batch_idx)
    if batch_total is not None:
        rec["batch_total"] = int(batch_total)
    if detect_start_time is not None:
        rec["detect_start_time"] = str(detect_start_time)
    if detect_end_time is not None:
        rec["detect_end_time"] = str(detect_end_time)
    # 视频元信息
    if video_fps is not None:
        rec["video_fps"] = round(float(video_fps), 2)
    if video_width is not None:
        rec["video_width"] = int(video_width)
    if video_height is not None:
        rec["video_height"] = int(video_height)
    if video_total_frames is not None:
        rec["video_total_frames"] = int(video_total_frames)
    if video_duration_sec is not None:
        rec["video_duration_sec"] = round(float(video_duration_sec), 1)
    # 处理速度指标
    if processed_frames is not None:
        rec["processed_frames"] = int(processed_frames)
    if proc_fps is not None:
        rec["proc_fps"] = round(float(proc_fps), 1)
    if speed_vs_realtime is not None:
        rec["speed_vs_realtime"] = round(float(speed_vs_realtime), 2)
    # YOLO 跳过统计
    if yolo_called is not None:
        rec["yolo_called"] = int(yolo_called)
    if yolo_cond_skipped is not None:
        rec["yolo_cond_skipped"] = int(yolo_cond_skipped)
    if yolo_skip_rate_pct is not None:
        rec["yolo_skip_rate_pct"] = round(float(yolo_skip_rate_pct), 1)
    # YOLO 确认/否决统计
    if yolo_confirmed is not None:
        rec["yolo_confirmed"] = int(yolo_confirmed)
    if yolo_rejected is not None:
        rec["yolo_rejected"] = int(yolo_rejected)
    if yolo_confirm_rate_pct is not None:
        rec["yolo_confirm_rate_pct"] = round(float(yolo_confirm_rate_pct), 1)
    # 进球路径细分
    if cross_above is not None:
        rec["cross_above"] = int(cross_above)
    if cross_below is not None:
        rec["cross_below"] = int(cross_below)
    if in_hoop is not None:
        rec["in_hoop"] = int(in_hoop)
    if reject_cooldown is not None:
        rec["reject_cooldown"] = int(reject_cooldown)
    # 自适应阈值详情
    if auto_threshold_value is not None:
        rec["auto_threshold_value"] = int(auto_threshold_value)
    if warmup_p95_median is not None:
        rec["warmup_p95_median"] = round(float(warmup_p95_median), 1)
    if warmup_sample_count is not None:
        rec["warmup_sample_count"] = int(warmup_sample_count)
    records.insert(0, rec)
    records = records[:MAX_HISTORY_RECORDS]
    save_history(records)


# ============ 片段缓存 ============
def load_clip_cache():
    """从磁盘加载片段缓存索引，仅保留片段文件仍存在的条目。"""
    cache = {}
    try:
        if os.path.exists(CLIP_CACHE_FILE):
            with open(CLIP_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                key = (item["video"], tuple(float(g) for g in item["goals"]))
                clips = [{"ts": float(c["ts"]), "path": c["path"], "idx": int(c["idx"])}
                         for c in item.get("clips", [])]
                if clips and all(os.path.exists(c["path"]) for c in clips):
                    cache[key] = clips
    except Exception:
        pass
    return cache


def save_clip_cache():
    """把内存片段缓存索引写入磁盘。"""
    try:
        data = [{"video": v, "goals": list(g),
                 "clips": [{"ts": c["ts"], "path": c["path"], "idx": c["idx"]} for c in clips]}
                for (v, g), clips in clip_cache.items()]
        os.makedirs(os.path.dirname(CLIP_CACHE_FILE), exist_ok=True)
        with open(CLIP_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] 保存片段缓存失败: {e}", flush=True)


def init_clip_cache():
    """启动时调用，从磁盘恢复片段缓存到内存。"""
    clip_cache.clear()
    clip_cache.update(load_clip_cache())
