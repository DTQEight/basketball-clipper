"""全局状态与持久化（历史记录、片段缓存）。

所有运行时状态集中在此模块，业务函数和 UI 层通过 `from services import state`
访问/修改状态，避免全局变量散落各处。
"""
import os
import json
import time
import threading
import logging
from logging.handlers import TimedRotatingFileHandler
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

# 预览片段缓存最大条目数（超出按插入顺序驱逐最旧）
CLIP_CACHE_MAX_ENTRIES = 20

# 预览片段目录
DEMO_OUTPUT_DIR = os.path.join(CACHE_ROOT, "demo_output")


# ============ 日志 ============
def setup_logging():
    """初始化日志：控制台 + 按日轮转文件（cache/logs/app.log，保留 7 天）。

    旧实现只靠 start 脚本 tee 落盘，直接 `python demo_nicegui.py` 启动时
    没有任何日志文件；日志能力应回归程序自身。formatter 用裸消息，
    控制台输出与旧 print 行为完全一致。
    """
    log_dir = os.path.join(CACHE_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:          # 防重复初始化（reload/二次调用）
        return
    fmt = logging.Formatter("%(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = TimedRotatingFileHandler(os.path.join(log_dir, "app.log"),
                                  when="midnight", backupCount=7, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    root.setLevel(logging.INFO)


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

# 检测取消标志（UI 点击「取消」时 set，检测循环轮询后中断）
# 用 threading.Event 替代裸 bool：跨线程标志的 set/clear/is_set 天然原子且语义明确
cancel_event = threading.Event()

# 预览片段缓存：key=(视频路径, 进球时间戳元组) -> [片段dict, ...]
clip_cache = {}

# 文件夹批量模式状态
batch_files = []
batch_calibs = {}
batch_current_video = None

# 批量结果快照：视频路径 -> {"goals": [...], "clips": [...], "kept": set(), "finished_at": str}
# 检测线程只写入新 key；前台人工确认（删卡片/导出集锦）只读写快照。
# 与全局 last_goals/last_goal_clips 完全隔离，支持流水线：后台跑检测 + 前台确认已完成视频
batch_results = {}


# ============ 历史记录 ============
def load_history() -> list:
    """加载历史记录列表。

    文件损坏（JSON 解析失败）时先改名备份为 .corrupt-<时间戳>.bak 再返回空列表：
    旧实现静默返回 [] 后，下一次 add_history 会用空列表覆盖写整个文件，
    一次写盘中断/损坏 = 全部历史丢失。
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        bak = f"{HISTORY_FILE}.corrupt-{time.strftime('%Y%m%d%H%M%S')}.bak"
        try:
            os.replace(HISTORY_FILE, bak)
            logging.getLogger("state").warning(
                f"[WARN] 历史记录解析失败（{e}），原文件已备份到 {bak}，将以空记录重新开始")
        except OSError:
            logging.getLogger("state").warning(
                f"[WARN] 历史记录解析失败（{e}），且备份失败: {bak}")
        return []


def save_history(records):
    """保存历史记录到磁盘。"""
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] 保存历史记录失败: {e}")


# add_history 可选字段表：字段名 -> (类型转换函数, 四舍五入位数或 None)
# 新增字段只需在此表加一行（旧实现要同时改签名/赋值块/调用方三处）。
# diff_threshold 例外：可能为 int 或 str 'auto'，原样保存，不入表。
_HISTORY_FIELD_CASTS = {
    # 核心检测参数
    "ball_conf": (float, None),
    "min_gap_sec": (float, None),
    "auto_threshold": (bool, None),
    "yolo_step": (int, None),
    "skip_yolo_no_motion": (bool, None),
    "min_circularity": (float, None),
    "min_in_hoop_frames": (int, None),
    "min_blob_area": (int, None),
    "search_margin": (int, None),
    # 时间/批量
    "elapsed_sec": (float, 1),
    "batch_idx": (int, None),
    "batch_total": (int, None),
    "detect_start_time": (str, None),
    "detect_end_time": (str, None),
    # 视频元信息
    "video_fps": (float, 2),
    "video_width": (int, None),
    "video_height": (int, None),
    "video_total_frames": (int, None),
    "video_duration_sec": (float, 1),
    # 处理速度指标
    "processed_frames": (int, None),
    "proc_fps": (float, 1),
    "speed_vs_realtime": (float, 2),
    # YOLO 跳过统计
    "yolo_called": (int, None),
    "yolo_cond_skipped": (int, None),
    "yolo_skip_rate_pct": (float, 1),
    # YOLO 确认/否决统计
    "yolo_confirmed": (int, None),
    "yolo_rejected": (int, None),
    "yolo_confirm_rate_pct": (float, 1),
    # 进球路径细分
    "cross_above": (int, None),
    "cross_below": (int, None),
    "in_hoop": (int, None),
    "reject_cooldown": (int, None),
    # 自适应阈值详情
    "auto_threshold_value": (int, None),
    "warmup_p95_median": (float, 1),
    "warmup_sample_count": (int, None),
}


def add_history(video_path, hoop, goals, baseline_idx=-1,
                diff_threshold=None, **fields):
    """添加一条历史记录（同视频会覆盖旧记录）。

    可选字段见 _HISTORY_FIELD_CASTS；未知字段抛 TypeError（防调用方拼错字段名静默丢数据）。
    """
    unknown = set(fields) - set(_HISTORY_FIELD_CASTS)
    if unknown:
        raise TypeError(f"add_history 收到未知字段: {sorted(unknown)}")
    records = load_history()
    records = [r for r in records if r.get("video") != video_path]
    rec = {
        "video": video_path,
        "video_name": os.path.basename(video_path),
        "hoop": list(hoop) if hoop else None,
        "baseline_idx": int(baseline_idx),
        "goals": [float(t) for t in goals],
        "total": len(goals),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if diff_threshold is not None:
        rec["diff_threshold"] = diff_threshold  # 可能为 int 或 str 'auto'
    for name, value in fields.items():
        if value is None:
            continue
        cast, ndigits = _HISTORY_FIELD_CASTS[name]
        value = cast(value)
        if ndigits is not None:
            value = round(value, ndigits)
        rec[name] = value
    records.insert(0, rec)
    records = records[:MAX_HISTORY_RECORDS]
    save_history(records)


# ============ 片段缓存 ============
def clip_cache_key(video_path: str, goals) -> tuple:
    """构造片段缓存 key：进球时间戳排序 + 保留 3 位小数。

    写入方（检测完成，goals 已 sorted）与读取方（历史回读，原序）必须
    用同一规则生成 key，否则顺序不同会导致缓存永不命中。
    """
    return (video_path, tuple(sorted(round(float(t), 3) for t in goals)))


def put_clip_cache(key, clips):
    """写入片段缓存条目 + 超限驱逐最旧 + 落盘（三个调用点共用的完整流程）。"""
    clip_cache[key] = list(clips)
    if len(clip_cache) > CLIP_CACHE_MAX_ENTRIES:
        clip_cache.pop(next(iter(clip_cache)))
    save_clip_cache()


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
        logging.getLogger("state").warning(f"[WARN] 保存片段缓存失败: {e}")


def init_clip_cache():
    """启动时调用，从磁盘恢复片段缓存到内存。"""
    clip_cache.clear()
    clip_cache.update(load_clip_cache())
