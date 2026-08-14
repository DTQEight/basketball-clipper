"""全局状态与持久化（历史记录、片段缓存）。

所有运行时状态集中在此模块，业务函数和 UI 层通过 `from services import state`
访问/修改状态，避免全局变量散落各处。
"""
import os
import json
import time
from pathlib import Path

# ============ 路径常量 ============
# 项目根目录：basketball-clipper/
_ROOT = Path(__file__).parent.parent.resolve()

# 缓存目录：优先用环境变量，否则用项目同级 cache 目录（跨平台、可移植）
if os.environ.get("BBALL_CACHE_ROOT"):
    CACHE_ROOT = os.environ["BBALL_CACHE_ROOT"]
else:
    CACHE_ROOT = str(_ROOT.parent / "cache")

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

DEFAULT_VIDEO = r"D:\Downloads\highlights.mp4"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ts"}

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


def add_history(video_path, hoop, goals, kept_goals, baseline_idx=-1):
    """添加一条历史记录（同视频会覆盖旧记录）。"""
    records = load_history()
    records = [r for r in records if r.get("video") != video_path]
    records.insert(0, {
        "video": video_path,
        "video_name": os.path.basename(video_path),
        "hoop": list(hoop) if hoop else None,
        "baseline_idx": int(baseline_idx),
        "goals": [float(t) for t in goals],
        "kept_goals": [float(t) for t in kept_goals],
        "total": len(goals),
        "kept": len(kept_goals),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    records = records[:50]
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
