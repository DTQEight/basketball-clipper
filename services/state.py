"""全局状态与持久化（历史记录、片段缓存）。

所有运行时状态集中在此模块，业务函数和 UI 层通过 `from services import state`
访问/修改状态，避免全局变量散落各处。
"""
import os
import json
import shutil
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
# 检测断点（断点续识别）：每个视频一份，检测成功/彻底失败后清理
CHECKPOINT_FILE = os.path.join(CACHE_ROOT, "detect_checkpoint.json")

DEFAULT_VIDEO = ""
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ts"}

# 历史记录不限量保留：hoop 标定和人工标签都存在记录里，
# 截断会永久丢失旧视频的标定数据（曾导致 755 个训练事件 hoop 无法恢复）
MAX_HISTORY_RECORDS = None

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
    """启动时自动清理 demo_output/ 下超过 max_days 天的旧预览片段。

    `-highlights.mp4` 后缀的集锦是用户最终产物，豁免清理
    （旧实现与预览片段同目录无差别删除，一周后集锦静默消失）。
    """
    out = DEMO_OUTPUT_DIR
    if not os.path.isdir(out):
        return
    cutoff = time.time() - max_days * 86400
    for fname in os.listdir(out):
        if fname.endswith("-highlights.mp4"):
            continue  # 集锦成品不按临时文件清理
        fpath = os.path.join(out, fname)
        try:
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
        except OSError:
            pass


def _purge_orphan_clip_dirs(max_days=1):
    """启动时清理 clips/ 下进程残留的 hl-* 临时目录。

    cut_clips 正常路径会清理自己的临时目录，但进程被 taskkill/崩溃时
    finally 不执行，残留目录（每个可达数百 MB）会无限累积。
    """
    clips_root = os.path.join(CACHE_ROOT, "clips")
    if not os.path.isdir(clips_root):
        return
    cutoff = time.time() - max_days * 86400
    for dname in os.listdir(clips_root):
        dpath = os.path.join(clips_root, dname)
        if not dname.startswith("hl-") or not os.path.isdir(dpath):
            continue
        try:
            if os.path.getmtime(dpath) < cutoff:
                shutil.rmtree(dpath, ignore_errors=True)
        except OSError:
            pass


_purge_old_clips()
_purge_orphan_clip_dirs()

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
# 最近一次 run_detect 的篮筐移位轨迹 [{"frame","ts","hoop"}]
# （检测中支架被撞时由 hoop_tracker 写入，批量快照验证按事件时间取坐标用）
last_hoop_track = []

# 检测取消标志（UI 点击「取消」时 set，检测循环轮询后中断）
# 用 threading.Event 替代裸 bool：跨线程标志的 set/clear/is_set 天然原子且语义明确
cancel_event = threading.Event()

# 流水线集锦独立取消标志：批量检测的「取消」不应连带杀死正在进行的
# 流水线集锦生成（两者是对不同对象的独立操作，见 detection.generate_highlights）
hl_cancel_event = threading.Event()

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

# ============ 批量任务全局进度（跨页面连接/刷新可见）============
# NiceGUI 页面函数每连接执行一次：浏览器刷新销毁旧页面的进度条，
# 但 io_bound 检测线程还在跑。进度必须写在进程级 state，
# 新页面加载时读它恢复「识别中」UI（进度条/取消按钮/已完成列表）。
# 线程安全：检测线程是唯一写者（每视频一次 + 每帧回调），UI 只读；
# dict 整体替换赋值在 CPython 下原子，读侧拿到新旧快照之一，均一致。
batch_task_status = {
    "running": False,      # 批量识别是否在跑
    "current": None,       # 当前正在检测的视频名（basename）
    "index": 0,            # 当前是第几个视频（1-based）
    "total": 0,            # 本批视频总数
    "pct": 0.0,            # 整批百分比 0~100（含单视频内部进度）
    "message": "",         # 最近一条进度消息
    "started_at": "",      # 开始时间
    "cancel_requested": False,  # UI 请求取消（页面刷新后取消按钮仍可用）
}

# ============ 全局任务互斥（进程级，跨页面连接/刷新共享）============
# 旧实现的 _busy 锁在 NiceGUI 页面函数局部：每个浏览器连接/刷新独立执行页面函数，
# 刷新后旧检测线程仍在跑、新页面可再启动任务 → 两线程并发 clear/extend 同一列表、
# 并发写历史文件、cancel_event 被误 clear。锁必须下沉到进程级模块。
#
# 锁的所有权归**任务本体**（run_detect / run_batch_detect / generate_highlights
# 在自己的 finally 里 release）：UI 协程在页面刷新/断开时会被取消，
# io_bound 线程却无法取消继续跑，若由 UI finally release 会造成
# "锁已释放、旧线程还在写 state"的并发窗口。
# task_token 用于 UI 侧防御：release 只对持有当前 token 的任务生效。
task_lock = threading.Lock()
_busy_task = None
_task_token = 0


def try_acquire_task(task: str) -> int:
    """尝试占用全局任务锁（原子 test-and-set）；已被占用返回 0。

    task: 'detect' / 'batch' / 'highlights' / 'load' 等任务名
    返回: token（正整数，释放时校验）；0 表示占用失败
    """
    global _busy_task, _task_token
    with task_lock:
        if _busy_task is not None:
            return 0
        _busy_task = task
        _task_token += 1
        return _task_token


def release_task(token: int = 0) -> None:
    """释放全局任务锁。

    token 传入时仅当与当前持有 token 匹配才释放：
    防止旧任务（已超时/被 UI 提前放弃）误释放新任务的锁。
    """
    global _busy_task
    with task_lock:
        if token and token != _task_token:
            return
        _busy_task = None


def current_task():
    """返回当前占用锁的任务名（无任务返回 None）。"""
    with task_lock:
        return _busy_task


# 流水线集锦小锁（批量运行中对快照视频生成集锦，不占全局任务锁）。
# 同理下沉到模块级：页面局部 dict 刷新后失效会导致两连接同时跑集锦。
hl_busy = {"on": False}

# NVENC 编码会话信号量：GeForce 消费卡驱动限制同时 2 路编码会话。
# 预览片段线程池（services/detection）与集锦（cutter/ffmpeg_cutter）跨模块
# 共用同一配额：超限时第 3 路在 acquire 上排队等待，而不是 OpenEncodeSession 失败。
nvenc_semaphore = threading.Semaphore(2)


# ============ 历史记录 ============
def _atomic_write_json(path, data, indent=None):
    """原子写 JSON：先写临时文件再 os.replace，失败清理残留临时文件。

    直接 open("w") 覆盖写时，进程中途退出/断电会留下半截 JSON；
    temp + replace 保证磁盘上要么是完整旧文件、要么是完整新文件。
    Windows 下目标文件恰好被其他线程读取时 replace 抛 PermissionError，
    做 3 次短重试（读写窗口只有几十 ms）。
    """
    tmp = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(3):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def load_history() -> list:
    """加载历史记录列表。

    - JSON 解析失败（真实损坏）：先改名备份为 .corrupt-<时间戳>.bak 再返回空列表，
      避免下一次 add_history 用空列表覆盖写丢失全部历史。
    - IO 异常（如 Windows 共享冲突，文件恰好被写入方替换中）：短重试后抛出 OSError，
      **不动原文件** —— 旧实现把这类瞬态错误也当损坏改名，完好的历史会被误"清空"。
      调用方需捕获 OSError（add_history 内部已处理：读取失败时放弃写入，不覆盖）。
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    last_err = None
    for attempt in range(3):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            bak = f"{HISTORY_FILE}.corrupt-{time.strftime('%Y%m%d%H%M%S')}.bak"
            try:
                os.replace(HISTORY_FILE, bak)
                logging.getLogger("state").warning(
                    f"[WARN] 历史记录解析失败（{e}），原文件已备份到 {bak}，将以空记录重新开始")
            except OSError:
                logging.getLogger("state").warning(
                    f"[WARN] 历史记录解析失败（{e}），且备份失败: {bak}")
            return []
        except UnicodeDecodeError as e:
            # 文件被外部工具以 GBK/ANSI 保存等编码错误：同样按损坏备份处理
            # （ValueError 子类，不并入 OSError 分支会被 add_history 漏接，
            #  把检测成功的结果整体报成失败）
            bak = f"{HISTORY_FILE}.corrupt-{time.strftime('%Y%m%d%H%M%S')}.bak"
            try:
                os.replace(HISTORY_FILE, bak)
                logging.getLogger("state").warning(
                    f"[WARN] 历史记录编码错误（{e}），原文件已备份到 {bak}，将以空记录重新开始")
            except OSError:
                logging.getLogger("state").warning(
                    f"[WARN] 历史记录编码错误（{e}），且备份失败: {bak}")
            return []
        except OSError as e:
            last_err = e
            time.sleep(0.05 * (attempt + 1))
    raise last_err


def save_history(records) -> bool:
    """保存历史记录到磁盘（原子写）。返回是否成功。"""
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        _atomic_write_json(HISTORY_FILE, records, indent=2)
        return True
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] 保存历史记录失败: {e}")
        return False


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
                diff_threshold=None,
                kept_ts_list=None, deleted_ts_list=None, label_time=None,
                hoop_track=None,
                **fields):
    """添加一条历史记录（同视频会覆盖旧记录）。

    平铺可选字段见 _HISTORY_FIELD_CASTS；未知字段抛 TypeError（防调用方拼错字段名静默丢数据）。
    kept_ts_list / deleted_ts_list / label_time 是特殊嵌套字段（组成 "labels" 块），
    不走单值 cast 表。
    hoop_track 是特殊嵌套字段：检测中途篮筐移位的轨迹
    [{"frame", "ts", "hoop"}]，验证阶段按事件时间取各自的篮筐坐标。
    返回: 新记录 dict；读取/写入失败（IO 错误）时返回 None 且**不覆盖**磁盘旧数据。
    """
    unknown = set(fields) - set(_HISTORY_FIELD_CASTS)
    if unknown:
        raise TypeError(f"add_history 收到未知字段: {sorted(unknown)}")
    try:
        records = load_history()
    except OSError as e:
        logging.getLogger("state").warning(
            f"[WARN] 历史记录读取失败，本次结果未写入（避免空列表覆盖旧数据）: {e}")
        return None
    # 同视频重跑：先取出旧记录的人工标签再覆盖（不显式传标签时继承，防止重跑丢标注）
    old_labels = None
    kept_records = []
    for r in records:
        if r.get("video") == video_path:
            if old_labels is None:
                old_labels = r.get("labels")
        else:
            kept_records.append(r)
    records = kept_records
    rec = {
        "video": video_path,
        "video_name": os.path.basename(video_path),
        "hoop": list(hoop) if hoop else None,
        "baseline_idx": int(baseline_idx),
        "goals": [float(t) for t in goals],
        "total": len(goals),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # 人工确认标签（L4 训练飞轮）
    if kept_ts_list is not None or deleted_ts_list is not None:
        labels: dict = {}
        if kept_ts_list is not None:
            labels["kept"] = sorted({round(float(t), 3) for t in kept_ts_list})
        if deleted_ts_list is not None:
            labels["deleted"] = sorted({round(float(t), 3) for t in deleted_ts_list})
        labels["label_time"] = label_time or rec["time"]
        rec["labels"] = labels
    elif old_labels:
        # 继承旧标签：时间戳与新 goals 对得上的在读取端继续生效，对不上的自动忽略
        rec["labels"] = old_labels
    if diff_threshold is not None:
        rec["diff_threshold"] = diff_threshold  # 可能为 int 或 str 'auto'
    # 篮筐移位轨迹（检测中途支架被撞等）：验证阶段按事件时间取各自坐标
    if hoop_track:
        rec["hoop_track"] = [
            {"frame": int(t.get("frame", 0)),
             "ts": round(float(t.get("ts", 0.0)), 2),
             "hoop": [int(v) for v in t.get("hoop", [])]}
            for t in hoop_track
        ]
    for name, value in fields.items():
        if value is None:
            continue
        cast, ndigits = _HISTORY_FIELD_CASTS[name]
        value = cast(value)
        if ndigits is not None:
            value = round(value, ndigits)
        rec[name] = value
    records.insert(0, rec)
    if not save_history(records):
        return None
    return rec


def _find_history_record(records, video_path):
    """按 video 字段找记录：精确匹配优先，失败后按文件名兜底
    （视频迁移目录后历史记录里还是旧路径，basename 一致即视为同一条）。
    返回记录下标或 None。
    """
    for i, r in enumerate(records):
        if r.get("video") == video_path:
            return i
    base = os.path.basename(video_path)
    for i, r in enumerate(records):
        rv = r.get("video", "")
        if rv and os.path.basename(rv) == base:
            return i
    return None


def update_history_labels(video_path, kept_ts_list, deleted_ts_list):
    """对已有历史记录打/更新人工确认标签（增量写，不重建整条记录，不会丢检测元信息）。

    找不到对应记录时返回 False；写入磁盘成功返回 True。
    """
    try:
        records = load_history()
    except OSError:
        return False
    hit_idx = _find_history_record(records, video_path)
    if hit_idx is None:
        return False
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    target = records[hit_idx]
    labels = dict(target.get("labels") or {})
    if kept_ts_list is not None:
        labels["kept"] = sorted({round(float(t), 3) for t in kept_ts_list})
    if deleted_ts_list is not None:
        labels["deleted"] = sorted({round(float(t), 3) for t in deleted_ts_list})
    labels["label_time"] = now
    target["labels"] = labels
    target["time"] = now
    # 上浮到顶部（最近更新优先），保持其他条目顺序
    final = [target] + records[:hit_idx] + records[hit_idx + 1:]
    return save_history(final)


def get_labels(video_path):
    """读取某视频的已保存标签。

    返回 {"kept": list|None, "deleted": list|None, "label_time": str|None}。
    找不到记录 / 无标签时返回 None 字段。
    """
    try:
        records = load_history()
    except OSError:
        return {"kept": None, "deleted": None, "label_time": None}
    hit_idx = _find_history_record(records, video_path)
    if hit_idx is None:
        return {"kept": None, "deleted": None, "label_time": None}
    lab = records[hit_idx].get("labels") or {}
    return {"kept": list(lab["kept"]) if "kept" in lab else None,
            "deleted": list(lab["deleted"]) if "deleted" in lab else None,
            "label_time": lab.get("label_time")}


def update_history_marks(video_path, clips):
    """verify 完成后把自动分数/标记回写历史记录（clip_marks: ts字符串 → {score,mark,mark_source}）。

    片段缓存被驱逐/重启清空时的兜底：历史回读按 ts 补标记，免重跑整轮验证。
    重跑检测会重建记录（clip_marks 随之失效，新检测本来就要重新验证）。
    找不到记录/无可写标记返回 False。
    """
    marks = {}
    for c in clips:
        m = {k: c[k] for k in ("score", "mark", "mark_source") if k in c}
        if m:
            marks[str(round(float(c["ts"]), 3))] = m
    if not marks:
        return False
    try:
        records = load_history()
    except OSError:
        return False
    hit_idx = _find_history_record(records, video_path)
    if hit_idx is None:
        return False
    target = records[hit_idx]
    merged = dict(target.get("clip_marks") or {})
    merged.update(marks)
    target["clip_marks"] = merged
    return save_history(records)


# ============ 片段缓存 ============
def clip_cache_key(video_path: str, goals) -> tuple:
    """构造片段缓存 key：进球时间戳排序 + 保留 3 位小数。

    写入方（检测完成，goals 已 sorted）与读取方（历史回读，原序）必须
    用同一规则生成 key，否则顺序不同会导致缓存永不命中。
    """
    return (video_path, tuple(sorted(round(float(t), 3) for t in goals)))


def put_clip_cache(key, clips):
    """写入片段缓存条目 + 超限驱逐最旧（同步删除其磁盘片段）+ 落盘。"""
    clip_cache[key] = list(clips)
    if len(clip_cache) > CLIP_CACHE_MAX_ENTRIES:
        evicted = clip_cache.pop(next(iter(clip_cache)))
        # 驱逐时同步删除磁盘片段文件：旧实现只弹内存条目，
        # mp4 靠 7 天兜底清理，重度使用短期可堆积数 GB
        for c in evicted:
            try:
                p = c.get("path", "")
                if p and os.path.exists(p) and "-highlights" not in os.path.basename(p):
                    os.remove(p)
            except OSError:
                pass
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
                clips = []
                for c in item.get("clips", []):
                    d = {"ts": float(c["ts"]), "path": c["path"], "idx": int(c["idx"])}
                    # verify 回写的分数/标记（旧格式无这些键 → 按未验证处理）
                    for k in ("score", "mark", "mark_source"):
                        if k in c:
                            d[k] = c[k]
                    clips.append(d)
                if clips and all(os.path.exists(c["path"]) for c in clips):
                    cache[key] = clips
    except json.JSONDecodeError as e:
        # 损坏至少留一条日志（旧实现静默 pass，缓存消失无从归因）
        logging.getLogger("state").warning(f"[WARN] 片段缓存解析失败，已忽略: {e}")
    except OSError as e:
        logging.getLogger("state").warning(f"[WARN] 片段缓存读取失败，已忽略: {e}")
    except (KeyError, TypeError, ValueError) as e:
        logging.getLogger("state").warning(f"[WARN] 片段缓存结构异常，已忽略: {e}")
    return cache


def save_clip_cache():
    """把内存片段缓存索引写入磁盘（原子写）。"""
    try:
        data = []
        for (v, g), clips in clip_cache.items():
            out_clips = []
            for c in clips:
                d = {"ts": c["ts"], "path": c["path"], "idx": c["idx"]}
                for k in ("score", "mark", "mark_source"):
                    if k in c:
                        d[k] = c[k]
                out_clips.append(d)
            data.append({"video": v, "goals": list(g), "clips": out_clips})
        os.makedirs(os.path.dirname(CLIP_CACHE_FILE), exist_ok=True)
        _atomic_write_json(CLIP_CACHE_FILE, data)
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] 保存片段缓存失败: {e}")


def update_clip_cache_marks(video_path, clips):
    """verify 完成后把分数/标记回写该视频的缓存条目（按 ts 匹配，就地改并落盘）。

    缓存条目在检测成功时写入（此时还没打分）；不回写的话，历史/缓存回读
    拿到的是无分数副本 → needs_verify 触发整轮重验（~5 分钟/视频）。
    返回是否有条目被更新。
    """
    by_ts = {round(float(c["ts"]), 3): c for c in clips}
    hit = False
    for (v, _g), cached in clip_cache.items():
        if v != video_path:
            continue
        for cc in cached:
            src = by_ts.get(round(float(cc["ts"]), 3))
            if src is None:
                continue
            for k in ("score", "mark", "mark_source"):
                if k in src:
                    cc[k] = src[k]
            hit = True
    if hit:
        save_clip_cache()
    return hit


def init_clip_cache():
    """启动时调用，从磁盘恢复片段缓存到内存。"""
    clip_cache.clear()
    clip_cache.update(load_clip_cache())


# ============ 检测断点（断点续识别） ============
# 结构: {video_path: {"frame": 已处理到的下一帧号, "goals": [已检出进球 ts],
#                     "params": 本次检测参数快照, "time": 写盘时间}}
# 同视频同参数才可续跑；检测成功/换参数重跑时清除。

def save_checkpoint(video_path, frame, goals, params):
    """写检测断点（原子写，失败仅记日志不影响检测主流程）。"""
    try:
        data = {}
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
            except (json.JSONDecodeError, OSError):
                data = {}
        data[video_path] = {
            "frame": int(frame),
            "goals": [round(float(t), 3) for t in goals],
            "params": dict(params),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        _atomic_write_json(CHECKPOINT_FILE, data, indent=2)
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] 断点写入失败（不影响检测）: {e}")


def load_checkpoint(video_path, params):
    """读取断点。参数一致（含标定/帧区间）才返回断点 dict，否则 None。"""
    try:
        if not os.path.exists(CHECKPOINT_FILE):
            return None
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cp = data.get(video_path)
        if not isinstance(cp, dict):
            return None
        saved = cp.get("params") or {}
        # 逐项严格比对：参数变了旧断点不可用（阈值/帧区间/标定不同 → 结果不一致）
        for k, v in (params or {}).items():
            if saved.get(k) != v:
                return None
        if int(cp.get("frame", 0)) <= 0:
            return None
        return cp
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return None


def clear_checkpoint(video_path=None):
    """清除断点。video_path=None 清全部；检测成功或用户放弃续跑时调用。"""
    try:
        if not os.path.exists(CHECKPOINT_FILE):
            return
        if video_path is None:
            os.remove(CHECKPOINT_FILE)
            return
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if video_path in data:
            del data[video_path]
            _atomic_write_json(CHECKPOINT_FILE, data, indent=2)
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] 断点清理失败: {e}")


def has_checkpoint(video_path):
    """是否存在该视频的断点（用于 UI 提示，不校验参数）。"""
    try:
        if not os.path.exists(CHECKPOINT_FILE):
            return False
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return video_path in (json.load(f) or {})
    except (json.JSONDecodeError, OSError):
        return False
