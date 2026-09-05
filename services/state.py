"""全局状态与持久化（历史记录、片段缓存）。

所有运行时状态集中在此模块，业务函数和 UI 层通过 `from services import state`
访问/修改状态，避免全局变量散落各处。
"""
import os
import re
import json
import hashlib
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

# 历史记录目录：每个视频一个独立 JSON 文件（<safe_basename>__<md5_8>.json）
HISTORY_DIR = os.path.join(CACHE_ROOT, "history")
# 旧版单文件历史记录路径（迁移用）
HISTORY_FILE = os.path.join(CACHE_ROOT, "detection_history.json")
# 断点续识别目录：每个视频+参数组合一个 checkpoint 文件
CHECKPOINT_DIR = os.path.join(CACHE_ROOT, "checkpoints")
# 预览片段缓存索引（持久化，重启后可复用上一次生成的片段）
CLIP_CACHE_FILE = os.path.join(CACHE_ROOT, "clip_cache.json")
# 全局人物名单（跨视频复用）：人物分类时曾使用过的名字，最近使用在前
PERSONS_FILE = os.path.join(CACHE_ROOT, "persons.json")

DEFAULT_VIDEO = ""
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ts"}

# 历史记录不再设上限：每个视频独立文件，互不影响
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
def _safe_filename_component(name: str) -> str:
    """将文件名中的路径不安全字符替换为下划线。"""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)


def _history_file_for(video_path: str) -> str:
    """根据视频路径生成独立历史记录文件路径。

    格式: <safe_basename>__<md5_8>.json
    - safe_basename: 视频文件名（不安全字符替换为下划线）
    - md5_8: 视频完整路径的 MD5 前 8 位（区分不同目录下同名视频）
    """
    base = os.path.basename(video_path) or "unknown"
    safe = _safe_filename_component(base)
    h = hashlib.md5(video_path.encode("utf-8")).hexdigest()[:8]
    return os.path.join(HISTORY_DIR, f"{safe}__{h}.json")


# ============ 断点续识别 ============
# 影响检测结果的关键参数（用于 checkpoint 指纹，参数变化则 checkpoint 失效）
_CHECKPOINT_PARAM_KEYS = (
    "hoop", "baseline_idx", "fps",
    "start_frame", "end_frame",
    "ball_conf", "min_gap_sec", "diff_threshold",
    "min_circularity", "min_in_hoop_frames", "min_blob_area", "search_margin",
    "auto_threshold", "yolo_step", "skip_yolo_no_motion",
)


def _checkpoint_params_fingerprint(params: dict) -> str:
    """计算检测参数的指纹（用于隔离不同参数下的 checkpoint）。

    只纳入影响检测结果的关键参数；UI 展示性参数不计入。
    """
    items = []
    for k in _CHECKPOINT_PARAM_KEYS:
        v = params.get(k)
        if isinstance(v, (list, tuple)):
            v = tuple(v)  # list 不可 hash，转 tuple
        items.append((k, v))
    raw = repr(items)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _checkpoint_file_for(video_path: str, params: dict = None) -> str:
    """根据视频路径 + 参数指纹生成 checkpoint 文件路径。

    格式: <safe_basename>__<video_md5_8>__<params_fp_12>.json
    - video_md5_8: 区分不同目录下同名视频
    - params_fp_12: 区分不同检测参数（参数变化则 checkpoint 失效）
    """
    base = os.path.basename(video_path) or "unknown"
    safe = _safe_filename_component(base)
    vh = hashlib.md5(video_path.encode("utf-8")).hexdigest()[:8]
    if params:
        ph = _checkpoint_params_fingerprint(params)
    else:
        ph = "any"
    return os.path.join(CHECKPOINT_DIR, f"{safe}__{vh}__{ph}.json")


def save_checkpoint(video_path: str, params: dict, current_frame: int,
                    detector_state: dict, extra: dict = None) -> bool:
    """保存断点续识别 checkpoint。

    Args:
        video_path: 视频完整路径
        params: 检测参数字典（用于指纹 + 记录）
        current_frame: 下一帧帧号（= 已处理完的最后一帧 + 1，恢复时从此帧继续）
        detector_state: GoalDetector.get_state() 的返回值
        extra: 附加信息（如统计计数器、开始时间等）
    Returns:
        True 保存成功，False 失败（不抛出异常，避免打断检测）
    """
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        data = {
            "video_path": video_path,
            "params": params,
            "current_frame": int(current_frame),
            "detector_state": detector_state,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "saved_ts": time.time(),
        }
        if extra:
            data["extra"] = extra
        path = _checkpoint_file_for(video_path, params)
        _atomic_write_json(path, data, indent=None)
        return True
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] checkpoint 保存失败: {e}")
        return False


def load_checkpoint(video_path: str, params: dict = None) -> dict | None:
    """加载断点续识别 checkpoint。

    - params 非空: 只加载参数指纹匹配的 checkpoint（参数变化则视为无断点）
    - params 为空: 加载该视频最新的 checkpoint（用于 UI 提示"是否继续"）

    Returns:
        checkpoint 字典或 None
    """
    try:
        if not os.path.isdir(CHECKPOINT_DIR):
            return None
        if params is not None:
            path = _checkpoint_file_for(video_path, params)
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        # params=None: 找该视频所有 checkpoint，返回最新的
        vh = hashlib.md5(video_path.encode("utf-8")).hexdigest()[:8]
        best = None
        best_ts = 0
        for fname in os.listdir(CHECKPOINT_DIR):
            if not fname.endswith(".json"):
                continue
            # 文件名格式: <safe>__<vh>__<ph>.json
            parts = fname.rsplit("__", 2)
            if len(parts) == 3 and parts[1] == vh:
                fpath = os.path.join(CHECKPOINT_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        cp = json.load(f)
                    ts = cp.get("saved_ts", 0)
                    if ts > best_ts:
                        best_ts = ts
                        best = cp
                except Exception:
                    continue
        return best
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] checkpoint 加载失败: {e}")
        return None


def delete_checkpoint(video_path: str, params: dict = None) -> None:
    """删除 checkpoint。

    - params 非空: 删除参数指纹匹配的 checkpoint
    - params 为空: 删除该视频所有 checkpoint（检测成功后清理）
    """
    try:
        if not os.path.isdir(CHECKPOINT_DIR):
            return
        if params is not None:
            path = _checkpoint_file_for(video_path, params)
            if os.path.exists(path):
                os.remove(path)
            return
        vh = hashlib.md5(video_path.encode("utf-8")).hexdigest()[:8]
        for fname in os.listdir(CHECKPOINT_DIR):
            if not fname.endswith(".json"):
                continue
            parts = fname.rsplit("__", 2)
            if len(parts) == 3 and parts[1] == vh:
                try:
                    os.remove(os.path.join(CHECKPOINT_DIR, fname))
                except OSError:
                    pass
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] checkpoint 删除失败: {e}")


def has_checkpoint(video_path: str) -> bool:
    """检查某视频是否存在任何 checkpoint（用于 UI 提示）。"""
    if not os.path.isdir(CHECKPOINT_DIR):
        return False
    vh = hashlib.md5(video_path.encode("utf-8")).hexdigest()[:8]
    for fname in os.listdir(CHECKPOINT_DIR):
        if not fname.endswith(".json"):
            continue
        parts = fname.rsplit("__", 2)
        if len(parts) == 3 and parts[1] == vh:
            return True
    return False


def _migrate_old_history() -> None:
    """若存在旧版单文件 detection_history.json，拆分为按视频存储的独立文件。

    迁移成功后将旧文件重命名为 .migrated-<时间戳>.bak（不直接删除，便于回滚）。
    """
    if not os.path.exists(HISTORY_FILE):
        return
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
        os.makedirs(HISTORY_DIR, exist_ok=True)
        for r in records:
            video = r.get("video")
            if not video:
                continue
            _atomic_write_json(_history_file_for(video), r, indent=2)
        bak = f"{HISTORY_FILE}.migrated-{time.strftime('%Y%m%d%H%M%S')}.bak"
        os.replace(HISTORY_FILE, bak)
        logging.getLogger("state").info(
            f"[MIGRATE] 旧历史记录已拆分为按视频存储（{len(records)} 条），原文件备份为 {bak}")
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] 历史记录迁移失败: {e}")


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


# ============ 全局人物名单（跨视频复用） ============

def load_persons() -> list:
    """读取全局人物名单（最近使用在前）。

    文件不存在/损坏时返回 []（损坏只告警不清空，下 add_person 成功后自愈）。
    """
    try:
        if os.path.exists(PERSONS_FILE):
            with open(PERSONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(n) for n in data if n]
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] 读取人物名单失败: {e}")
    return []


def add_person(name) -> bool:
    """登记人物到全局名单（跨视频复用）。

    已存在 → 提到队首（最近使用优先，对话框里排在前面）；
    新人物 → 插入队首。空名/写入失败返回 False。
    """
    name = (str(name) or "").strip()
    if not name:
        return False
    try:
        persons = load_persons()
        if name in persons:
            persons.remove(name)
        persons.insert(0, name)
        _atomic_write_json(PERSONS_FILE, persons, indent=2)
        return True
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] 保存人物名单失败: {e}")
        return False


def harvest_persons_from_history() -> int:
    """从所有历史记录的 labels.persons 存量收集人物名，合并进全局名单（幂等）。

    背景：全局名单（persons.json）上线前，人物分类只写进单视频 labels.persons、
    未登记全局名单 —— 旧场次建过的人物在新场次的对话框里不可选。这里一次性
    把存量名字补收进名单；已存在的跳过（不改变现有最近使用顺序），新收录的
    追加在名单末尾。返回新收录的名字数。
    """
    try:
        records = load_history()
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] 人物名单存量回填读取历史失败: {e}")
        return 0
    names = set()
    for r in records:
        lab = r.get("labels") or {}
        for v in (lab.get("persons") or {}).values():
            v = (str(v) or "").strip()
            if v:
                names.add(v)
    if not names:
        return 0
    persons = load_persons()
    new_names = sorted(n for n in names if n not in set(persons))
    if not new_names:
        return 0
    try:
        _atomic_write_json(PERSONS_FILE, persons + new_names, indent=2)
        logging.getLogger("state").info(
            f"[PERSONS] 从历史记录存量回填 {len(new_names)} 个人物: {new_names}")
        return len(new_names)
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] 人物名单存量回填写盘失败: {e}")
        return 0


def get_record(video_path: str):
    """读取单个视频的历史记录（只读该视频对应的 JSON 文件，比 load_history 轻）。

    返回记录 dict / None（无记录或文件损坏）。
    """
    fpath = _history_file_for(video_path)
    try:
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] 读取 {video_path} 历史记录失败: {e}")
    return None


def load_history() -> list:
    """加载所有历史记录列表（按检测时间倒序）。

    每个视频独立存储为 cache/history/<name>__<hash>.json。
    损坏的单个文件会被备份为 .corrupt-<时间戳>.bak，不影响其他视频的记录。
    """
    _migrate_old_history()
    if not os.path.isdir(HISTORY_DIR):
        return []
    records = []
    for fname in os.listdir(HISTORY_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(HISTORY_DIR, fname)
        last_err = None
        for attempt in range(3):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    r = json.load(f)
                records.append(r)
                break
            except json.JSONDecodeError as e:
                bak = f"{fpath}.corrupt-{time.strftime('%Y%m%d%H%M%S')}.bak"
                try:
                    os.replace(fpath, bak)
                except OSError:
                    pass
                logging.getLogger("state").warning(
                    f"[WARN] 历史记录 {fname} 解析失败（{e}），已备份到 {bak}")
                break
            except UnicodeDecodeError as e:
                bak = f"{fpath}.corrupt-{time.strftime('%Y%m%d%H%M%S')}.bak"
                try:
                    os.replace(fpath, bak)
                except OSError:
                    pass
                logging.getLogger("state").warning(
                    f"[WARN] 历史记录 {fname} 编码错误（{e}），已备份到 {bak}")
                break
            except OSError as e:
                last_err = e
                time.sleep(0.05 * (attempt + 1))
        else:
            # 三次 IO 重试都失败：跳过该文件，不影响其他记录
            logging.getLogger("state").warning(
                f"[WARN] 历史记录 {fname} 读取失败（{last_err}），已跳过")
    # 按检测时间倒序：优先用高精度 timestamp（旧记录无此字段则从 time 字符串解析，解析失败用 0）
    def _sort_key(r):
        ts = r.get("timestamp")
        if ts is not None:
            return float(ts)
        # 旧记录只有 time 字符串，解析为时间戳用于排序
        tstr = r.get("time", "")
        try:
            return time.mktime(time.strptime(tstr, "%Y-%m-%d %H:%M:%S"))
        except Exception:
            return 0.0
    records.sort(key=_sort_key, reverse=True)
    return records


def save_history(records) -> bool:
    """保存历史记录：每条记录写入对应视频的独立文件。返回是否成功。

    注意：不做"删除列表外旧文件"的清理——当前没有任何移除历史记录的功能，
    该清理唯一实际触发的场景是 load_history 因瞬态 IO 错误（Windows 文件
    共享冲突）3 次重试仍失败而跳过了某个文件，此时清理会把这条记录
    永久删除（数据丢失）。跳过的文件保留在磁盘上，下次自然恢复。
    """
    try:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        for r in records:
            video = r.get("video")
            if not video:
                continue
            _atomic_write_json(_history_file_for(video), r, indent=2)
        return True
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] 保存历史记录失败: {e}")
        return False


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


def update_history_labels(video_path, kept_ts_list, deleted_ts_list, person_map=None):
    """对已有历史记录打/更新人工确认标签（增量写，不重建整条记录，不会丢检测元信息）。

    √ 确认 → kept_ts_list（正样本），× 误报 → deleted_ts_list（负样本）。
    person_map: {进球ts: 人物名} 增量合并进 labels["persons"]；值为 "" 清除该 ts
    的分类；None 表示本次不改人物分类。
    找不到对应记录时返回 False；写入磁盘成功返回 True。
    """
    try:
        records = load_history()
    except Exception as e:
        logging.getLogger("state").warning(
            f"[WARN] update_history_labels 读取历史失败，标记未保存: {e}")
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
    if person_map is not None:
        # JSON 对象键只能是字符串：存 str(round(ts,3))，读取方（get_labels）转回 float
        persons = {str(k): v for k, v in (labels.get("persons") or {}).items()
                   if v not in (None, "")}
        for ts, name in person_map.items():
            key = str(round(float(ts), 3))
            if name:
                persons[key] = str(name)
            else:
                persons.pop(key, None)
        labels["persons"] = persons
    labels["label_time"] = now
    target["labels"] = labels
    # 保留记录的"检测时间"（time/timestamp）不动：旧实现覆盖成标记时间会让
    # 历史列表按标记时间排序/显示（刚标记的旧记录浮到顶部），"检测时间"
    # 字段语义混淆。标记时间已单独记录在 labels.label_time
    return save_history(records)


def get_labels(video_path):
    """读取某视频的已保存标签。

    返回 {"kept": list|None, "deleted": list|None, "label_time": str|None,
          "persons": {float ts: 人物名}}。
    找不到记录 / 无标签时返回 None 字段。

    瞬态 IO 错误（Windows 文件共享冲突）时额外重试：load_history 内置 3 次
    短重试仍可能不足（文件被 update_history_labels 的原子写占用），此处再补
    2 次较长等待，避免标签读空导致标记无法回填。
    """
    records = None
    last_err = None
    for attempt in range(2):
        try:
            records = load_history()
            break
        except OSError as e:
            last_err = e
            time.sleep(0.1 * (attempt + 1))
    if records is None:
        logging.getLogger("state").warning(
            f"[WARN] get_labels 读取历史失败（{last_err}），本次不回填标记")
        return {"kept": None, "deleted": None, "label_time": None, "persons": {}}
    hit_idx = _find_history_record(records, video_path)
    if hit_idx is None:
        return {"kept": None, "deleted": None, "label_time": None, "persons": {}}
    lab = records[hit_idx].get("labels") or {}
    # persons 键是 JSON 字符串（存盘时 str(round(ts,3))），转回 float 供精确匹配
    persons = {}
    for k, v in (lab.get("persons") or {}).items():
        try:
            persons[round(float(k), 3)] = str(v)
        except (TypeError, ValueError):
            continue
    return {"kept": list(lab["kept"]) if "kept" in lab else None,
            "deleted": list(lab["deleted"]) if "deleted" in lab else None,
            "label_time": lab.get("label_time"),
            "persons": persons}


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


# 标签时间戳重匹配容差（秒）：改参数重跑后进球时间戳整体小幅偏移，
# ±容差内的旧标签 ts 视为同一球
_LABEL_REMAP_TOL_SEC = 0.5
# 旧标签与新进球的匹配率低于该值时，判定两次检测结果差异过大，丢弃旧标签
_LABEL_MIN_MATCH_RATE = 0.5


def _remap_labels_to_goals(labels: dict, new_goals) -> tuple:
    """把旧人工标签的时间戳近似匹配到新一次检测的进球时间戳。

    重新检测（尤其改参数）后进球时间戳会整体偏移（±0.2s 以上），直接沿用
    旧 ts 会在加载历史时精确匹配失败 → 时间戳偏移的进球全部被排除出默认
    集锦且不显示任何标记，用户需手动逐个重新确认。±tol 内的旧 ts 重映射
    为新 ts（贪心最近邻：距离近的旧 ts 优先占用新球，避免远者抢占）。

    Returns:
        (remapped_labels, matched, total)：
        - remapped_labels: kept/deleted 已换成新 ts 的 labels（原样保留其他键）
        - matched/total: 匹配到新进球的旧标签数 / 旧标签总数（调用方按匹配率决策）
    """
    kept = [float(t) for t in (labels.get("kept") or [])]
    deleted = [float(t) for t in (labels.get("deleted") or [])]
    persons = {}
    for k, v in (labels.get("persons") or {}).items():
        try:
            persons[round(float(k), 3)] = str(v)
        except (TypeError, ValueError):
            continue
    # 分母 = 去重后的时间戳数（用户常对同一进球同时 √ + 人物分类，
    # 若 kept/deleted 与 persons 各自计数会把同一 ts 计两次，total 虚高、
    # 真实匹配率被低估到阈值以下 → 一个进球消失就整体丢弃全部标签）
    total = len(set(kept) | set(deleted) | set(persons))
    if total == 0:
        # 没有任何时间戳标签（如只有 label_time）：原样保留
        return dict(labels), 0, 0
    goals = sorted(float(g) for g in new_goals)
    # 每个旧 ts 找最近的新球；按距离排序后贪心占位
    pairs = []
    for ot in sorted(set(kept) | set(deleted) | set(persons)):
        best_g, best_d = None, None
        for g in goals:
            d = abs(g - ot)
            if best_d is None or d < best_d:
                best_g, best_d = g, d
        if best_g is not None and best_d <= _LABEL_REMAP_TOL_SEC:
            pairs.append((best_d, ot, best_g))
    pairs.sort()
    used = set()
    mapping = {}
    for _, ot, g in pairs:
        if g in used:
            continue
        used.add(g)
        mapping[ot] = g
    new_labels = dict(labels)
    new_labels["kept"] = sorted(round(mapping[ot], 3) for ot in kept if ot in mapping)
    new_labels["deleted"] = sorted(round(mapping[ot], 3) for ot in deleted if ot in mapping)
    # 人物分类同样重映射到新 ts（键统一转 str 与存盘格式一致）
    new_labels["persons"] = {str(round(mapping[ot], 3)): name
                             for ot, name in persons.items() if ot in mapping}
    return new_labels, len(mapping), total


def add_history(video_path, hoop, goals, baseline_idx=-1,
                diff_threshold=None, **fields):
    """添加一条历史记录（同视频会覆盖旧记录）。

    可选字段见 _HISTORY_FIELD_CASTS；未知字段抛 TypeError（防调用方拼错字段名静默丢数据）。
    返回: 新记录 dict；读取/写入失败（IO 错误）时返回 None 且**不覆盖**磁盘旧数据。
    """
    unknown = set(fields) - set(_HISTORY_FIELD_CASTS)
    if unknown:
        raise TypeError(f"add_history 收到未知字段: {sorted(unknown)}")
    try:
        records = load_history()
    except Exception as e:
        logging.getLogger("state").warning(
            f"[WARN] 历史记录读取失败，本次结果未写入（避免空列表覆盖旧数据）: {e}")
        return None
    # 保留旧记录的人工标记（kept/deleted）：重新检测不应丢失已做的标记。
    # 但旧 ts 必须先按容差重映射到本次检测的进球时间戳——改参数重跑后进球
    # 时间戳会偏移，直接沿用旧 ts 会导致加载历史时精确匹配失败，被标记过的
    # 球掉出默认集锦；匹配率过低说明两次检测结果差异过大，旧标签已无意义。
    old_labels = None
    for r in records:
        if r.get("video") == video_path and r.get("labels"):
            old_labels = r.get("labels")
            break
    records = [r for r in records if r.get("video") != video_path]
    rec = {
        "video": video_path,
        "video_name": os.path.basename(video_path),
        "hoop": list(hoop) if hoop else None,
        "baseline_idx": int(baseline_idx),
        "goals": [float(t) for t in goals],
        "total": len(goals),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.time(),
    }
    if old_labels is not None:
        remapped, matched, total = _remap_labels_to_goals(old_labels, goals)
        if total == 0 or matched >= total * _LABEL_MIN_MATCH_RATE:
            rec["labels"] = remapped
            if 0 < matched < total:
                logging.getLogger("state").warning(
                    f"[WARN] 历史标记部分未匹配上新进球（{matched}/{total}），"
                    f"未匹配的标记已丢弃: {video_path}")
        else:
            logging.getLogger("state").warning(
                f"[WARN] 历史标记与新检测结果匹配率过低（{matched}/{total}），"
                f"已丢弃旧标记: {video_path}")
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
    # 不再限制历史记录条数：每个视频独立文件，互不影响
    if not save_history(records):
        return None
    return rec


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
                clips = [{"ts": float(c["ts"]), "path": c["path"], "idx": int(c["idx"])}
                         for c in item.get("clips", [])]
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
        data = [{"video": v, "goals": list(g),
                 "clips": [{"ts": c["ts"], "path": c["path"], "idx": c["idx"]} for c in clips]}
                for (v, g), clips in clip_cache.items()]
        os.makedirs(os.path.dirname(CLIP_CACHE_FILE), exist_ok=True)
        _atomic_write_json(CLIP_CACHE_FILE, data)
    except Exception as e:
        logging.getLogger("state").warning(f"[WARN] 保存片段缓存失败: {e}")


def init_clip_cache():
    """启动时调用，从磁盘恢复片段缓存到内存。"""
    clip_cache.clear()
    clip_cache.update(load_clip_cache())
