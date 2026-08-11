"""篮球进球检测与自动剪辑交互式 Demo（NiceGUI）。

功能：
  1. 输入视频文件路径 → 加载
  2. 滑动到含篮筐的帧，点击画面 2 个点标定篮筐
  3. 设置起止帧、置信度、最小进球间隔
  4. 点击「开始检测」→ diff + YOLO 双确认检测进球
  5. 每个进球生成独立预览片段，人工确认保留/删除
  6. 生成集锦视频（GPU 硬编加速）
  7. 历史记录支持加载后直接剪辑（无需重新检测）

用法:
    python demo_nicegui.py
浏览器打开 http://127.0.0.1:7871
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# 缓存目录：优先用环境变量，否则按平台默认
if os.environ.get("BBALL_CACHE_ROOT"):
    _CACHE_ROOT = os.environ["BBALL_CACHE_ROOT"]
elif os.name == "nt":
    _CACHE_ROOT = r"E:\basketball-project\cache"
else:
    _CACHE_ROOT = os.path.join(os.path.expanduser("~"), "basketball-project", "cache")

# Windows subprocess 屏蔽控制台窗口（Linux 下为 0）
_SBOX = 0x08000000 if os.name == "nt" else 0
# 临时文件重定向到缓存目录
_tmp_root = os.path.join(_CACHE_ROOT, "tmp")
os.makedirs(_tmp_root, exist_ok=True)
os.environ["TMPDIR"] = _tmp_root
os.environ["TEMP"] = _tmp_root
os.environ["TMP"] = _tmp_root

import time
import json
import cv2
import numpy as np
import base64

from nicegui import ui

from video_io import get_video_info, read_frame, VideoReader
from app import get_ball_model, get_device
from tracker import GoalDetector
from cutter.ffmpeg_cutter import cut_clips, _build_encode_args

# 历史记录文件
_HISTORY_FILE = os.path.join(_CACHE_ROOT, "detection_history.json")
# 预览片段缓存索引（持久化，重启后可复用上一次生成的片段）
_CLIP_CACHE_FILE = os.path.join(_CACHE_ROOT, "clip_cache.json")


def _load_clip_cache():
    """从磁盘加载片段缓存索引，仅保留片段文件仍存在的条目。"""
    cache = {}
    try:
        if os.path.exists(_CLIP_CACHE_FILE):
            with open(_CLIP_CACHE_FILE, "r", encoding="utf-8") as f:
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


def _save_clip_cache():
    """把内存片段缓存索引写入磁盘。"""
    try:
        data = [{"video": v, "goals": list(g),
                 "clips": [{"ts": c["ts"], "path": c["path"], "idx": c["idx"]} for c in clips]}
                for (v, g), clips in _clip_cache.items()]
        os.makedirs(os.path.dirname(_CLIP_CACHE_FILE), exist_ok=True)
        with open(_CLIP_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] 保存片段缓存失败: {e}", flush=True)


# ============ 全局状态 ============
_video_state = {"path": None, "total": 0, "fps": 30.0, "codec": "unknown",
                "current_frame": 0, "width": 0, "height": 0}
_calib = {
    "clicks": [],
    "hoop": None,
    "baseline_frame": None,
    "baseline_idx": -1,
}
_last_goals = []
_last_goal_clips = []
_kept_goal_indices = set()

# 检测取消标志（UI 点击「取消」时置 True，检测循环轮询后中断）
_cancel_requested = False
# 历史加载预览片段缓存：key=(视频路径, 进球时间戳元组) -> [片段dict, ...]，启动时从磁盘恢复
_clip_cache = _load_clip_cache()

_DEFAULT_VIDEO = r"D:\Downloads\highlights.mp4"
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ts"}

# 文件夹批量模式状态
_batch_files = []
_batch_calibs = {}
_batch_current_video = None

# ============ 历史记录 ============

def _load_history():
    try:
        if os.path.exists(_HISTORY_FILE):
            with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_history(records):
    try:
        os.makedirs(os.path.dirname(_HISTORY_FILE), exist_ok=True)
        with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 保存历史记录失败: {e}", flush=True)


def _add_history(video_path, hoop, goals, kept_goals):
    records = _load_history()
    records = [r for r in records if r.get("video") != video_path]
    records.insert(0, {
        "video": video_path,
        "video_name": os.path.basename(video_path),
        "hoop": list(hoop) if hoop else None,
        "goals": [float(t) for t in goals],
        "kept_goals": [float(t) for t in kept_goals],
        "total": len(goals),
        "kept": len(kept_goals),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    records = records[:50]
    _save_history(records)


# ============ 视频工具 ============

def _frame_to_base64(frame):
    """将 cv2 帧转为 base64 PNG data URI。

    输入约定为 RGB（与 load_video / preview_frame / click_calibrate
    等返回值一致），而 cv2.imencode 按 BGR 处理，
    因此先转回 BGR 再编码，避免预览画面红蓝通道互换。
    """
    if frame is None:
        return None
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.png', frame)
    b64 = base64.b64encode(buf).decode('utf-8')
    return f'data:image/png;base64,{b64}'


def _scan_video_files(folder):
    """扫描文件夹内的视频文件，按自然顺序排序。"""
    folder = folder.strip().strip('"').strip("'")
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in _VIDEO_EXTS:
            files.append(os.path.join(folder, name))
    import re
    def _natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
    files.sort(key=_natural_key)
    return files


# ============ 核心函数 ============

def load_video(video_path):
    """加载视频，返回预览帧和信息字符串。"""
    if not video_path or not video_path.strip():
        return None, "请输入视频文件路径"
    video_path = video_path.strip().strip('"').strip("'")
    if not os.path.exists(video_path):
        return None, f"❌ 文件不存在: {video_path}"
    try:
        info = get_video_info(video_path)
    except Exception as e:
        return None, f"读取失败: {e}"
    _video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                        codec=info["codec"], current_frame=0,
                        width=info["width"], height=info["height"])
    _calib["clicks"] = []
    _calib["hoop"] = None
    _calib["baseline_frame"] = None
    _calib["baseline_idx"] = -1
    frame = read_frame(video_path, 0, total=info["total"], fps=info["fps"])
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")
    return preview, info_str


def preview_frame(frame_idx):
    """预览指定帧。"""
    if _video_state["path"] is None:
        return None, "请先加载视频"
    frame = read_frame(_video_state["path"], int(frame_idx),
                       total=_video_state["total"], fps=_video_state["fps"])
    if frame is None:
        return None, "读取帧失败"
    out = frame.copy()
    if _calib["hoop"]:
        x1, y1, x2, y2 = _calib["hoop"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(out, "HOOP", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.line(out, (x1 - 30, y1), (x2 + 30, y1), (0, 255, 255), 1)
        cv2.line(out, (x1 - 30, y2), (x2 + 30, y2), (255, 0, 255), 1)
    for i, (x, y) in enumerate(_calib["clicks"]):
        cv2.circle(out, (x, y), 10, (255, 255, 0), -1)
        cv2.putText(out, str(i + 1), (x - 6, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    ts = int(frame_idx) / _video_state["fps"]
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), f"帧 {frame_idx} ({ts:.1f}s)"


def click_calibrate(x, y):
    """点击标定篮筐。"""
    if _video_state["path"] is None:
        return None, "请先加载视频"
    frame_idx = _video_state["current_frame"]
    _calib["clicks"].append((x, y))
    status = f"点击 ({x},{y})，已收集 {len(_calib['clicks'])}/2 个点"
    if len(_calib["clicks"]) >= 2:
        p1, p2 = _calib["clicks"][:2]
        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
        _calib["hoop"] = (x1, y1, x2, y2)
        base_frame = read_frame(_video_state["path"], int(frame_idx),
                                total=_video_state["total"], fps=_video_state["fps"])
        if base_frame is not None:
            _calib["baseline_frame"] = base_frame.copy()
            _calib["baseline_idx"] = int(frame_idx)
        status = f"篮筐已标定: ({x1},{y1}) - ({x2},{y2}) | 基准帧: 第 {int(frame_idx)} 帧"
        _calib["clicks"] = []
    frame = read_frame(_video_state["path"], int(frame_idx),
                       total=_video_state["total"], fps=_video_state["fps"])
    if frame is None:
        return None, status
    out = frame.copy()
    if _calib["hoop"]:
        x1, y1, x2, y2 = _calib["hoop"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(out, "HOOP", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.line(out, (x1 - 30, y1), (x2 + 30, y1), (0, 255, 255), 1)
        cv2.line(out, (x1 - 30, y2), (x2 + 30, y2), (255, 0, 255), 1)
    for i, (px, py) in enumerate(_calib["clicks"]):
        cv2.circle(out, (px, py), 10, (255, 255, 0), -1)
        cv2.putText(out, str(i + 1), (px - 6, py - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), status


def reset_hoop():
    _calib["clicks"] = []
    _calib["hoop"] = None
    _calib["baseline_frame"] = None
    _calib["baseline_idx"] = -1
    return "已重置，请重新点击 2 个点标定篮筐"


def run_detect(start_frame, end_frame, ball_conf, min_gap_sec,
               diff_threshold=15, min_circularity=0.35, min_in_hoop_frames=2,
               min_blob_area=30, search_margin=80, progress_callback=None):
    """运行进球检测。返回 (结果文本, 是否成功)。"""
    global _kept_goal_indices, _last_goal_clips, _last_goals, _cancel_requested
    if _video_state["path"] is None:
        return "❌ 请先加载视频", False
    if _calib["hoop"] is None:
        return "❌ 请先点击画面标定篮筐", False
    if _calib["baseline_frame"] is None:
        return "❌ 基准帧差法需要基准帧，请重新标定", False

    def _report(pct, msg):
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    hoop = _calib["hoop"]
    fps = _video_state["fps"]
    total = _video_state["total"]
    start = int(start_frame)
    end = int(end_frame) if end_frame and int(end_frame) > 0 else total
    end = min(end, total)
    if end <= start:
        return "❌ 结束帧必须大于起始帧", False

    _report(5, '初始化检测器...')
    try:
        detector = GoalDetector(hoop, baseline_frame=_calib["baseline_frame"],
                                min_gap_sec=float(min_gap_sec),
                                diff_threshold=int(diff_threshold),
                                min_blob_area=int(min_blob_area),
                                search_margin=int(search_margin),
                                fusion_mode="visual_only", loose_mode=True,
                                yolo_confirm=True, rolling_baseline_sec=60.0,
                                min_circularity=float(min_circularity),
                                min_in_hoop_frames=int(min_in_hoop_frames))
        _report(10, '加载 YOLO 模型...')
        model, _ = get_ball_model()

        out_dir = Path(_CACHE_ROOT) / "demo_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        import time as _t
        _stamp = int(_t.time())

        t0 = time.time()
        processed = 0
        n_frames = end - start
        clip_half = int(fps * 3)

        _report(15, f'开始检测 {n_frames} 帧...')
        reader = VideoReader(_video_state["path"])
        try:
            for fidx, frame in reader.iter_frames(start=start, end=end, batch=1):
                if _cancel_requested:
                    break
                ball_pos = None
                try:
                    res = model.predict(frame, conf=float(ball_conf), imgsz=1280,
                                        device=get_device(), verbose=False)[0]
                    if res.boxes is not None and len(res.boxes) > 0:
                        names = res.names
                        clses = res.boxes.cls.cpu().numpy().astype(int)
                        xyxy = res.boxes.xyxy.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()
                        best = None
                        for j, c in enumerate(clses):
                            n = names.get(c, "").lower()
                            if "ball" in n or "basketball" in n:
                                if best is None or confs[j] > confs[best]:
                                    best = j
                        if best is not None:
                            x1, y1, x2, y2 = xyxy[best]
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            ball_pos = (float(cx), float(cy), float(x1), float(y1),
                                        float(x2), float(y2), float(confs[best]))
                except Exception:
                    pass
                detector.feed(ball_pos, fidx, fps, frame=frame)
                processed += 1
                # 每 10 帧更新一次进度
                if processed % 10 == 0:
                    pct = 15 + 60 * processed / n_frames
                    _report(pct, f'检测帧 {processed}/{n_frames} ({processed*100//n_frames}%)')
                # 定期释放 CUDA 缓存，降低 4GB 显存长视频检测时的峰值压力
                if processed % 100 == 0:
                    try:
                        import torch as _torch
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                    except Exception:
                        pass
        finally:
            reader.close()

        _report(80, '生成预览片段...')
        detector.finalize()
        elapsed = time.time() - t0

        # 生成预览片段
        goals = sorted(detector.goals)
        _last_goal_clips.clear()
        _kept_goal_indices.clear()

        import imageio_ffmpeg
        import subprocess as _sp
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        # 编码参数只检测一次（NVENC 探测是子进程，循环内重复调用会显著拖慢）
        _enc = _build_encode_args(ff, quality="preview")

        for gi, gts in enumerate(goals):
            if _cancel_requested:
                break
            gframe = int(gts * fps)
            seg_start = max(start, gframe - clip_half)
            seg_end = min(end, gframe + clip_half)
            if seg_end <= seg_start:
                continue
            clip_path = str(out_dir / f"goal_{gi}_{int(gts)}s_{_stamp}.mp4")
            seg_start_sec = seg_start / fps
            seg_dur_sec = (seg_end - seg_start) / fps
            try:
                _sp.run([ff, "-y", "-loglevel", "error",
                         "-ss", f"{seg_start_sec:.3f}", "-i", _video_state["path"],
                         "-t", f"{seg_dur_sec:.3f}",
                         "-vf", "scale=-2:480"] + _enc +
                        ["-movflags", "+faststart", clip_path],
                        creationflags=_SBOX, capture_output=True, timeout=60)
                if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                    _last_goal_clips.append({"ts": gts, "path": clip_path, "idx": gi})
            except Exception as e:
                print(f"[WARN] 预览片段生成失败 ({gts:.1f}s): {e}", flush=True)
            if len(goals) > 0:
                _report(80 + 18 * (gi + 1) / len(goals), f'生成片段 {gi+1}/{len(goals)}')

        if _cancel_requested:
            # 用户取消：清空本次部分生成的片段，不写入历史
            _last_goal_clips.clear()
            _kept_goal_indices.clear()
            _last_goals.clear()
            return f"已取消 | 已处理 {processed} 帧", False

        # 检测成功后写入片段缓存并持久化，之后加载同一视频历史可直接复用
        if _last_goal_clips:
            _ckey = (_video_state["path"], tuple(round(g, 3) for g in goals))
            _clip_cache[_ckey] = list(_last_goal_clips)
            if len(_clip_cache) > 20:
                _clip_cache.pop(next(iter(_clip_cache)))
            _save_clip_cache()

        _report(100, '完成！')
        _kept_goal_indices = set(range(len(_last_goal_clips)))
        _last_goals.clear()
        _last_goals.extend(detector.goals)

        d = detector.diag
        total_yolo = d['yolo_confirmed'] + d['yolo_rejected']
        confirm_rate = d['yolo_confirmed'] / max(total_yolo, 1) * 100

        status = (f"检测完成 | 处理 {processed} 帧 | 耗时 {elapsed:.0f}s\n"
                  f"进球: {len(detector.goals)} 个 | "
                  f"YOLO确认: {d['yolo_confirmed']}/{total_yolo} ({confirm_rate:.0f}%)")
        _add_history(_video_state["path"], hoop, detector.goals, detector.goals)
        return status, True
    except Exception as e:
        import traceback
        return f"❌ 检测失败: {e}\n{traceback.format_exc()}", False


def clip_action(action, idx):
    """处理卡片按钮操作。"""
    if idx < 0 or idx >= len(_last_goal_clips):
        return None, ""
    if action == "preview":
        return _last_goal_clips[idx]["path"], f"▶ 正在预览第 {idx+1} 个片段"
    elif action == "export":
        return _last_goal_clips[idx]["path"], f"已导出: {_last_goal_clips[idx]['path']}"
    elif action == "delete":
        global _kept_goal_indices, _last_goals
        ts = _last_goal_clips[idx]["ts"]
        del _last_goal_clips[idx]
        new_kept = set()
        for old_i in sorted(_kept_goal_indices):
            if old_i < idx:
                new_kept.add(old_i)
            elif old_i > idx:
                new_kept.add(old_i - 1)
        _kept_goal_indices = new_kept
        kept_ts = [_last_goal_clips[i]["ts"] for i in sorted(_kept_goal_indices) if i < len(_last_goal_clips)]
        _last_goals.clear()
        _last_goals.extend(kept_ts)
        return None, f"已删除第 {idx+1} 个片段（{ts:.1f}s）| 剩余 {len(_last_goal_clips)} 个"
    return None, ""


def generate_highlights(pre_roll, post_roll, min_gap, progress_callback=None):
    """生成集锦视频。"""

    def _report(pct, msg):
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    if _video_state["path"] is None:
        return None, "❌ 请先加载视频并检测进球"
    if not _last_goals:
        return None, "❌ 没有检测到进球"
    try:
        _report(5, '准备集锦素材...')
        out_path = cut_clips(_video_state["path"], list(_last_goals),
                             pre_roll=int(pre_roll), post_roll=int(post_roll),
                             min_gap=int(min_gap),
                             progress_callback=progress_callback)
        if out_path and os.path.exists(out_path):
            n = len(_last_goals)
            return out_path, f"集锦已生成（{n} 个进球片段）\n输出: {out_path}"
        return None, "❌ 集锦生成失败"
    except Exception as e:
        import traceback
        return None, f"❌ 剪辑失败: {e}\n{traceback.format_exc()}"


def on_load_history(idx_choice, progress_callback=None):
    """从历史记录加载。"""
    global _last_goals, _video_state, _calib, _last_goal_clips, _kept_goal_indices

    def _report(pct, msg):
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    _report(5, '读取历史记录...')
    if idx_choice is None:
        return None, "请先选择一条历史记录", ""
    records = _load_history()
    if idx_choice < 0 or idx_choice >= len(records):
        return None, "历史记录不存在", ""
    r = records[idx_choice]
    video_path = r.get("video", "")
    if not os.path.exists(video_path):
        return None, f"视频文件不存在: {video_path}", ""

    _report(15, '读取视频信息...')
    try:
        info = get_video_info(video_path)
    except Exception as e:
        return None, f"读取视频失败: {e}", ""
    _video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                        codec=info["codec"], current_frame=0,
                        width=info["width"], height=info["height"])
    hoop = r.get("hoop")
    if hoop and len(hoop) == 4:
        _calib["hoop"] = tuple(int(v) for v in hoop)
        _calib["clicks"] = []
        base_frame = read_frame(video_path, 0, total=info["total"], fps=info["fps"])
        if base_frame is not None:
            _calib["baseline_frame"] = base_frame.copy()
            _calib["baseline_idx"] = 0
    kept = [float(t) for t in r.get("kept_goals", [])]
    all_goals = [float(t) for t in r.get("goals", [])]
    _last_goals.clear()
    _last_goals.extend(kept if kept else all_goals)

    fps = info["fps"]
    total = info["total"]
    clip_half = int(fps * 3)
    out_dir = Path(_CACHE_ROOT) / "demo_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    import time as _t
    _stamp = int(_t.time())
    _last_goal_clips.clear()
    _kept_goal_indices.clear()

    _report(30, f'生成 {len(all_goals)} 个预览片段...')
    cache_key = (video_path, tuple(round(t, 3) for t in all_goals))
    cached = _clip_cache.get(cache_key)
    if cached and all(os.path.exists(c["path"]) for c in cached):
        # 命中缓存：直接复用已生成的片段，跳过 ffmpeg（进度保持 30，避免跳跃）
        _last_goal_clips.extend(list(cached))
        _report(30, f'命中缓存，复用 {len(_last_goal_clips)} 个片段')
    elif all_goals:
        import imageio_ffmpeg
        import subprocess as _sp
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        # 编码参数只检测一次，避免循环内重复子进程探测
        _enc = _build_encode_args(ff, quality="preview")
        for gi, gts in enumerate(all_goals):
            pct = 30 + 60 * (gi + 1) / len(all_goals) if all_goals else 90
            _report(pct, f'生成片段 {gi+1}/{len(all_goals)}')
            gframe = int(gts * fps)
            seg_start = max(0, gframe - clip_half)
            seg_end = min(total, gframe + clip_half)
            if seg_end <= seg_start:
                continue
            clip_path = str(out_dir / f"goal_{gi}_{int(gts)}s_{_stamp}.mp4")
            seg_start_sec = seg_start / fps
            seg_dur_sec = (seg_end - seg_start) / fps
            try:
                _sp.run([ff, "-y", "-loglevel", "error",
                         "-ss", f"{seg_start_sec:.3f}", "-i", video_path,
                         "-t", f"{seg_dur_sec:.3f}",
                         "-vf", "scale=-2:480"] + _enc +
                        ["-movflags", "+faststart", clip_path],
                        creationflags=_SBOX, capture_output=True, timeout=60)
                if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                    _last_goal_clips.append({"ts": float(gts), "path": clip_path, "idx": gi})
            except Exception as e:
                print(f"[WARN] 预览片段生成失败 ({gts:.1f}s): {e}", flush=True)
        # 写入缓存并持久化（保留最近 20 条，避免无限增长）
        if _last_goal_clips:
            _clip_cache[cache_key] = list(_last_goal_clips)
            if len(_clip_cache) > 20:
                _clip_cache.pop(next(iter(_clip_cache)))
            _save_clip_cache()

    kept_set = set(kept)
    for i, clip in enumerate(_last_goal_clips):
        if clip["ts"] in kept_set or not kept:
            _kept_goal_indices.add(i)

    frame = read_frame(video_path, 0, total=total, fps=fps)
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")
    status = (f"已加载历史记录\n视频: {r.get('video_name', '')}\n"
              f"进球: {len(all_goals)} 个 | 保留: {len(kept)} 个\n"
              f"已生成 {len(_last_goal_clips)} 个预览片段")
    return preview, info_str, status


# ============ 文件夹批量模式 ============

def on_batch_load_video(selected):
    """批量模式：加载选中的视频，并应用该视频已保存的标定。"""
    global _batch_current_video, _calib
    if not selected or not _batch_files:
        return None, "", "请先扫描文件夹并选择视频"
    video_path = selected
    _batch_current_video = video_path
    if video_path in _batch_calibs:
        cal = _batch_calibs[video_path]
        _calib["hoop"] = cal["hoop"]
        _calib["baseline_frame"] = cal["baseline_frame"]
        _calib["baseline_idx"] = cal["baseline_idx"]
        _calib["clicks"] = []
    else:
        _calib["hoop"] = None
        _calib["baseline_frame"] = None
        _calib["baseline_idx"] = -1
        _calib["clicks"] = []
    try:
        info = get_video_info(video_path)
    except Exception as e:
        return None, "", f"读取失败: {e}"
    _video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                        codec=info["codec"], current_frame=0,
                        width=info["width"], height=info["height"])
    frame = read_frame(video_path, 0, total=info["total"], fps=info["fps"])
    if frame is not None and _calib["hoop"]:
        x1, y1, x2, y2 = _calib["hoop"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, "HOOP", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")
    status = (f"已加载: {os.path.basename(video_path)}\n"
              f"{'已标定' if video_path in _batch_calibs else '未标定，请点击画面 2 个点标定'}")
    return preview, info_str, status


def on_batch_save_calib():
    """保存当前标定到当前批量视频。"""
    global _batch_calibs
    if _batch_current_video is None:
        return "请先从列表选择视频"
    if _calib["hoop"] is None or _calib["baseline_frame"] is None:
        return "请先标定篮筐"
    _batch_calibs[_batch_current_video] = {
        "hoop": _calib["hoop"],
        "baseline_frame": _calib["baseline_frame"].copy(),
        "baseline_idx": _calib["baseline_idx"],
    }
    n_calib = len(_batch_calibs)
    n_total = len(_batch_files)
    status = f"已保存: {os.path.basename(_batch_current_video)} | 已标定: {n_calib}/{n_total}"
    if n_calib >= n_total:
        status += "，全部标定完成，可点击「批量识别」"
    return status


def run_batch_detect(start_frame, end_frame, ball_conf, min_gap_sec,
                     diff_threshold=15, min_circularity=0.35, min_in_hoop_frames=2,
                     min_blob_area=30, search_margin=80, progress_callback=None):
    """批量识别：遍历文件夹内全部视频逐个检测，每个视频独立写入历史。

    返回 (状态文本, 是否成功)。状态文本逐条列出每个视频的结果，
    未标定/读取失败/检测失败都会单独说明，不再静默跳过。
    """
    global _video_state, _calib
    if not _batch_files:
        return "请先加载文件夹", False
    # 当前视频若已标定但未点「保存标定」，批量前自动保存，避免漏处理
    cur = _batch_current_video
    if (cur and cur in _batch_files and cur not in _batch_calibs
            and _calib["hoop"] is not None and _calib["baseline_frame"] is not None):
        _batch_calibs[cur] = {
            "hoop": _calib["hoop"],
            "baseline_frame": _calib["baseline_frame"].copy(),
            "baseline_idx": _calib["baseline_idx"],
        }
    lines = []
    n_ok = 0
    total_goals = 0
    n_total = len(_batch_files)
    cancelled = False
    for i, video_path in enumerate(_batch_files):
        if _cancel_requested:
            cancelled = True
            break
        name = os.path.basename(video_path)
        if video_path not in _batch_calibs:
            lines.append(f"✗ {name}: 未标定，跳过")
            continue
        cal = _batch_calibs[video_path]
        try:
            info = get_video_info(video_path)
        except Exception as e:
            lines.append(f"✗ {name}: 读取失败 ({e})")
            continue
        _video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                            codec=info["codec"], current_frame=0,
                            width=info["width"], height=info["height"])
        _calib["hoop"] = cal["hoop"]
        _calib["baseline_frame"] = cal["baseline_frame"]
        _calib["baseline_idx"] = cal["baseline_idx"]
        _calib["clicks"] = []

        def _cb(pct, msg, vname=name, idx=i):
            if progress_callback:
                try:
                    progress_callback(pct, f'[{idx+1}/{n_total}] {vname} · {msg}')
                except Exception:
                    pass

        try:
            _status, ok = run_detect(start_frame, end_frame, ball_conf, min_gap_sec,
                                     diff_threshold, min_circularity, min_in_hoop_frames,
                                     min_blob_area, search_margin, progress_callback=_cb)
        except Exception as e:
            _status, ok = f"异常: {e}", False
        if _cancel_requested:
            # 本视频检测中被取消：run_detect 已清理状态，停止后续视频
            cancelled = True
            reason = _status.splitlines()[0] if _status else "已取消"
            lines.append(f"⏹ {name}: {reason}")
            break
        if ok:
            n_ok += 1
            total_goals += len(_last_goals)
            lines.append(f"✓ {name}: 成功，{len(_last_goals)} 个进球")
        else:
            reason = _status.splitlines()[0] if _status else "失败"
            lines.append(f"✗ {name}: {reason}")
    if cancelled:
        msg = f"已取消 | 已完成 {n_ok}/{n_total} 个视频 | 共 {total_goals} 个进球"
    else:
        msg = f"批量识别完成: {n_ok}/{n_total} 个视频 | 共 {total_goals} 个进球"
    return msg + "\n" + "\n".join(lines), n_ok > 0


# ============ NiceGUI 界面 ============

# 全局 UI 引用（用于在回调中更新）
_ui = {}


@ui.page('/')
def main_page():
    ui.dark_mode().enable()
    ui.add_head_html('''
    <style>
    body { background: #0f172a !important; margin: 0; overflow: hidden; }
    .nicegui-content { max-width: 100% !important; padding: 0 !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #1a2433; }
    ::-webkit-scrollbar-thumb { background: #3b4a63; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #4a5a78; }
    /* 折叠区头部标题不换行 */
    .q-expansion-item__header { white-space: nowrap; }
    /* 输入框聚焦时金色边框 */
    .q-field--outlined.q-field--focused .q-field__control { border-color: #FFB320 !important; }
    .q-field--outlined.q-field--focused .q-field__label { color: #FFB320 !important; }
    /* 透明金色按钮悬停时金色填充 */
    .btn-gold-outline:hover { background: rgba(255, 179, 32, 0.12); border-color: #FFB320; color: #FFB320; }
    </style>
    ''')

    with ui.column().classes('w-full h-[100dvh] bg-[#0f172a] p-0 gap-0').style('overflow: hidden'):
        # ====== 主容器：左右分栏 ======
        with ui.row().classes('w-full h-full'):

            # ========== 左侧面板 ==========
            with ui.column().classes('w-[350px] min-w-[350px] bg-[#1a2433] border-r border-[#2d3a4f]').style('height: 100%; overflow: hidden'):

                # 标题行 + 折叠功能区按钮（始终可见）
                with ui.row().classes('w-full items-center justify-between px-3 py-2 border-b border-[#2d3a4f]').style('flex-shrink: 0'):
                    ui.label('🏀 进球集锦助手').classes('text-white text-sm font-bold')
                    collapse_btn = ui.button('▲ 收起功能区', on_click=lambda: _toggle_func_collapse()).classes(
                        'bg-[#2d3a4f] text-gray-200 text-xs')

                # 功能区域（固定高度，紧凑）
                with ui.column().classes('w-full px-3 py-2 gap-1 border-b border-[#2d3a4f]').style('flex-shrink: 0') as func_container:

                    # 输入框 + 加载按钮（一行）
                    path_input = ui.input(value=_DEFAULT_VIDEO, placeholder='文件路径').classes('w-full').props('dense')
                    info_text = ui.label('').classes('text-gray-400 text-xs font-mono hidden')
                    calib_status = ui.label('').classes('text-gray-400 text-xs font-mono hidden')
                    with ui.row().classes('w-full gap-2'):
                        ui.button('加载', on_click=lambda: _on_load()).classes('flex-1 bg-[#2d3a4f] text-gray-200 text-sm')
                        ui.button('重置', on_click=lambda: _on_reset()).classes('bg-[#2d3a4f] text-gray-200 text-sm')

                    # 开始识别
                    detect_btn = ui.button('开始识别', on_click=lambda: _on_detect()).classes(
                        'w-full bg-[#FFB320] text-black font-bold text-sm')

                    # 文件夹批量模式面板（加载文件夹后显示）
                    batch_panel = ui.column().classes('w-full gap-1 hidden')
                    with batch_panel:
                        batch_select = ui.select(options={}, value=None).classes('w-full').props('outlined dense dark')
                        with ui.row().classes('w-full gap-2'):
                            ui.button('保存标定', on_click=lambda: _on_batch_save_calib()).classes('flex-1 bg-[#2d3a4f] text-gray-200 text-xs')
                            batch_run_btn = ui.button('批量识别', on_click=lambda: _on_batch_run()).classes('flex-1 bg-[#FFB320] text-black text-xs')

                    # 结果状态
                    result_status = ui.label('').classes('text-gray-400 text-xs')

                    # 折叠区域：参数 / 集锦 / 历史 合并到一个框（展开时占满整框）
                    with ui.row().classes('w-full border border-[#2d3a4f] rounded-lg overflow-hidden').style('gap: 0; flex-wrap: wrap'):
                        exp_params = ui.expansion('参数', group='leftpanel').classes('w-1/3 text-gray-300 text-xs').style('min-width: 0').props('duration=0')
                        with exp_params:
                            with ui.column().classes('gap-1 w-full p-1 max-h-[300px] overflow-y-auto'):
                                with ui.row().classes('gap-2 w-full'):
                                    start_frame = ui.number(label='起始帧', value=0, format='%d').classes('flex-1')
                                    end_frame = ui.number(label='结束帧(0=末尾)', value=0, format='%d').classes('flex-1')
                                with ui.row().classes('gap-2 w-full'):
                                    ball_conf = ui.slider(min=0.1, max=0.9, value=0.3, step=0.05).classes('flex-1')
                                    ui.label().bind_text_from(ball_conf, 'value', lambda v: f'置信度: {v:.2f}').classes('text-gray-400 text-xs')
                                with ui.row().classes('gap-2 w-full'):
                                    min_gap = ui.slider(min=1.0, max=10.0, value=3.0, step=0.5).classes('flex-1')
                                    ui.label().bind_text_from(min_gap, 'value', lambda v: f'进球间隔: {v:.1f}s').classes('text-gray-400 text-xs')
                                with ui.expansion('高级', icon='tune').classes('w-full text-gray-400'):
                                    diff_threshold = ui.slider(min=5, max=40, value=15, step=5).classes('w-full')
                                    ui.label().bind_text_from(diff_threshold, 'value', lambda v: f'帧差阈值: {v}').classes('text-gray-400 text-xs')
                                    min_circularity = ui.slider(min=0.0, max=0.8, value=0.35, step=0.05).classes('w-full')
                                    ui.label().bind_text_from(min_circularity, 'value', lambda v: f'圆形度: {v:.2f}').classes('text-gray-400 text-xs')
                                    min_in_hoop_frames = ui.slider(min=1, max=6, value=2, step=1).classes('w-full')
                                    ui.label().bind_text_from(min_in_hoop_frames, 'value', lambda v: f'进框帧数: {v}').classes('text-gray-400 text-xs')
                                    min_blob_area = ui.slider(min=10, max=200, value=30, step=10).classes('w-full')
                                    ui.label().bind_text_from(min_blob_area, 'value', lambda v: f'最小斑块: {v}').classes('text-gray-400 text-xs')
                                    search_margin = ui.slider(min=20, max=150, value=80, step=10).classes('w-full')
                                    ui.label().bind_text_from(search_margin, 'value', lambda v: f'搜索范围: {v}px').classes('text-gray-400 text-xs')
                        exp_hl = ui.expansion('集锦', group='leftpanel').classes('w-1/3 text-gray-300 text-xs').style('min-width: 0').props('duration=0')
                        with exp_hl:
                            with ui.column().classes('gap-1 w-full p-1'):
                                hl_pre_roll = ui.slider(min=0, max=10, value=5, step=1).classes('w-full')
                                ui.label().bind_text_from(hl_pre_roll, 'value', lambda v: f'提前: {v:.0f}s').classes('text-gray-400 text-xs')
                                hl_post_roll = ui.slider(min=0, max=10, value=5, step=1).classes('w-full')
                                ui.label().bind_text_from(hl_post_roll, 'value', lambda v: f'延后: {v:.0f}s').classes('text-gray-400 text-xs')
                                hl_min_gap = ui.slider(min=1, max=30, value=8, step=1).classes('w-full')
                                ui.label().bind_text_from(hl_min_gap, 'value', lambda v: f'合并间隔: {v:.0f}s').classes('text-gray-400 text-xs')
                        exp_hist = ui.expansion('历史', group='leftpanel').classes('w-1/3 text-gray-300 text-xs').style('min-width: 0').props('duration=0')
                        with exp_hist:
                               history_list = ui.column().classes('w-full gap-1 max-h-[120px] overflow-y-auto')

                    # 展开的折叠区占满整框，其余两个隐藏；折叠后恢复并排
                    _exp_list = [exp_params, exp_hl, exp_hist]
                    def _sync_expand(exp=None):
                        # 从实际状态重新计算，避免触发顺序导致不一致
                        active = None
                        for _e in _exp_list:
                            if _e.value:
                                active = _e
                                break
                        if active is None:
                            for _e in _exp_list:
                                _e.classes(add='w-1/3', remove='hidden w-full')
                        else:
                            for _e in _exp_list:
                                if _e is active:
                                    _e.classes(add='w-full', remove='w-1/3 hidden')
                                else:
                                    _e.classes(add='hidden', remove='w-1/3 w-full')
                    for _e in _exp_list:
                        _e.on_value_change(lambda evt, x=_e: _sync_expand(x))
                    # 展开「历史」时自动加载记录，无需手动点刷新
                    exp_hist.on_value_change(lambda e: _refresh_history() if e.value else None)
                    _sync_expand()

                # 导出集锦按钮（固定在列表上方，仅列表有内容时显示）
                with ui.row().classes('w-full px-3 pt-2 flex-shrink-0 hidden') as export_row:
                    ui.button('导出集锦', on_click=lambda: _on_highlights()).classes(
                        'w-full bg-[#FFB320] text-black text-sm font-bold')

                # 进球列表区域（独立滚动）
                with ui.column().classes('w-full p-2 gap-1 overflow-y-auto flex-1').style('min-height: 0'):
                    # 结果列表容器
                    result_container = ui.column().classes('w-full gap-1')

                # ====== 顶部功能区折叠控制 ======
                _func_state = {"collapsed": False}

                def _set_func_collapsed(collapsed: bool):
                    """折叠/展开顶部功能区，把空间让给进球列表。"""
                    _func_state["collapsed"] = collapsed
                    if collapsed:
                        func_container.classes(add='hidden')
                        collapse_btn.set_text('▾ 展开功能区')
                        collapse_btn.classes(remove='bg-[#2d3a4f] text-gray-200').classes(
                            add='btn-gold-outline border border-[#FFB320]/60 bg-transparent text-[#FFB320]')
                    else:
                        func_container.classes(remove='hidden')
                        collapse_btn.set_text('▲ 收起功能区')
                        collapse_btn.classes(remove='btn-gold-outline border border-[#FFB320]/60 bg-transparent text-[#FFB320]').classes(
                            add='bg-[#2d3a4f] text-gray-200')

                def _toggle_func_collapse():
                    _set_func_collapsed(not _func_state["collapsed"])

            # ========== 右侧面板 ==========
            with ui.column().classes('flex-1 bg-[#0f172a] p-4 gap-3'):

                # 视频预览区
                preview_image = ui.interactive_image(
                    source='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
                ).classes('w-full rounded-xl bg-black').style('aspect-ratio: 16/9; object-fit: contain')

                result_video_el = ui.video(src='').classes('w-full rounded-xl hidden').style('aspect-ratio: 16/9')
                highlights_video_el = ui.video(src='').classes('w-full rounded-xl hidden').style('aspect-ratio: 16/9')

                # 进度显示区（检测时显示，居中）
                progress_container = ui.column().classes('w-full hidden items-center justify-center').style('aspect-ratio: 16/9')
                with progress_container:
                    progress_text = ui.label('检测中...').classes('text-[#FFB320] text-lg font-semibold')
                    progress_bar = ui.linear_progress(show_value=False).classes('w-64 mt-3')
                    progress_bar.style('background-color: #2d3a4f; color: #FFB320')
                    progress_detail = ui.label('').classes('text-gray-400 text-xs mt-2')

                # 底部留白
                ui.label('').classes('h-4')

    # ====== 事件处理函数 ======
    def _set_status(text, kind='info'):
        """设置结果状态文本并切换颜色（ok=绿 / err=红 / busy=金 / info=灰）。"""
        result_status.set_text(text)
        result_status.classes(
            remove='text-green-400 text-red-400 text-[#FFB320] text-gray-400')
        if kind == 'ok':
            result_status.classes(add='text-green-400')
        elif kind == 'err':
            result_status.classes(add='text-red-400')
        elif kind == 'busy':
            result_status.classes(add='text-[#FFB320]')
        else:
            result_status.classes(add='text-gray-400')

    def _on_load():
        global _batch_files, _batch_calibs, _batch_current_video
        path = path_input.value
        # 文件夹路径 → 批量标定 + 批量识别模式
        if path and os.path.isdir(path.strip().strip('"')):
            files = _scan_video_files(path)
            if not files:
                _set_status('文件夹内没有找到视频文件', 'err')
                return
            _batch_files = files
            _batch_calibs = {}
            _batch_current_video = None
            batch_panel.classes(remove='hidden')
            _refresh_batch_list()
            # 自动加载第一个视频（同时同步下拉框）
            _on_batch_load_video(files[0])
            _set_status(f'批量模式 | 扫描到 {len(files)} 个视频，逐个标定后批量识别', 'info')
            return
        # 单视频文件路径 → 原有流程
        batch_panel.classes(add='hidden')
        result = load_video(path)
        frame, info = result
        if frame is not None:
            # 切换视频：清空上一轮的进球结果，避免列表残留旧视频数据
            _last_goal_clips.clear()
            _last_goals.clear()
            _kept_goal_indices.clear()
            _refresh_result_cards()
            b64 = _frame_to_base64(frame)
            preview_image.set_source(b64)
            preview_image.classes(remove='hidden')
            result_video_el.classes(add='hidden')
            highlights_video_el.classes(add='hidden')
        info_text.set_text(info)
        calib_status.set_text('请点击画面 2 个点标定篮筐' if frame is not None else info)

    def _refresh_batch_list():
        """刷新批量下拉框：文件名前带标定状态标记（✓已标定 / ○未标定），保留当前选中。"""
        if not _batch_files:
            return
        cur_val = batch_select.value if batch_select.value in _batch_files else None
        # 用 dict（值->标签）作为 options：dict 时 select 的值才是纯路径字符串；
        # 若用 [(值,标签)] 列表，NiceGUI 会把整个元组当值，导致加载失败
        batch_select.set_options(
            {f: (f'✓ {os.path.basename(f)}' if f in _batch_calibs
                 else f'○ {os.path.basename(f)}') for f in _batch_files},
            value=cur_val)

    _batch_loading = False  # 防重入（set_value 可能触发 change 事件）

    def _on_batch_load_video(path=None):
        """加载批量视频（从下拉或列表点击）。"""
        nonlocal _batch_loading
        # 兼容旧版（值,标签）元组，防御性解包
        if isinstance(path, (tuple, list)):
            path = path[0]
        print(f"[BATCH-LOAD] path={path!r} current={_batch_current_video!r} loading={_batch_loading}", flush=True)
        if _batch_loading:
            return
        if path is None:
            path = batch_select.value
        if not path:
            # 下拉框被重置（如重新扫描）时静默跳过，避免误报
            if _batch_current_video is None:
                _set_status('请先选择视频', 'err')
            return
        if path == _batch_current_video:
            # 已是当前视频：仅同步下拉框显示，避免重复加载
            if batch_select.value != path:
                batch_select.set_value(path)
            return
        _batch_loading = True
        try:
            batch_select.set_value(path)
            frame, info, status = on_batch_load_video(path)
            print(f"[BATCH-LOAD] frame={'OK' if frame is not None else 'None'} path={_video_state['path']!r}", flush=True)
            if frame is not None:
                # 切换批量视频：清空上一轮的进球结果
                _last_goal_clips.clear()
                _last_goals.clear()
                _kept_goal_indices.clear()
                _refresh_result_cards()
                b64 = _frame_to_base64(frame)
                preview_image.set_source(b64)
                preview_image.classes(remove='hidden')
                result_video_el.classes(add='hidden')
                highlights_video_el.classes(add='hidden')
            info_text.set_text(info)
            calib_status.set_text(status)
        finally:
            _batch_loading = False

    def _on_batch_save_calib():
        status = on_batch_save_calib()
        calib_status.set_text(status)
        _refresh_batch_list()
        _set_status(status, 'ok' if '已保存' in status else 'err')

    _batch_running = {"active": False}

    async def _on_batch_run():
        global _cancel_requested
        if _batch_running["active"]:
            # 批量中点击 → 请求取消，run_batch_detect 轮询后中断
            _cancel_requested = True
            batch_run_btn.set_text('正在取消...')
            batch_run_btn.disable()
            return
        if not _batch_files:
            _set_status('请先加载文件夹', 'err')
            return
        _batch_running["active"] = True
        _cancel_requested = False
        batch_run_btn.set_text('取消')
        batch_run_btn.enable()
        # 显示进度
        progress_container.classes(remove='hidden')
        preview_image.classes(add='hidden')
        result_video_el.classes(add='hidden')
        highlights_video_el.classes(add='hidden')
        progress_bar.set_value(0)
        progress_text.set_text('正在批量识别...')
        progress_detail.set_text('')

        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                progress_text.set_text('批量识别中...')
                progress_detail.set_text(msg)
            except Exception:
                pass

        from nicegui import run
        status, ok = await run.io_bound(
            run_batch_detect,
            start_frame.value, end_frame.value, ball_conf.value, min_gap.value,
            diff_threshold.value, min_circularity.value, int(min_in_hoop_frames.value),
            min_blob_area.value, search_margin.value,
            progress_callback=_progress_callback)

        _batch_running["active"] = False
        batch_run_btn.set_text('批量识别')
        batch_run_btn.enable()
        progress_container.classes(add='hidden')
        preview_image.classes(remove='hidden')
        _set_status(status, 'ok' if ok else 'err')
        # 批量结束后显示最后一个视频的结果
        _refresh_result_cards()
        _refresh_batch_list()

    def _on_image_click(e):
        """点击预览图标定篮筐。

        使用 ui.interactive_image 的 on_mouse 事件，e.image_x/e.image_y
        已由前端按 显示尺寸/原始尺寸 比例换算为原始帧坐标。
        """
        try:
            x = int(round(e.image_x))
            y = int(round(e.image_y))
            nat_w = int(_video_state.get('width', 0) or 0)
            nat_h = int(_video_state.get('height', 0) or 0)
            if nat_w > 0:
                x = max(0, min(x, nat_w - 1))
            if nat_h > 0:
                y = max(0, min(y, nat_h - 1))
            frame, status = click_calibrate(x, y)
            if frame is not None:
                b64 = _frame_to_base64(frame)
                preview_image.set_source(b64)
            calib_status.set_text(status)
        except Exception as ex:
            calib_status.set_text(f'点击解析失败: {ex}')

    # 注册鼠标事件（interactive_image 默认监听 click，on_mouse 回调直接返回换算后的坐标）
    preview_image.on_mouse(_on_image_click)

    # 下拉框选择视频后立即加载（无需再点「加载」按钮），直接用事件携带的新值
    batch_select.on_value_change(lambda e: _on_batch_load_video(e.value))

    def _on_reset():
        status = reset_hoop()
        calib_status.set_text(status)
        # 刷新预览
        if _video_state["path"]:
            frame, _ = preview_frame(_video_state["current_frame"])
            if frame is not None:
                preview_image.set_source(_frame_to_base64(frame))

    _detecting = {"active": False}

    async def _on_detect():
        global _cancel_requested
        if _detecting["active"]:
            # 检测中点击 → 请求取消，run_detect 轮询后中断
            _cancel_requested = True
            detect_btn.set_text('正在取消...')
            detect_btn.disable()
            return
        _detecting["active"] = True
        _cancel_requested = False
        detect_btn.set_text('取消')
        detect_btn.enable()
        # 显示进度条
        progress_container.classes(remove='hidden')
        preview_image.classes(add='hidden')
        result_video_el.classes(add='hidden')
        highlights_video_el.classes(add='hidden')
        progress_bar.set_value(0)
        progress_text.set_text('正在加载模型...')
        progress_detail.set_text('')

        # 进度回调函数
        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                progress_text.set_text(msg)
            except Exception:
                pass

        # 使用 run.io_bound 在后台线程执行，避免阻塞事件循环
        from nicegui import run
        status, ok = await run.io_bound(
            run_detect,
            start_frame.value, end_frame.value, ball_conf.value, min_gap.value,
            diff_threshold.value, min_circularity.value, int(min_in_hoop_frames.value),
            min_blob_area.value, search_margin.value,
            progress_callback=_progress_callback)

        _detecting["active"] = False
        # 隐藏进度条，显示结果
        progress_container.classes(add='hidden')
        preview_image.classes(remove='hidden')
        if _cancel_requested:
            _set_status(status, 'info')  # 用户取消属中性提示，不用红色
            _refresh_result_cards()      # 同步清空列表
        else:
            _set_status(status, 'ok' if ok else 'err')
        detect_btn.set_text('开始识别')
        detect_btn.enable()
        if ok:
            _refresh_result_cards()

    def _refresh_result_cards():
        """刷新结果卡片列表。"""
        result_container.clear()
        if not _last_goal_clips:
            with result_container:
                ui.label('暂无进球结果').classes('text-gray-300 text-xs text-center w-full py-4')
                ui.label('请先加载视频 → 标定篮筐 → 开始识别').classes('text-gray-400 text-xs text-center w-full')
            export_row.classes(add='hidden')  # 无结果时隐藏导出按钮
            _set_func_collapsed(False)  # 列表为空时展开功能区
            return
        export_row.classes(remove='hidden')  # 有结果时显示导出按钮
        _set_func_collapsed(True)  # 进球列表出来后自动折叠顶部功能区，把空间让给列表
        for i, clip in enumerate(_last_goal_clips):
            ts = clip["ts"]
            t_min, t_sec = int(ts // 60), ts % 60
            end_ts = ts + 10
            end_min, end_sec = int(end_ts // 60), end_ts % 60
            with result_container:
                with ui.card().props('flat').classes('w-full bg-[#1f2b3d] rounded-lg px-2 py-1.5').style('margin: 0; border: 1px solid #334155'):
                    # 第一行：序号 + 时间范围（紧凑）
                    with ui.row().classes('w-full items-center gap-2 mb-1'):
                        ui.label(str(i+1)).classes(
                            'bg-[#FFB320] text-black font-bold text-xs w-5 h-5 flex items-center justify-center rounded-full')
                        ui.label(f'{t_min}:{t_sec:04.1f} - {end_min}:{end_sec:04.1f}').classes('text-white text-sm font-bold font-mono')
                    # 第二行：操作按钮（删除用危险色区分，填满宽度）
                    with ui.row().classes('w-full gap-1'):
                        ui.button('预览', on_click=lambda e, idx=i: _on_preview_clip(idx)).classes(
                            'flex-1 border border-gray-600 bg-transparent text-gray-300 text-xs rounded-lg py-1')
                        ui.button('导出', on_click=lambda e, idx=i: _on_export_clip(idx)).classes(
                            'flex-1 border border-gray-600 bg-transparent text-gray-300 text-xs rounded-lg py-1')
                        ui.button('删除', on_click=lambda e, idx=i: _on_delete_clip(idx)).classes(
                            'flex-1 border border-red-500/60 bg-transparent text-red-400 text-xs rounded-lg py-1')

    def _on_preview_clip(idx):
        path, status = clip_action("preview", idx)
        if path and os.path.exists(path):
            result_video_el.set_source(path)
            preview_image.classes(add='hidden')
            result_video_el.classes(remove='hidden')
            highlights_video_el.classes(add='hidden')
            # 自动播放
            result_video_el.run_method('play')
        _set_status(status, 'info')

    def _on_export_clip(idx):
        path, status = clip_action("export", idx)
        if path and os.path.exists(path):
            ui.download(path)
        _set_status(status, 'ok' if path and os.path.exists(path) else 'err')

    def _on_delete_clip(idx):
        # 点击直接删除，不弹确认框
        clip_action("delete", idx)
        _refresh_result_cards()
        _set_status(f'已删除第 {idx+1} 个片段 | 剩余 {len(_last_goal_clips)} 个', 'info')

    async def _on_highlights():
        from nicegui import run
        # 在右侧预览区显示进度
        progress_container.classes(remove='hidden')
        preview_image.classes(add='hidden')
        result_video_el.classes(add='hidden')
        highlights_video_el.classes(add='hidden')
        progress_bar.set_value(0)
        progress_text.set_text('正在生成集锦...')
        progress_detail.set_text('')

        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                progress_text.set_text(msg)
            except Exception:
                pass

        _set_status('正在生成集锦...', 'busy')
        path, status = await run.io_bound(
            generate_highlights, hl_pre_roll.value, hl_post_roll.value,
            hl_min_gap.value, _progress_callback)

        progress_container.classes(add='hidden')
        if path and os.path.exists(path):
            ui.download(path)
            highlights_video_el.set_source(path)
            preview_image.classes(add='hidden')
            result_video_el.classes(add='hidden')
            highlights_video_el.classes(remove='hidden')
        else:
            preview_image.classes(remove='hidden')
        _set_status(status, 'ok' if path and os.path.exists(path) else 'err')

    _selected_history_idx = {"idx": None}

    def _refresh_history():
        history_list.clear()
        records = _load_history()
        if not records:
            with history_list:
                ui.label('暂无历史记录').classes('text-gray-400 text-xs')
            return
        for i, r in enumerate(records):
            video = r.get("video", "")
            name = os.path.basename(video) if video else "未知"
            goals = len(r.get("goals", []))
            selected = (_selected_history_idx.get("idx") == i)
            row_cls = ('border border-[#FFB320] bg-[#2a2f3e]'
                       if selected else 'border border-transparent hover:bg-[#2a2f3e]')
            with history_list:
                def _make_click(idx=i):
                    async def _on_click():
                        _selected_history_idx["idx"] = idx
                        _refresh_history()
                        await _on_load_history()
                        exp_hist.set_value(False)  # 加载完成后自动收起历史面板
                    return _on_click
                with ui.row().classes(f'w-full items-center gap-1 p-1 rounded cursor-pointer {row_cls}').on('click', _make_click(i)):
                    ui.label(f'{i+1}.').classes('text-[#FFB320] text-xs font-bold')
                    ui.label(name).classes('text-gray-300 text-xs flex-1 truncate')
                    ui.label(f'{goals}球').classes('text-gray-400 text-xs')

    async def _on_load_history():
        idx = _selected_history_idx.get("idx")
        if idx is None:
            _set_status('请先点击选择一条历史记录', 'err')
            return

        # 显示进度
        progress_container.classes(remove='hidden')
        preview_image.classes(add='hidden')
        progress_bar.set_value(0)
        progress_text.set_text('正在加载历史记录...')

        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                progress_text.set_text(msg)
            except Exception:
                pass

        from nicegui import run
        result = await run.io_bound(on_load_history, int(idx), _progress_callback)
        frame, info, status = result

        progress_container.classes(add='hidden')
        preview_image.classes(remove='hidden')
        if frame is not None:
            preview_image.set_source(_frame_to_base64(frame))
        # 同步路径输入框，显示当前加载的视频
        if _video_state["path"]:
            path_input.set_value(_video_state["path"])
        info_text.set_text(info)
        _set_status(status, 'ok' if frame is not None else 'err')
        _refresh_result_cards()

    # 页面初始化：不自动加载历史列表（空白初始状态），点「刷新」才加载
    _refresh_result_cards()


# ============ 启动 ============

if __name__ == "__main__":
    _out_dir = str(Path(_CACHE_ROOT) / "demo_output")
    os.makedirs(_out_dir, exist_ok=True)

    # 预加载 YOLO 模型到 GPU 并在主线程预热推理。
    # 原因：检测在后台线程首次初始化 CUDA 上下文可能导致驱动层崩溃
    # （Windows 事件日志: nvcuda64.dll 0xC0000409），预热后后台线程只复用主线程上下文。
    _warmup_log = os.path.join(_CACHE_ROOT, "warmup_status.log")
    try:
        import torch
        model, _ = get_ball_model()
        device = get_device()
        if device != "cpu":
            warm = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(warm, conf=0.5, imgsz=640, device=device, verbose=False)
            torch.cuda.empty_cache()
            _msg = f"WARMUP-OK: {device} context created on main thread"
        else:
            _msg = "WARMUP-SKIP: CUDA not available, using CPU"
    except Exception as e:
        _msg = f"WARMUP-FAIL: {e}"
    with open(_warmup_log, "w", encoding="utf-8") as _f:
        _f.write(_msg)
    print(_msg, flush=True)

    ui.run(host="127.0.0.1", port=7871, title="进球集锦助手",
           dark=True, reload=False)
