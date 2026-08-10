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
    E:\\bball-env\\python.exe demo_nicegui.py
浏览器打开 http://127.0.0.1:7871
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

_CACHE_ROOT = r"E:\bball_cache"
# Python 临时文件重定向到 E 盘，避免 C 盘空间不足
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
from app import get_ball_model
from tracker import GoalDetector
from cutter.ffmpeg_cutter import cut_clips, _build_encode_args

# 历史记录文件
_HISTORY_FILE = os.path.join(_CACHE_ROOT, "detection_history.json")

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
_last_goal_types = []

_DEFAULT_VIDEO = r"D:\Downloads\highlights.mp4"

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
    """将 cv2 帧转为 base64 PNG data URI。"""
    if frame is None:
        return None
    _, buf = cv2.imencode('.png', frame)
    b64 = base64.b64encode(buf).decode('utf-8')
    return f'data:image/png;base64,{b64}'


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
        status = f"✅ 篮筐已标定: ({x1},{y1}) - ({x2},{y2}) | 基准帧: 第 {int(frame_idx)} 帧"
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
    global _kept_goal_indices, _last_goal_clips, _last_goals
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
                ball_pos = None
                try:
                    res = model.predict(frame, conf=float(ball_conf), imgsz=1280,
                                        device="cuda:0", verbose=False)[0]
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

        for gi, gts in enumerate(goals):
            gframe = int(gts * fps)
            seg_start = max(start, gframe - clip_half)
            seg_end = min(end, gframe + clip_half)
            if seg_end <= seg_start:
                continue
            clip_path = str(out_dir / f"goal_{gi}_{int(gts)}s_{_stamp}.mp4")
            seg_start_sec = seg_start / fps
            seg_dur_sec = (seg_end - seg_start) / fps
            try:
                _enc = _build_encode_args(ff, quality="preview")
                _sp.run([ff, "-y", "-loglevel", "error",
                         "-ss", f"{seg_start_sec:.3f}", "-i", _video_state["path"],
                         "-t", f"{seg_dur_sec:.3f}",
                         "-vf", "scale=-2:480"] + _enc +
                        ["-movflags", "+faststart", clip_path],
                        creationflags=0x08000000, capture_output=True, timeout=60)
                if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                    _last_goal_clips.append({"ts": gts, "path": clip_path, "idx": gi})
            except Exception:
                pass
            if len(goals) > 0:
                _report(80 + 18 * (gi + 1) / len(goals), f'生成片段 {gi+1}/{len(goals)}')

        _report(100, '完成！')
        _kept_goal_indices = set(range(len(_last_goal_clips)))
        _last_goals.clear()
        _last_goals.extend(detector.goals)

        d = detector.diag
        total_yolo = d['yolo_confirmed'] + d['yolo_rejected']
        confirm_rate = d['yolo_confirmed'] / max(total_yolo, 1) * 100

        status = (f"✅ 检测完成 | 处理 {processed} 帧 | 耗时 {elapsed:.0f}s\n"
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
        return _last_goal_clips[idx]["path"], f"✅ 已导出: {_last_goal_clips[idx]['path']}"
    elif action == "delete":
        global _kept_goal_indices, _last_goals
        ts = _last_goal_clips[idx]["ts"]
        del _last_goal_clips[idx]
        while len(_last_goal_types) < len(_last_goal_clips) + 1:
            _last_goal_types.append("进球")
        if idx < len(_last_goal_types):
            del _last_goal_types[idx]
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
        return None, f"✅ 已删除第 {idx+1} 个片段（{ts:.1f}s）| 剩余 {len(_last_goal_clips)} 个"
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
            return out_path, f"✅ 集锦已生成（{n} 个进球片段）\n输出: {out_path}"
        return None, "❌ 集锦生成失败"
    except Exception as e:
        import traceback
        return None, f"❌ 剪辑失败: {e}\n{traceback.format_exc()}"


def on_load_history(idx_choice, pre_roll, post_roll, cut_min_gap, progress_callback=None):
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
    if all_goals:
        import imageio_ffmpeg
        import subprocess as _sp
        ff = imageio_ffmpeg.get_ffmpeg_exe()
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
                _enc = _build_encode_args(ff, quality="preview")
                _sp.run([ff, "-y", "-loglevel", "error",
                         "-ss", f"{seg_start_sec:.3f}", "-i", video_path,
                         "-t", f"{seg_dur_sec:.3f}",
                         "-vf", "scale=-2:480"] + _enc +
                        ["-movflags", "+faststart", clip_path],
                        creationflags=0x08000000, capture_output=True, timeout=60)
                if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                    _last_goal_clips.append({"ts": float(gts), "path": clip_path, "idx": gi})
            except Exception:
                pass

    kept_set = set(kept)
    for i, clip in enumerate(_last_goal_clips):
        if clip["ts"] in kept_set or not kept:
            _kept_goal_indices.add(i)

    frame = read_frame(video_path, 0, total=total, fps=fps)
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")
    status = (f"✅ 已加载历史记录\n视频: {r.get('video_name', '')}\n"
              f"进球: {len(all_goals)} 个 | 保留: {len(kept)} 个\n"
              f"已生成 {len(_last_goal_clips)} 个预览片段")
    return preview, info_str, status


# ============ NiceGUI 界面 ============

# 全局 UI 引用（用于在回调中更新）
_ui = {}


@ui.page('/')
def main_page():
    ui.dark_mode().enable()
    ui.add_head_html('''
    <style>
    body { background: #0d1117 !important; margin: 0; overflow: hidden; }
    .nicegui-content { max-width: 100% !important; padding: 0 !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #161b22; }
    ::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #555; }
    /* 折叠区头部标题不换行 */
    .q-expansion-item__header { white-space: nowrap; }
    </style>
    ''')

    with ui.column().classes('w-full h-screen bg-[#0d1117] p-0 gap-0').style('overflow: hidden'):
        # ====== 主容器：左右分栏 ======
        with ui.row().classes('w-full h-full'):

            # ========== 左侧面板 ==========
            with ui.column().classes('w-[350px] min-w-[350px] bg-[#161b22] border-r border-[#30363d]').style('height: 100%; overflow: hidden'):

                # 功能区域（固定高度，紧凑）
                with ui.column().classes('w-full px-3 py-2 gap-1 border-b border-[#30363d]').style('flex-shrink: 0'):
                    # 标题行
                    with ui.row().classes('items-center gap-2'):
                        ui.label('🏀').classes('text-base')
                        ui.label('进球集锦助手').classes('text-white text-sm font-bold')

                    # 输入框 + 加载按钮（一行）
                    path_input = ui.input(value=_DEFAULT_VIDEO, placeholder='文件路径').classes('w-full')
                    info_text = ui.label('').classes('text-gray-500 text-xs font-mono hidden')
                    calib_status = ui.label('').classes('text-gray-500 text-xs font-mono hidden')
                    with ui.row().classes('w-full gap-2'):
                        ui.button('加载', on_click=lambda: _on_load()).classes('flex-1 bg-[#30363d] text-gray-300 text-xs')
                        ui.button('重置', on_click=lambda: _on_reset()).classes('bg-[#30363d] text-gray-300 text-xs')

                    # 开始识别
                    detect_btn = ui.button('🔍 开始识别', on_click=lambda: _on_detect()).classes(
                        'w-full bg-[#FFB320] text-black font-bold text-xs')

                    # 操作按钮行
                    with ui.row().classes('gap-2 w-full'):
                        ui.button('导出合集', on_click=lambda: _on_highlights()).classes(
                            'flex-1 bg-[#30363d] text-gray-300 text-xs')

                    # 结果状态
                    result_status = ui.label('').classes('text-gray-500 text-xs')

                    # 折叠区域：参数 / 集锦 / 历史 合并到一个框（展开时占满整框）
                    with ui.row().classes('w-full border border-[#30363d] rounded-lg overflow-hidden').style('gap: 0; flex-wrap: wrap'):
                        exp_params = ui.expansion('⚙️ 参数', group='leftpanel').classes('w-1/3 text-gray-400 text-xs').style('min-width: 0').props('duration=0')
                        with exp_params:
                            with ui.column().classes('gap-1 w-full p-1 max-h-[300px] overflow-y-auto'):
                                with ui.row().classes('gap-2 w-full'):
                                    start_frame = ui.number(label='起始帧', value=0, format='%d').classes('flex-1')
                                    end_frame = ui.number(label='结束帧(0=末尾)', value=0, format='%d').classes('flex-1')
                                with ui.row().classes('gap-2 w-full'):
                                    ball_conf = ui.slider(min=0.1, max=0.9, value=0.3, step=0.05).classes('flex-1')
                                    ui.label().bind_text_from(ball_conf, 'value', lambda v: f'置信度: {v:.2f}').classes('text-gray-500 text-xs')
                                with ui.row().classes('gap-2 w-full'):
                                    min_gap = ui.slider(min=1.0, max=10.0, value=3.0, step=0.5).classes('flex-1')
                                    ui.label().bind_text_from(min_gap, 'value', lambda v: f'间隔: {v:.1f}s').classes('text-gray-500 text-xs')
                                with ui.expansion('🔧 高级', icon='tune').classes('w-full text-gray-500'):
                                    diff_threshold = ui.slider(min=5, max=40, value=15, step=5).classes('w-full')
                                    ui.label().bind_text_from(diff_threshold, 'value', lambda v: f'帧差阈值: {v}').classes('text-gray-500 text-xs')
                                    min_circularity = ui.slider(min=0.0, max=0.8, value=0.35, step=0.05).classes('w-full')
                                    ui.label().bind_text_from(min_circularity, 'value', lambda v: f'圆形度: {v:.2f}').classes('text-gray-500 text-xs')
                                    min_in_hoop_frames = ui.slider(min=1, max=6, value=2, step=1).classes('w-full')
                                    ui.label().bind_text_from(min_in_hoop_frames, 'value', lambda v: f'进框帧数: {v}').classes('text-gray-500 text-xs')
                                    min_blob_area = ui.slider(min=10, max=200, value=30, step=10).classes('w-full')
                                    ui.label().bind_text_from(min_blob_area, 'value', lambda v: f'最小斑块: {v}').classes('text-gray-500 text-xs')
                                    search_margin = ui.slider(min=20, max=150, value=80, step=10).classes('w-full')
                                    ui.label().bind_text_from(search_margin, 'value', lambda v: f'搜索范围: {v}px').classes('text-gray-500 text-xs')
                        exp_hl = ui.expansion('🎬 集锦', group='leftpanel').classes('w-1/3 text-gray-400 text-xs').style('min-width: 0').props('duration=0')
                        with exp_hl:
                            with ui.column().classes('gap-1 w-full p-1'):
                                hl_pre_roll = ui.slider(min=0, max=10, value=5, step=1).classes('w-full')
                                ui.label().bind_text_from(hl_pre_roll, 'value', lambda v: f'提前: {v:.0f}s').classes('text-gray-500 text-xs')
                                hl_post_roll = ui.slider(min=0, max=10, value=5, step=1).classes('w-full')
                                ui.label().bind_text_from(hl_post_roll, 'value', lambda v: f'延后: {v:.0f}s').classes('text-gray-500 text-xs')
                                hl_min_gap = ui.slider(min=1, max=30, value=8, step=1).classes('w-full')
                                ui.label().bind_text_from(hl_min_gap, 'value', lambda v: f'合并间隔: {v:.0f}s').classes('text-gray-500 text-xs')
                        exp_hist = ui.expansion('📂 历史', group='leftpanel').classes('w-1/3 text-gray-400 text-xs').style('min-width: 0').props('duration=0')
                        with exp_hist:
                               history_list = ui.column().classes('w-full gap-1 max-h-[120px] overflow-y-auto')
                               with ui.row().classes('gap-2'):
                                   ui.button('🔄', on_click=lambda: _refresh_history()).classes('bg-[#30363d] text-gray-300 text-xs')
                                   ui.button('加载', on_click=lambda: _on_load_history()).classes('bg-[#FFB320] text-black text-xs')

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
                    _sync_expand()

                # 进球列表区域（独立滚动）
                with ui.column().classes('w-full p-3 gap-2 overflow-y-auto flex-1').style('min-height: 0'):
                    # 结果列表容器
                    result_container = ui.column().classes('w-full gap-2')

            # ========== 右侧面板 ==========
            with ui.column().classes('flex-1 bg-[#0d1117] p-4 gap-3'):

                # 视频预览区
                preview_image = ui.interactive_image(
                    source='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
                ).classes('w-full rounded-xl bg-black').style('aspect-ratio: 16/9; object-fit: contain')

                result_video_el = ui.video(src='').classes('w-full rounded-xl hidden').style('aspect-ratio: 16/9')
                highlights_video_el = ui.video(src='').classes('w-full rounded-xl hidden').style('aspect-ratio: 16/9')

                # 进度显示区（检测时显示，居中）
                progress_container = ui.column().classes('w-full hidden items-center justify-center').style('aspect-ratio: 16/9')
                with progress_container:
                    ui.label('🔍').classes('text-4xl mb-4')
                    progress_text = ui.label('检测中...').classes('text-[#FFB320] text-lg font-semibold')
                    progress_bar = ui.linear_progress(show_value=False).classes('w-64 mt-3')
                    progress_bar.style('background-color: #30363d; color: #FFB320')
                    progress_detail = ui.label('').classes('text-gray-500 text-xs mt-2')

                # 底部留白
                ui.label('').classes('h-4')

    # ====== 事件处理函数 ======
    def _on_load():
        path = path_input.value
        result = load_video(path)
        frame, info = result
        if frame is not None:
            b64 = _frame_to_base64(frame)
            preview_image.set_source(b64)
            preview_image.classes(remove='hidden')
            result_video_el.classes(add='hidden')
            highlights_video_el.classes(add='hidden')
        info_text.set_text(info)
        calib_status.set_text('请点击画面 2 个点标定篮筐' if frame is not None else info)

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

    def _on_reset():
        status = reset_hoop()
        calib_status.set_text(status)
        # 刷新预览
        if _video_state["path"]:
            frame, _ = preview_frame(_video_state["current_frame"])
            if frame is not None:
                preview_image.set_source(_frame_to_base64(frame))

    async def _on_detect():
        detect_btn.set_text('检测中...')
        detect_btn.disable()
        # 显示进度条
        progress_container.classes(remove='hidden')
        preview_image.classes(add='hidden')
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

        # 隐藏进度条，显示结果
        progress_container.classes(add='hidden')
        preview_image.classes(remove='hidden')
        result_status.set_text(status)
        detect_btn.set_text('开始识别')
        detect_btn.enable()
        if ok:
            _refresh_result_cards()

    def _refresh_result_cards():
        """刷新结果卡片列表。"""
        result_container.clear()
        if not _last_goal_clips:
            return
        for i, clip in enumerate(_last_goal_clips):
            ts = clip["ts"]
            t_min, t_sec = int(ts // 60), ts % 60
            end_ts = ts + 10
            end_min, end_sec = int(end_ts // 60), end_ts % 60
            with result_container:
                with ui.card().classes('w-full bg-[#1a1a1a] rounded-xl p-3').style('margin: 0; border: 1px solid #333'):
                    # 第一行：序号 + 时间范围 + 进球点
                    with ui.row().classes('w-full items-center gap-2 mb-2'):
                        ui.label(str(i+1)).classes(
                            'bg-[#FFB320] text-black font-bold text-xs w-6 h-6 flex items-center justify-center rounded-full')
                        ui.label(f'{t_min}:{t_sec:04.1f}-{end_min}:{end_sec:04.1f}').classes('text-white text-sm font-bold')
                        ui.label(f'可能进球点 {t_min}:{t_sec:04.1f}').classes('text-gray-500 text-xs ml-auto')
                        ui.html('<div style="width:6px;height:6px;border-radius:50%;background:#ef4444"></div>')
                    # 第二行：操作按钮（边框样式，填满宽度）
                    with ui.row().classes('w-full gap-2'):
                        ui.button('预览', on_click=lambda e, idx=i: _on_preview_clip(idx)).classes(
                            'flex-1 border border-gray-600 bg-transparent text-gray-300 text-xs rounded-lg py-2')
                        ui.button('导出', on_click=lambda e, idx=i: _on_export_clip(idx)).classes(
                            'flex-1 border border-gray-600 bg-transparent text-gray-300 text-xs rounded-lg py-2')
                        ui.button('删除', on_click=lambda e, idx=i: _on_delete_clip(idx)).classes(
                            'flex-1 border border-gray-600 bg-transparent text-gray-300 text-xs rounded-lg py-2')

    def _on_preview_clip(idx):
        path, status = clip_action("preview", idx)
        if path and os.path.exists(path):
            result_video_el.set_source(path)
            preview_image.classes(add='hidden')
            result_video_el.classes(remove='hidden')
            highlights_video_el.classes(add='hidden')
            # 自动播放
            result_video_el.run_method('play')
        result_status.set_text(status)

    def _on_export_clip(idx):
        path, status = clip_action("export", idx)
        if path and os.path.exists(path):
            ui.download(path)
        result_status.set_text(status)

    def _on_delete_clip(idx):
        clip_action("delete", idx)
        _refresh_result_cards()
        result_status.set_text(f'✅ 已删除第 {idx+1} 个片段 | 剩余 {len(_last_goal_clips)} 个')

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

        result_status.set_text('正在生成集锦...')
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
        result_status.set_text(status)

    _selected_history_idx = {"idx": None}

    def _refresh_history():
        history_list.clear()
        records = _load_history()
        if not records:
            with history_list:
                ui.label('暂无历史记录').classes('text-gray-500 text-xs')
            return
        for i, r in enumerate(records):
            video = r.get("video", "")
            name = os.path.basename(video) if video else "未知"
            goals = len(r.get("goals", []))
            with history_list:
                def _make_click(idx=i):
                    def _on_click():
                        _selected_history_idx["idx"] = idx
                    return _on_click
                with ui.row().classes('w-full items-center gap-1 p-1 rounded cursor-pointer hover:bg-[#2a2f3e]').on('click', _make_click(i)):
                    ui.label(f'{i+1}.').classes('text-[#FFB320] text-xs font-bold')
                    ui.label(name).classes('text-gray-300 text-xs flex-1 truncate')
                    ui.label(f'{goals}球').classes('text-gray-500 text-xs')

    async def _on_load_history():
        idx = _selected_history_idx.get("idx")
        if idx is None:
            result_status.set_text('请先点击选择一条历史记录')
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
        result = await run.io_bound(on_load_history, int(idx), 5, 5, 8, _progress_callback)
        frame, info, status = result

        progress_container.classes(add='hidden')
        preview_image.classes(remove='hidden')
        if frame is not None:
            preview_image.set_source(_frame_to_base64(frame))
        info_text.set_text(info)
        result_status.set_text(status)
        _refresh_result_cards()

    # 页面初始化时加载历史列表（所有函数已定义）
    _refresh_history()


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
        if torch.cuda.is_available():
            warm = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(warm, conf=0.5, imgsz=640, device="cuda:0", verbose=False)
            torch.cuda.empty_cache()
            _msg = "WARMUP-OK: CUDA context created on main thread"
        else:
            _msg = "WARMUP-SKIP: CUDA not available"
    except Exception as e:
        _msg = f"WARMUP-FAIL: {e}"
    with open(_warmup_log, "w", encoding="utf-8") as _f:
        _f.write(_msg)
    print(_msg, flush=True)

    ui.run(host="127.0.0.1", port=7871, title="进球集锦助手",
           dark=True, reload=False)
