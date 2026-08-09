"""篮球进球检测与自动剪辑交互式 Demo（Gradio）。

功能：
  1. 输入视频文件或文件夹路径 → 加载
  2. 滑动到含篮筐的帧，点击画面 2 个点标定篮筐
  3. 设置起止帧、置信度、最小进球间隔
  4. 点击「开始检测」→ diff + YOLO 双确认检测进球
  5. 每个进球生成独立预览片段，人工确认保留/删除
  6. 生成集锦视频（GPU 硬编加速）
  7. 历史记录支持加载后直接剪辑（无需重新检测）
  8. 文件夹模式：批量标定 + 批量识别，每个视频独立保存

算法：diff 基准帧差法（候选筛选）+ YOLO 软确认（可信度标记）+ 滚动基准帧（长视频适应）

用法:
    E:\\bball-env\\python.exe demo_interactive.py
浏览器打开 http://127.0.0.1:7871
"""
# ====== 必须在导入 gradio 之前设置环境变量 ======
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# 缓存目录改到 E 盘（C 盘空间不足会导致上传中断）
_CACHE_ROOT = r"E:\bball_cache"
_gradio_tmp = os.path.join(_CACHE_ROOT, "gradio")
os.makedirs(_gradio_tmp, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = _gradio_tmp
os.environ["TMPDIR"] = _gradio_tmp
os.environ["TEMP"] = _gradio_tmp
os.environ["TMP"] = _gradio_tmp

import time
import json
import cv2
import numpy as np
import gradio as gr

from video_io import get_video_info, read_frame, VideoReader
from app import get_ball_model
from tracker import GoalDetector
from cutter.ffmpeg_cutter import cut_clips, _build_encode_args, _detect_nvenc, _DEFAULT_FFMPEG

# 历史记录文件（保存检测过的视频与进球时间戳，支持直接回剪辑）
_HISTORY_FILE = os.path.join(_CACHE_ROOT, "detection_history.json")


def _load_history():
    """加载历史检测记录。返回 list[dict]，每条含视频路径、篮筐、进球时间戳等。"""
    try:
        if os.path.exists(_HISTORY_FILE):
            with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_history(records):
    """保存历史记录到 JSON 文件。"""
    try:
        os.makedirs(os.path.dirname(_HISTORY_FILE), exist_ok=True)
        with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 保存历史记录失败: {e}", flush=True)


def _add_history(video_path, hoop, goals, kept_goals):
    """新增一条检测记录到历史。同视频路径则覆盖。"""
    records = _load_history()
    # 用视频路径作为唯一标识，同路径覆盖
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
    # 最多保留 50 条
    records = records[:50]
    _save_history(records)


def _history_choices():
    """生成历史记录下拉选项列表。"""
    records = _load_history()
    choices = []
    for i, r in enumerate(records):
        name = r.get("video_name", "未知")
        kept = r.get("kept", 0)
        total = r.get("total", 0)
        tm = r.get("time", "")
        choices.append(f"[{i}] {name} | 保留{kept}/{total} | {tm}")
    return choices


def on_load_history(idx_choice, pre_roll, post_roll, cut_min_gap, progress=gr.Progress()):
    """从历史记录加载检测数据，重新生成预览片段，直接进入可剪辑状态。"""
    global _last_goals, _video_state, _calib, _last_goal_clips, _kept_goal_indices
    if not idx_choice:
        return (None, gr.update(), "请先选择一条历史记录",
                "请先选择一条历史记录", None, "",
                gr.update(choices=[], value=None), gr.update(choices=[], value=[]))

    # 解析索引
    try:
        idx = int(idx_choice.split("]")[0].replace("[", ""))
    except (ValueError, IndexError):
        return (None, gr.update(), "解析历史记录索引失败",
                "解析历史记录索引失败", None, "",
                gr.update(choices=[], value=None), gr.update(choices=[], value=[]))

    records = _load_history()
    if idx < 0 or idx >= len(records):
        return (None, gr.update(), "历史记录不存在",
                "历史记录不存在", None, "",
                gr.update(choices=[], value=None), gr.update(choices=[], value=[]))

    r = records[idx]
    video_path = r.get("video", "")
    if not os.path.exists(video_path):
        return (None, gr.update(), f"视频文件不存在: {video_path}",
                f"视频文件不存在: {video_path}", None, "",
                gr.update(choices=[], value=None), gr.update(choices=[], value=[]))

    # 加载视频信息
    try:
        info = get_video_info(video_path)
    except Exception as e:
        return (None, gr.update(), f"读取视频失败: {e}",
                f"读取视频失败: {e}", None, "",
                gr.update(choices=[], value=None), gr.update(choices=[], value=[]))

    _video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                        codec=info["codec"])

    # 恢复篮筐标定
    hoop = r.get("hoop")
    if hoop and len(hoop) == 4:
        _calib["hoop"] = tuple(int(v) for v in hoop)
        _calib["clicks"] = []
        base_frame = read_frame(video_path, 0, total=info["total"], fps=info["fps"])
        if base_frame is not None:
            _calib["baseline_frame"] = base_frame.copy()
            _calib["baseline_idx"] = 0

    # 恢复进球列表：始终加载全部进球，保留勾选状态供用户再次调整
    kept = [float(t) for t in r.get("kept_goals", [])]
    all_goals = [float(t) for t in r.get("goals", [])]
    goals_to_use = all_goals  # 始终用全部进球生成预览片段

    # _last_goals 用保留的（供集锦使用），没有保留则用全部
    _last_goals.clear()
    _last_goals.extend(kept if kept else all_goals)

    # 重新生成预览片段（ffmpeg 快速切片，480p）
    fps = info["fps"]
    total = info["total"]
    clip_half = int(fps * 3)  # 进球前后各 3 秒
    out_dir = Path(_CACHE_ROOT) / "demo_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    import time as _t
    _stamp = int(_t.time())

    _last_goal_clips.clear()
    _kept_goal_indices.clear()

    if goals_to_use:
        import imageio_ffmpeg
        import subprocess as _sp
        ff = imageio_ffmpeg.get_ffmpeg_exe()

        for gi, gts in enumerate(goals_to_use):
            progress(0.2 + 0.7 * gi / max(len(goals_to_use), 1),
                     desc=f"生成预览片段 {gi+1}/{len(goals_to_use)} ({gts:.1f}s)...")
            gframe = int(gts * fps)
            seg_start = max(0, gframe - clip_half)
            seg_end = min(total, gframe + clip_half)
            if seg_end <= seg_start:
                continue

            clip_path = str(out_dir / f"goal_{gi}_{int(gts)}s_{_stamp}.mp4")
            seg_start_sec = seg_start / fps
            seg_dur_sec = (seg_end - seg_start) / fps
            try:
                # NVENC 硬编加速预览片段生成（回退 libx264）
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

    # 根据历史记录中的 kept_goals 设置勾选状态（保留之前的勾选）
    _kept_goal_indices.clear()
    kept_set = set(kept)
    for i, clip in enumerate(_last_goal_clips):
        if clip["ts"] in kept_set or not kept:
            _kept_goal_indices.add(i)

    # 预览第 0 帧
    frame = read_frame(video_path, 0, total=total, fps=fps)
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None

    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")

    goal_choices = [f"{i+1}) {c['ts']:.1f}s" for i, c in enumerate(_last_goal_clips)]
    default_preview = goal_choices[0] if goal_choices else None
    first_clip = _last_goal_clips[0]["path"] if _last_goal_clips else None
    # CheckboxGroup 默认勾选保留的进球
    kept_choices = [goal_choices[i] for i in sorted(_kept_goal_indices)
                    if i < len(goal_choices)]

    status = (f"✅ 已加载历史记录\n"
              f"视频: {r.get('video_name', '')}\n"
              f"检测时间: {r.get('time', '')}\n"
              f"进球: {len(all_goals)} 个 | 保留: {len(kept)} 个\n"
              f"已生成 {len(_last_goal_clips)} 个预览片段\n"
              f"保留的进球时间: {', '.join([f'{t:.1f}s' for t in kept]) if kept else '全部'}\n"
              f"可预览片段或直接点「生成集锦」剪辑")

    progress(1.0, desc="加载完成")
    return (preview,
            gr.update(maximum=max(total - 1, 1), value=0),
            info_str,
            status,
            first_clip,
            "",
            gr.update(choices=goal_choices, value=default_preview),
            gr.update(choices=goal_choices, value=kept_choices))


# 加载 .env 文件中的 API Key
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


# ============ 全局状态 ============
_video_state = {"path": None, "total": 0, "fps": 30.0, "codec": "unknown"}
_calib = {
    "clicks": [],   # [(x,y), ...]
    "hoop": None,   # (x1,y1,x2,y2)
    "baseline_frame": None,  # 标定时的帧（作为基准帧差法的无球基准）
    "baseline_idx": -1,
}
# 保存最近一次检测的进球时间戳，供「生成集锦」使用
_last_goals = []
# 保存每个进球的独立片段路径，供 UI 逐个预览/保留/删除
# 格式: [{"ts": 5.1, "path": "E:/bball_cache/demo_output/goal_0_5s.mp4"}, ...]
_last_goal_clips = []
# 用户选择保留的进球索引（基于 _last_goal_clips）
_kept_goal_indices = set()

# ============ 批量识别状态 ============
# 扫描到的视频文件列表
_batch_files = []
# 每个视频的标定数据：{video_path: {"hoop": (x1,y1,x2,y2), "baseline_frame": ndarray, "baseline_idx": int}}
_batch_calibs = {}
# 当前在预览区加载的批量视频路径（用于区分单视频模式和批量标定模式）
_batch_current_video = None


def on_load_video(video_path):
    """加载视频（通过文件路径，避免上传大文件中断）。"""
    if not video_path or not video_path.strip():
        return None, gr.update(maximum=1, value=0), "请输入视频文件路径"
    video_path = video_path.strip().strip('"').strip("'")
    if not os.path.exists(video_path):
        return None, gr.update(maximum=1, value=0), f"❌ 文件不存在: {video_path}"
    try:
        info = get_video_info(video_path)
    except Exception as e:
        return None, gr.update(maximum=1, value=0), f"读取失败: {e}"
    _video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                        codec=info["codec"])
    _calib["clicks"] = []
    _calib["hoop"] = None
    frame = read_frame(video_path, 0, total=info["total"], fps=info["fps"])
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")
    return preview, gr.update(maximum=max(info["total"] - 1, 1), value=0), info_str


def on_load_path(path_input):
    """统一加载入口：自动判断是文件还是文件夹。

    文件 → 单视频模式，加载到预览区，隐藏批量组件
    文件夹 → 批量模式，扫描视频列表，显示批量组件
    返回: (preview_image, frame_slider, info_text, batch_group_visible,
           batch_video_selector, batch_calib_status, click_status,
           run_btn_visible, batch_detect_btn_visible)
    """
    global _batch_files, _batch_calibs, _batch_current_video
    if not path_input or not path_input.strip():
        return (None, gr.update(maximum=1, value=0), "请输入视频文件或文件夹路径",
                gr.update(visible=False), gr.update(choices=[], value=None), "", "",
                gr.update(visible=True), gr.update(visible=False))
    path_input = path_input.strip().strip('"').strip("'")
    if not os.path.exists(path_input):
        return (None, gr.update(maximum=1, value=0), f"❌ 路径不存在: {path_input}",
                gr.update(visible=False), gr.update(choices=[], value=None), "", "",
                gr.update(visible=True), gr.update(visible=False))

    # 文件夹 → 批量模式
    if os.path.isdir(path_input):
        files = _scan_video_files(path_input)
        if not files:
            return (None, gr.update(maximum=1, value=0),
                    f"❌ 文件夹未找到视频文件: {path_input}",
                    gr.update(visible=False), gr.update(choices=[], value=None), "", "",
                    gr.update(visible=True), gr.update(visible=False))
        _batch_files = files
        _batch_calibs = {}
        _batch_current_video = None
        choices = _batch_video_choices()
        n = len(files)
        preview_list = "\n".join([f"  [{i+1}] {os.path.basename(f)}" for i, f in enumerate(files)])
        info_str = (f"📂 批量模式 | {n} 个视频 | 已标定: 0/{n}\n"
                    f"请从下方列表选择视频标定篮筐\n{preview_list}")
        calib_status = f"共 {n} 个视频 | 已标定: 0/{n} | 请选择视频加载到预览区标定"
        return (None, gr.update(maximum=1, value=0), info_str,
                gr.update(visible=True), gr.update(choices=choices, value=choices[0] if choices else None),
                calib_status, "批量模式：请选择视频并标定篮筐",
                gr.update(visible=False), gr.update(visible=True))

    # 文件 → 单视频模式
    _batch_files = []
    _batch_calibs = {}
    _batch_current_video = None
    try:
        info = get_video_info(path_input)
    except Exception as e:
        return (None, gr.update(maximum=1, value=0), f"❌ 读取失败: {e}",
                gr.update(visible=False), gr.update(choices=[], value=None), "", "",
                gr.update(visible=True), gr.update(visible=False))
    _video_state.update(path=path_input, total=info["total"], fps=info["fps"],
                        codec=info["codec"])
    _calib["clicks"] = []
    _calib["hoop"] = None
    _calib["baseline_frame"] = None
    _calib["baseline_idx"] = -1
    frame = read_frame(path_input, 0, total=info["total"], fps=info["fps"])
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")
    return (preview, gr.update(maximum=max(info["total"] - 1, 1), value=0), info_str,
            gr.update(visible=False), gr.update(choices=[], value=None), "", "",
            gr.update(visible=True), gr.update(visible=False))


def on_preview(frame_idx):
    """预览指定帧。"""
    if _video_state["path"] is None:
        return None, "请先加载视频"
    frame = read_frame(_video_state["path"], int(frame_idx),
                      total=_video_state["total"], fps=_video_state["fps"])
    if frame is None:
        return None, "读取帧失败"
    # 画已标定的篮筐框
    out = frame.copy()
    if _calib["hoop"]:
        x1, y1, x2, y2 = _calib["hoop"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(out, "HOOP", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # 上沿/下沿线
        cv2.line(out, (x1 - 30, y1), (x2 + 30, y1), (0, 255, 255), 1)
        cv2.line(out, (x1 - 30, y2), (x2 + 30, y2), (255, 0, 255), 1)
    # 画待确认点击点
    for i, (x, y) in enumerate(_calib["clicks"]):
        cv2.circle(out, (x, y), 10, (255, 255, 0), -1)
        cv2.putText(out, str(i + 1), (x - 6, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    ts = int(frame_idx) / _video_state["fps"]
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), f"帧 {frame_idx} ({ts:.1f}s)"


def on_image_click(evt: gr.SelectData, frame_idx):
    """点击图片标定篮筐：2 个点形成一个框。"""
    if _video_state["path"] is None:
        return None, "请先加载视频"
    x, y = int(evt.index[0]), int(evt.index[1])
    _calib["clicks"].append((x, y))

    status = f"点击 ({x},{y})，已收集 {len(_calib['clicks'])}/2 个点"
    if len(_calib["clicks"]) >= 2:
        p1, p2 = _calib["clicks"][:2]
        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
        _calib["hoop"] = (x1, y1, x2, y2)
        # 保存当前帧作为基准帧差法的无球基准帧
        base_frame = read_frame(_video_state["path"], int(frame_idx),
                                total=_video_state["total"], fps=_video_state["fps"])
        if base_frame is not None:
            _calib["baseline_frame"] = base_frame.copy()
            _calib["baseline_idx"] = int(frame_idx)
        status = (f"✅ 篮筐已标定: ({x1},{y1}) - ({x2},{y2}) | "
                  f"基准帧: 第 {int(frame_idx)} 帧")
        _calib["clicks"] = []

    # 重新预览
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


def on_reset_hoop():
    """重置篮筐标定。"""
    _calib["clicks"] = []
    _calib["hoop"] = None
    _calib["baseline_frame"] = None
    _calib["baseline_idx"] = -1
    return "已重置，请重新点击 2 个点标定篮筐（基准帧也会重新采集）"


def on_run_detect(start_frame, end_frame, ball_conf, min_gap_sec, progress=gr.Progress()):
    """运行进球检测并生成可视化视频（diff + YOLO 双确认）。"""
    global _kept_goal_indices, _last_goal_clips, _last_goals
    if _video_state["path"] is None:
        return None, "❌ 请先加载视频"
    if _calib["hoop"] is None:
        return None, "❌ 请先点击画面标定篮筐（2 个点）"

    hoop = _calib["hoop"]
    fps = _video_state["fps"]
    total = _video_state["total"]
    start = int(start_frame)
    end = int(end_frame) if end_frame and int(end_frame) > 0 else total
    end = min(end, total)

    if end <= start:
        return None, "❌ 结束帧必须大于起始帧"

    if _calib["baseline_frame"] is None:
        return None, "❌ 基准帧差法需要基准帧，请重新点击 2 个点标定篮筐"

    try:
        # 初始化检测器：diff 宽松模式 + YOLO 双确认
        detector = GoalDetector(hoop, baseline_frame=_calib["baseline_frame"],
                                min_gap_sec=float(min_gap_sec),
                                diff_threshold=15, min_blob_area=30, search_margin=80,
                                fusion_mode="visual_only",
                                loose_mode=True,
                                yolo_confirm=True,
                                rolling_baseline_sec=60.0)
        method_name = "diff+YOLO双确认"

        # 加载 YOLO 模型（双确认 + 可视化都需要）
        progress(0, desc="加载 YOLO 模型...")
        model, weights_path = get_ball_model()

        # 输出目录改到 E 盘（避免 C 盘中文路径导致 URL 编码问题）
        out_dir = Path(_CACHE_ROOT) / "demo_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        # 用时间戳命名避免并发/重复运行时文件占用
        import time as _t
        _stamp = int(_t.time())

        progress(0.01, desc=f"开始检测 [{method_name}]...")
        t0 = time.time()
        processed = 0
        n_frames = end - start

        # 诊断统计
        diag = {"ball_detected": 0, "in_x_range": 0,
                "above": 0, "in_hoop": 0, "below": 0,
                "ball_x_min": 99999, "ball_x_max": 0,
                "ball_y_min": 99999, "ball_y_max": 0,
                "blob_detected": 0}

        # 第一阶段：纯检测（不写视频），记录进球时刻附近的帧索引
        # 长视频写完整可视化 mp4 会很大且转码超时，改为只写进球前后片段
        goal_clip_frames = set()  # 需要写入的帧索引
        clip_half = int(fps * 3)  # 进球前后各 3 秒

        reader = VideoReader(_video_state["path"])
        try:
            for fidx, frame in reader.iter_frames(start=start, end=end, batch=1):
                # YOLO 检测球（用于双确认）
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

                # 诊断统计（球位置）
                if ball_pos is not None:
                    diag["ball_detected"] += 1
                    cx, cy = ball_pos[0], ball_pos[1]
                    diag["ball_x_min"] = min(diag["ball_x_min"], cx)
                    diag["ball_x_max"] = max(diag["ball_x_max"], cx)
                    diag["ball_y_min"] = min(diag["ball_y_min"], cy)
                    diag["ball_y_max"] = max(diag["ball_y_max"], cy)
                    x_ok = (hoop[0] - int((hoop[2]-hoop[0])*0.5) <= cx
                            <= hoop[2] + int((hoop[2]-hoop[0])*0.5))
                    if x_ok:
                        diag["in_x_range"] += 1
                        if cy < hoop[1]:
                            diag["above"] += 1
                        elif cy > hoop[3]:
                            diag["below"] += 1
                        else:
                            diag["in_hoop"] += 1

                # 喂入检测器（diff + YOLO 双确认，需传 ball_pos）
                detector.feed(ball_pos, fidx, fps, frame=frame)
                blob = detector.last_blob_box
                if blob is not None:
                    diag["blob_detected"] += 1

                processed += 1
                if processed % 30 == 0:
                    progress(processed / n_frames * 0.7,
                             desc=f"检测中 {processed}/{n_frames} | 进球: {len(detector.goals)}")
        finally:
            reader.close()

        # 基准帧差法结束后调用 finalize（处理仅音频候选，这里无音频所以无影响）
        detector.finalize()

        elapsed = time.time() - t0

        # 第二阶段：为每个进球生成独立预览片段
        # 每个进球前后各 clip_half 秒，单独一个 mp4，方便用户逐个保留/删除
        goals = sorted(detector.goals)
        print(f"[DEBUG] 第二阶段开始: 检测到 {len(goals)} 个进球, goals={goals}", flush=True)
        _last_goal_clips.clear()
        _kept_goal_indices.clear()  # 重置保留标记，待用户重新选择

        import imageio_ffmpeg
        import subprocess as _sp
        ff = imageio_ffmpeg.get_ffmpeg_exe()

        for gi, gts in enumerate(goals):
            progress(0.7 + 0.25 * gi / max(len(goals), 1),
                     desc=f"生成片段 {gi+1}/{len(goals)} ({gts:.1f}s)...")
            gframe = int(gts * fps)
            seg_start = max(start, gframe - clip_half)
            seg_end = min(end, gframe + clip_half)
            if seg_end <= seg_start:
                continue

            # ffmpeg 直接切片 + 转码 H.264（单次完成，跳过逐帧读取和中间文件）
            # 预览片段降到 480p 加速生成（仅供人工确认用，集锦保持原画质）
            clip_path = str(out_dir / f"goal_{gi}_{int(gts)}s_{_stamp}.mp4")
            seg_start_sec = seg_start / fps
            seg_dur_sec = (seg_end - seg_start) / fps
            try:
                # NVENC 硬编加速预览片段生成（回退 libx264）
                _enc = _build_encode_args(ff, quality="preview")
                _sp.run([ff, "-y", "-loglevel", "error",
                         "-ss", f"{seg_start_sec:.3f}", "-i", _video_state["path"],
                         "-t", f"{seg_dur_sec:.3f}",
                         "-vf", "scale=-2:480"] + _enc +  # 480p，保持宽高比
                        ["-movflags", "+faststart", clip_path],
                        creationflags=0x08000000, capture_output=True, timeout=60)
                if not (os.path.exists(clip_path) and os.path.getsize(clip_path) > 0):
                    clip_path = None
            except Exception:
                clip_path = None

            if clip_path:
                _last_goal_clips.append({"ts": gts, "path": clip_path, "idx": gi})
                print(f"[DEBUG] 片段 {gi+1}/{len(goals)} 生成完成: {clip_path}", flush=True)

        # 默认全部保留（用户可后续取消）
        _kept_goal_indices = set(range(len(_last_goal_clips)))
        print(f"[DEBUG] 第二阶段完成: 共生成 {len(_last_goal_clips)} 个片段", flush=True)

        progress(1.0, desc="完成")
        # 保存进球时间戳供「生成集锦」使用
        _last_goals.clear()
        _last_goals.extend(detector.goals)
        goals_str = "\n".join([f"  [{i+1}] {ts:.1f}s (帧 {int(ts*fps)})"
                               for i, ts in enumerate(detector.goals)])
        if not goals_str:
            goals_str = "  (无)"
        # 诊断信息
        bx = f"[{diag['ball_x_min']:.0f}, {diag['ball_x_max']:.0f}]" if diag["ball_detected"] else "N/A"
        by = f"[{diag['ball_y_min']:.0f}, {diag['ball_y_max']:.0f}]" if diag["ball_detected"] else "N/A"
        d = detector.diag
        total_yolo_triggers = d['yolo_confirmed'] + d['yolo_rejected']
        confirm_rate = d['yolo_confirmed'] / max(total_yolo_triggers, 1) * 100
        extra = (f"运动斑块检测: {diag['blob_detected']}/{processed} 帧 "
                 f"({diag['blob_detected']/max(processed,1)*100:.0f}%)\n"
                 f"━━━ 滚动基准帧 ━━━\n"
                 f"  基准帧更新次数: {d['baseline_updates']} | "
                 f"间隔: {detector.rolling_baseline_sec:.0f}s\n"
                 f"  (更新次数=0 说明基准帧从未更新，长视频会因背景变化漏检)\n"
                 f"━━━ diff 拒绝原因 ━━━\n"
                 f"  到达上方: {d['cross_above']} | 在框内: {d['in_hoop']} | 到达下方: {d['cross_below']}\n"
                 f"  冷却跳过: {d['reject_cooldown']} | 无斑块: {d['reject_no_blob']}\n"
                 f"  无上方直接到下方: {d['reject_no_above']} | x不在范围: {d['reject_in_x']}\n"
                 f"  斑块太宽: {d['reject_size']} | 趋势失败: {d['reject_trend']}\n"
                 f"  侧向进筐成功: {d['side_goal']} | 超时匹配成功: {d['timeout_goal']}\n"
                 f"━━━ YOLO 双确认 ━━━\n"
                 f"  diff触发次数: {total_yolo_triggers} | "
                 f"YOLO确认: {d['yolo_confirmed']} | YOLO否决: {d['yolo_rejected']}\n"
                 f"  确认率: {confirm_rate:.0f}%\n")
        status = (f"✅ 检测完成 [{method_name}]\n"
                  f"处理: {processed} 帧 | 耗时: {elapsed:.0f}s | "
                  f"{processed/max(elapsed,0.1):.1f} fps\n"
                  f"篮筐框: x[{hoop[0]},{hoop[2]}] y[{hoop[1]},{hoop[3]}]\n"
                  f"检测到进球: {len(detector.goals)} 个\n{goals_str}\n"
                  f"━━━ 诊断 ━━━\n"
                  f"{extra}"
                  f"YOLO 检测到球: {diag['ball_detected']}/{processed} 帧 "
                  f"({diag['ball_detected']/max(processed,1)*100:.0f}%)\n"
                  f"球 x 范围: {bx} | 篮筐 x: [{hoop[0]},{hoop[2]}]\n"
                  f"球 y 范围: {by} | 篮筐 y: [{hoop[1]},{hoop[3]}]\n"
                  f"球在篮筐 x 范围内: {diag['in_x_range']} 帧\n"
                  f"  其中 ABOVE(上方): {diag['above']} | IN_HOOP(框内): {diag['in_hoop']} | "
                  f"BELOW(下方): {diag['below']}")
        # 返回：第一个片段路径（供预览）+ 保留列表更新 + Dropdown更新 + 状态文本
        first_clip = _last_goal_clips[0]["path"] if _last_goal_clips else None
        # 进球选项：["1) 5.1s", "2) 14.6s", ...]
        goal_choices = [f"{i+1}) {c['ts']:.1f}s" for i, c in enumerate(_last_goal_clips)]
        # 默认全选（全部保留）
        default_kept = goal_choices[:]
        # Dropdown 默认选第一个
        default_preview = goal_choices[0] if goal_choices else None
        # 保存到历史记录（初始全部进球作为 goals，kept_goals 暂为全部）
        _add_history(_video_state["path"], hoop,
                     detector.goals, detector.goals)
        return (first_clip,
                gr.update(choices=goal_choices, value=default_kept),  # CheckboxGroup
                gr.update(choices=goal_choices, value=default_preview),  # Dropdown
                status)
    except Exception as e:
        import traceback
        return (None,
                gr.update(choices=[], value=[]),
                gr.update(choices=[], value=None),
                f"❌ 检测失败: {e}\n\n{traceback.format_exc()}")


def on_select_goal_preview(selected):
    """用户在 CheckboxGroup 选择某个进球时，预览该片段。
    取第一个被选中的片段作为预览（Gradio CheckboxGroup 不支持单选）。
    这里用单独的 Dropdown 选择预览更合适，但为简化用第一个选中项。
    实际预览通过下方的 on_preview_goal_by_idx 实现。
    """
    if not selected or not _last_goal_clips:
        return None
    # 解析第一个选中项的索引
    try:
        idx = int(selected[0].split(")")[0]) - 1
        return _last_goal_clips[idx]["path"]
    except (IndexError, ValueError):
        return None


def on_preview_goal_by_idx(idx_str):
    """通过下拉框选择预览某个进球片段。"""
    if not _last_goal_clips or not idx_str:
        return None
    try:
        idx = int(idx_str.split(")")[0]) - 1
        if 0 <= idx < len(_last_goal_clips):
            return _last_goal_clips[idx]["path"]
    except (ValueError, IndexError):
        pass
    return None


def on_update_kept_goals(kept_choices):
    """更新用户保留的进球列表，并删除未保留的片段文件。"""
    global _kept_goal_indices, _last_goals
    if not _last_goal_clips:
        return "没有可保留的进球片段"

    # 解析保留的索引
    kept_indices = set()
    for c in kept_choices:
        try:
            kept_indices.add(int(c.split(")")[0]) - 1)
        except ValueError:
            continue

    # 删除未保留的片段文件
    deleted = 0
    for i, clip in enumerate(_last_goal_clips):
        if i not in kept_indices:
            try:
                os.remove(clip["path"])
                deleted += 1
            except Exception:
                pass

    # 更新全局状态
    _kept_goal_indices = kept_indices
    # 更新 _last_goals 供集锦使用
    kept_goals = [_last_goal_clips[i]["ts"] for i in sorted(kept_indices)
                  if i < len(_last_goal_clips)]
    _last_goals.clear()
    _last_goals.extend(kept_goals)

    total = len(_last_goal_clips)
    kept = len(kept_indices)
    # 更新历史记录中的 kept_goals
    if _video_state["path"]:
        _add_history(_video_state["path"], _calib["hoop"],
                     [c["ts"] for c in _last_goal_clips], kept_goals)
    return f"✅ 保留 {kept}/{total} 个进球 | 已删除 {deleted} 个片段 | 保留的进球时间: {', '.join([f'{t:.1f}s' for t in kept_goals])}"


def on_generate_highlights(pre_roll, post_roll, min_gap, progress=gr.Progress()):
    """根据检测到的进球时间戳生成集锦视频。"""
    if _video_state["path"] is None:
        return None, "❌ 请先加载视频并检测进球"
    if not _last_goals:
        return None, "❌ 没有检测到进球，无法生成集锦（请先点「开始检测」）"

    progress(0.05, desc=f"开始剪辑（{len(_last_goals)} 个进球）...")
    try:
        out_path = cut_clips(
            _video_state["path"], list(_last_goals),
            pre_roll=int(pre_roll), post_roll=int(post_roll),
            min_gap=int(min_gap),
        )
        if out_path and os.path.exists(out_path):
            progress(1.0, desc="集锦生成完成")
            n = len(_last_goals)
            dur = sum(int(post_roll) + int(pre_roll) for _ in _last_goals)
            return out_path, (f"✅ 集锦已生成（{n} 个进球片段）\n"
                              f"前保留 {int(pre_roll)}s | 后保留 {int(post_roll)}s | "
                              f"最小合并间隔 {int(min_gap)}s\n"
                              f"输出: {out_path}")
        else:
            return None, "❌ 集锦生成失败（cut_clips 返回空）"
    except Exception as e:
        import traceback
        return None, f"❌ 剪辑失败: {e}\n\n{traceback.format_exc()}"


# ============ 批量文件夹识别 ============
# 支持的视频扩展名
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ts"}


def _scan_video_files(folder):
    """扫描文件夹中的视频文件，按文件名排序。返回绝对路径列表。"""
    folder = folder.strip().strip('"').strip("'")
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in _VIDEO_EXTS:
            files.append(os.path.join(folder, name))
    # 按文件名自然排序
    import re
    def _natural_key(s):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', s)]
    files.sort(key=_natural_key)
    return files


def _batch_video_choices():
    """生成批量视频下拉选项，标注已标定/未标定状态。"""
    global _batch_files, _batch_calibs
    choices = []
    for i, f in enumerate(_batch_files):
        name = os.path.basename(f)
        mark = "✅" if f in _batch_calibs else "⬜"
        choices.append(f"{mark} [{i+1}] {name}")
    return choices


def on_batch_load_video(selected):
    """批量模式：加载选中的视频到预览区（复用主预览区）。"""
    global _batch_current_video, _calib
    if not selected or not _batch_files:
        return None, gr.update(maximum=1, value=0), "", "请先扫描文件夹并选择视频"
    # 解析索引
    try:
        # 格式: "✅ [1] xxx.mp4" 或 "⬜ [1] xxx.mp4"
        idx_str = selected.split("]")[0].split("[")[1]
        idx = int(idx_str) - 1
    except (ValueError, IndexError):
        return None, gr.update(maximum=1, value=0), "", "解析视频索引失败"
    if idx < 0 or idx >= len(_batch_files):
        return None, gr.update(maximum=1, value=0), "", "视频索引超出范围"

    video_path = _batch_files[idx]
    _batch_current_video = video_path

    # 如果该视频已标定，恢复标定到 _calib；否则重置
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

    # 加载视频到 _video_state 和预览区
    try:
        info = get_video_info(video_path)
    except Exception as e:
        return None, gr.update(maximum=1, value=0), "", f"读取失败: {e}"
    _video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                        codec=info["codec"])
    frame = read_frame(video_path, 0, total=info["total"], fps=info["fps"])
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    # 画已标定的篮筐框（如有）
    if preview is not None and _calib["hoop"]:
        x1, y1, x2, y2 = _calib["hoop"]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(preview, "HOOP", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")
    status = (f"已加载: {os.path.basename(video_path)}\n"
              f"{'✅ 已标定' if video_path in _batch_calibs else '⬜ 未标定，请滑动到篮筐画面点击 2 个点标定'}")
    return preview, gr.update(maximum=max(info["total"] - 1, 1), value=0), info_str, status


def on_batch_save_calib():
    """保存当前 _calib（左侧标定的篮筐）到当前批量视频。"""
    global _batch_calibs
    if _batch_current_video is None:
        return "❌ 请先从列表选择视频并加载到预览区", gr.update(), "请先选择视频"
    if _calib["hoop"] is None or _calib["baseline_frame"] is None:
        return "❌ 请先在预览区点击 2 个点标定篮筐", gr.update(), "请先标定篮筐"

    _batch_calibs[_batch_current_video] = {
        "hoop": _calib["hoop"],
        "baseline_frame": _calib["baseline_frame"].copy(),
        "baseline_idx": _calib["baseline_idx"],
    }
    n_calib = len(_batch_calibs)
    n_total = len(_batch_files)
    # 刷新下拉列表（更新标定标记）
    choices = _batch_video_choices()
    # 保持当前选择
    cur_idx = _batch_files.index(_batch_current_video) if _batch_current_video in _batch_files else 0
    cur_value = choices[cur_idx] if choices else None
    status = f"✅ 已保存标定: {os.path.basename(_batch_current_video)} | 已标定: {n_calib}/{n_total}"
    if n_calib < n_total:
        status += f"\n请继续选择下一个视频标定（剩余 {n_total - n_calib} 个）"
    else:
        status += "\n🎉 全部标定完成！可点击「批量识别」开始检测"
    return status, gr.update(choices=choices, value=cur_value), status


def on_batch_detect(ball_conf, min_gap_sec, progress=gr.Progress()):
    """批量识别：每个视频用各自的标定数据检测，单独保存历史记录。"""
    global _batch_files, _batch_calibs
    if not _batch_files:
        return "❌ 请先扫描文件夹", gr.update(choices=_history_choices())
    n_calib = len(_batch_calibs)
    n_total = len(_batch_files)
    if n_calib == 0:
        return "❌ 请先逐个标定篮筐（至少标定 1 个视频）", gr.update(choices=_history_choices())

    # 提示未标定的视频
    uncalib = [os.path.basename(f) for f in _batch_files if f not in _batch_calibs]
    warning = ""
    if uncalib:
        warning = f"⚠️ {len(uncalib)} 个视频未标定，将跳过: {', '.join(uncalib[:5])}\n\n"

    progress(0, desc="加载 YOLO 模型...")
    try:
        model, _ = get_ball_model()
    except Exception as e:
        return f"❌ YOLO 模型加载失败: {e}", gr.update(choices=_history_choices())

    results = []
    total_to_process = len(_batch_calibs)  # 只处理已标定的
    t_batch_start = time.time()
    processed_count = 0

    for vi, video_path in enumerate(_batch_files):
        vname = os.path.basename(video_path)
        if video_path not in _batch_calibs:
            results.append(f"[{vi+1}] {vname} ⏭️ 跳过（未标定）")
            continue

        processed_count += 1
        cal = _batch_calibs[video_path]
        hoop = cal["hoop"]
        baseline_frame = cal["baseline_frame"]

        progress(0.05 + 0.9 * (processed_count - 1) / max(total_to_process, 1),
                 desc=f"[{processed_count}/{total_to_process}] {vname}")

        # 读取视频信息
        try:
            info = get_video_info(video_path)
        except Exception as e:
            results.append(f"[{vi+1}] {vname} ❌ 读取失败: {e}")
            continue

        fps = info["fps"]
        total = info["total"]

        # 初始化检测器（用该视频的标定数据：篮筐 + 基准帧）
        detector = GoalDetector(hoop, baseline_frame=baseline_frame,
                                min_gap_sec=float(min_gap_sec),
                                diff_threshold=15, min_blob_area=30, search_margin=80,
                                fusion_mode="visual_only",
                                loose_mode=True,
                                yolo_confirm=True,
                                rolling_baseline_sec=60.0)

        # 逐帧检测
        t0 = time.time()
        processed = 0
        try:
            reader = VideoReader(video_path)
            try:
                for fidx, frame in reader.iter_frames(start=0, end=total, batch=1):
                    # YOLO 检测球
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
                    if processed % 60 == 0:
                        sub_prog = ((processed_count - 1) + processed / max(total, 1)) / max(total_to_process, 1)
                        progress(0.05 + 0.9 * sub_prog,
                                 desc=f"[{processed_count}/{total_to_process}] {vname} {processed}/{total}帧 进球:{len(detector.goals)}")
            finally:
                reader.close()
        except Exception as e:
            results.append(f"[{vi+1}] {vname} ❌ 检测异常: {e}")
            continue

        detector.finalize()
        elapsed = time.time() - t0
        d = detector.diag
        total_yolo_triggers = d['yolo_confirmed'] + d['yolo_rejected']

        # 保存历史记录（每个视频单独一条，初始 kept_goals = goals）
        _add_history(video_path, hoop, detector.goals, detector.goals)

        n_goals = len(detector.goals)
        goals_str = ", ".join([f"{t:.1f}s" for t in detector.goals[:8]])
        if n_goals > 8:
            goals_str += f" ...(+{n_goals-8})"
        line = (f"[{vi+1}] {vname}\n"
                f"    进球: {n_goals} 个 | 帧数: {processed} | 耗时: {elapsed:.0f}s | "
                f"{processed/max(elapsed,0.1):.1f}fps\n"
                f"    基准帧更新: {d['baseline_updates']} | "
                f"斑块检测: {processed - d['reject_no_blob']}/{processed}帧有斑块\n"
                f"    YOLO确认: {d['yolo_confirmed']}/{total_yolo_triggers} | "
                f"进球时间: {goals_str if goals_str else '无'}")
        results.append(line)

    total_elapsed = time.time() - t_batch_start
    # 统计进球数
    total_goals = 0
    success_count = 0
    for r in results:
        if "❌" not in r and "⏭️" not in r:
            success_count += 1
            for part in r.split("|"):
                if "进球:" in part:
                    try:
                        total_goals += int(part.split("进球:")[1].split("个")[0].strip())
                    except (ValueError, IndexError):
                        pass
                    break

    summary = (f"{warning}"
               f"━━━ 批量检测完成 ━━━\n"
               f"视频总数: {n_total} | 已标定: {total_to_process} | 成功: {success_count}\n"
               f"总进球数: {total_goals} | 总耗时: {total_elapsed:.0f}s\n"
               f"━━━ 各视频详情 ━━━\n" + "\n".join(results))

    progress(1.0, desc="批量检测完成")
    return summary, gr.update(choices=_history_choices())


# ============ Gradio 界面 ============
# 预填已知视频路径
_DEFAULT_VIDEO = r"D:\Downloads\highlights.mp4"

# 自定义主题
_custom_theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Noto Sans SC"), "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="linear-gradient(135deg, #f97316 0%, #ea580c 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #fb923c 0%, #f97316 100%)",
    button_primary_text_color="white",
    block_title_text_weight="600",
    block_label_text_weight="500",
)

# 自定义 CSS
_custom_css = """
.gradio-container { max-width: 1400px !important; padding: 16px !important; }
h1 { text-align: center; margin-bottom: 4px !important; }
.step-badge {
    display: inline-block; background: #f97316; color: white;
    width: 26px; height: 26px; line-height: 26px; border-radius: 50%;
    text-align: center; font-weight: bold; margin-right: 6px; font-size: 14px;
}
.section-card { padding: 4px 0; }
.status-box textarea { font-family: 'Consolas', 'SF Mono', monospace !important; font-size: 12px !important; }
"""

with gr.Blocks(title="篮球进球检测") as demo:
    gr.Markdown("# 🏀 篮球进球检测与自动剪辑")
    gr.Markdown("<p style='text-align:center;color:#64748b;'>输入视频文件或文件夹路径 → 滑动到篮筐画面 → 点击 2 点标定 → 开始检测</p>")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=480):
            with gr.Group():
                gr.Markdown("### 📥 加载视频")
                video_path_input = gr.Textbox(
                    label="视频文件或文件夹路径",
                    value=_DEFAULT_VIDEO,
                    placeholder=r"文件: D:\xxx.mp4 | 文件夹: D:\Videos\basketball",
                    lines=1)
                load_btn = gr.Button("📥 加载", variant="primary")
                info_text = gr.Textbox(label="视频信息", interactive=False, lines=3,
                                       elem_classes="status-box")

            with gr.Group(visible=False) as batch_group:
                gr.Markdown("### 📁 批量标定")
                batch_video_selector = gr.Dropdown(
                    label="视频列表（✅=已标定 ⬜=未标定）",
                    choices=[],
                    value=None,
                    interactive=True,
                    info="选择视频 → 加载到预览区 → 点击 2 点标定 → 保存标定")
                with gr.Row():
                    batch_load_btn = gr.Button("📥 加载到预览区", size="sm")
                    batch_save_calib_btn = gr.Button("💾 保存标定", variant="primary", size="sm")
                batch_calib_status = gr.Textbox(label="标定进度", interactive=False, lines=2,
                                                elem_classes="status-box")

            with gr.Group():
                gr.Markdown("### 🎯 篮筐标定")
                frame_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1,
                                         label="帧选择器（滑动到篮筐画面）")
                preview_image = gr.Image(label="点击画面 2 个点标定篮筐（左上 + 右下）",
                                         type="numpy", interactive=False, height=320)
                with gr.Row():
                    click_status = gr.Textbox(label="标定状态", interactive=False,
                                              scale=3, elem_classes="status-box")
                    reset_btn = gr.Button("🔄 重置", size="sm", scale=1)

            with gr.Group():
                gr.Markdown("### ⚙️ 检测参数")
                with gr.Row():
                    start_frame = gr.Number(label="起始帧", value=0, precision=0)
                    end_frame = gr.Number(label="结束帧 (0=到末尾)", value=0, precision=0)
                with gr.Row():
                    ball_conf = gr.Slider(0.1, 0.9, value=0.3, step=0.05,
                                          label="球检测置信度")
                    min_gap = gr.Slider(1.0, 10.0, value=3.0, step=0.5,
                                        label="最小进球间隔(秒)")

                run_btn = gr.Button("🚀 开始检测", variant="primary")
                batch_detect_btn = gr.Button("🚀 批量识别（检测所有已标定视频）",
                                             variant="primary", visible=False)

        with gr.Column(scale=1, min_width=480):
            with gr.Group():
                gr.Markdown("### 🎬 进球片段预览")
                with gr.Row():
                    goal_selector = gr.Dropdown(
                        label="选择进球片段",
                        choices=[],
                        value=None,
                        interactive=True,
                        scale=4)
                    refresh_preview_btn = gr.Button("▶️ 预览", size="sm", scale=1)
                result_video = gr.Video(label="片段预览", height=280)
                gr.Markdown("<small>勾选要保留的进球，取消勾选要删除的，然后点「确认保留」</small>")
                kept_goals = gr.CheckboxGroup(
                    label="保留/删除进球（勾选=保留）",
                    choices=[],
                    value=[],
                    interactive=True)
                with gr.Row():
                    confirm_kept_btn = gr.Button("✅ 确认保留", variant="primary")
                    keep_all_btn = gr.Button("全选", size="sm")
                    clear_all_btn = gr.Button("全不选", size="sm")
                kept_status = gr.Textbox(label="保留状态", interactive=False, lines=2,
                                         elem_classes="status-box")
                result_status = gr.Textbox(label="检测统计", interactive=False, lines=10,
                                           elem_classes="status-box")

            with gr.Group():
                gr.Markdown("### ✂️ 自动剪辑集锦")
                with gr.Row():
                    pre_roll = gr.Number(label="进球前(秒)", value=5, precision=0)
                    post_roll = gr.Number(label="进球后(秒)", value=5, precision=0)
                    cut_min_gap = gr.Number(label="合并间隔(秒)", value=8, precision=0)
                cut_btn = gr.Button("✂️ 生成集锦", variant="primary")
                highlights_video = gr.Video(label="集锦视频", height=240)
                highlights_status = gr.Textbox(label="剪辑状态", interactive=False, lines=3,
                                               elem_classes="status-box")

            with gr.Group():
                gr.Markdown("### 📂 历史记录")
                history_selector = gr.Dropdown(
                    label="选择历史记录",
                    choices=_history_choices(),
                    value=None,
                    interactive=True,
                    info="[序号] 视频名 | 保留数/总数 | 时间")
                with gr.Row():
                    load_history_btn = gr.Button("📂 加载历史", variant="primary")
                    refresh_history_btn = gr.Button("🔄 刷新", size="sm")
                history_status = gr.Textbox(label="历史记录状态", interactive=False, lines=4,
                                            elem_classes="status-box")

    # 事件绑定
    # 统一加载入口（自动判断文件/文件夹）
    load_btn.click(on_load_path, inputs=[video_path_input],
                   outputs=[preview_image, frame_slider, info_text,
                            batch_group, batch_video_selector,
                            batch_calib_status, click_status,
                            run_btn, batch_detect_btn])
    frame_slider.change(on_preview, inputs=[frame_slider],
                        outputs=[preview_image, click_status])
    preview_image.select(on_image_click, inputs=[frame_slider],
                         outputs=[preview_image, click_status])
    reset_btn.click(on_reset_hoop, outputs=[click_status])
    # 单视频检测
    run_btn.click(on_run_detect,
                  inputs=[start_frame, end_frame, ball_conf, min_gap],
                  outputs=[result_video, kept_goals, goal_selector, result_status])
    # 批量检测
    batch_detect_btn.click(on_batch_detect,
                           inputs=[ball_conf, min_gap],
                           outputs=[result_status, history_selector])
    # 下拉选择预览某个进球片段
    goal_selector.change(on_preview_goal_by_idx, inputs=[goal_selector],
                         outputs=[result_video])
    refresh_preview_btn.click(on_preview_goal_by_idx, inputs=[goal_selector],
                              outputs=[result_video])
    # 全选/全不选
    keep_all_btn.click(lambda: gr.update(value=[c for c in kept_goals.choices]),
                       outputs=[kept_goals])
    clear_all_btn.click(lambda: gr.update(value=[]), outputs=[kept_goals])
    # 确认保留
    confirm_kept_btn.click(on_update_kept_goals, inputs=[kept_goals],
                           outputs=[kept_status])
    cut_btn.click(on_generate_highlights,
                  inputs=[pre_roll, post_roll, cut_min_gap],
                  outputs=[highlights_video, highlights_status])
    # 批量模式：加载视频到预览区 + 保存标定
    batch_load_btn.click(on_batch_load_video, inputs=[batch_video_selector],
                         outputs=[preview_image, frame_slider, info_text, click_status])
    batch_save_calib_btn.click(on_batch_save_calib,
                               outputs=[batch_calib_status, batch_video_selector, click_status])
    # 历史记录
    load_history_btn.click(on_load_history,
                           inputs=[history_selector, pre_roll, post_roll, cut_min_gap],
                           outputs=[preview_image, frame_slider, info_text,
                                    history_status, result_video, kept_status,
                                    goal_selector, kept_goals])
    refresh_history_btn.click(
        lambda: gr.update(choices=_history_choices()),
        outputs=[history_selector])


if __name__ == "__main__":
    _out_dir = str(Path(_CACHE_ROOT) / "demo_output")
    _gradio_dir = str(Path(_CACHE_ROOT) / "gradio")
    os.makedirs(_out_dir, exist_ok=True)
    os.makedirs(_gradio_dir, exist_ok=True)
    demo.queue(default_concurrency_limit=1)  # Gradio 6.x 需显式启用队列，SSE 才能正常工作
    demo.launch(server_name="127.0.0.1", server_port=7871,
                show_error=True, prevent_thread_lock=False,
                max_file_size=5000*1024*1024,  # 5GB 上限
                allowed_paths=[_CACHE_ROOT],  # 允许整个 bball_cache（含 gradio 会话子目录）
                theme=_custom_theme, css=_custom_css)
