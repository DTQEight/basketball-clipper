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
                        codec=info["codec"], current_frame=0)

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
            progress(0.2 + 0.7 * gi / max(len(goals_to_use), 1))
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

    checkbox_choices = [f"{i+1}) {c['ts']:.1f}s" for i, c in enumerate(_last_goal_clips)]
    # Radio 选项带保留标记（✅=保留 ❌=未保留），点击即预览
    radio_choices = [f"{i+1}) {c['ts']:.1f}s {'✅' if i in _kept_goal_indices else '❌'}"
                     for i, c in enumerate(_last_goal_clips)]
    default_radio = radio_choices[0] if radio_choices else None
    first_clip = _last_goal_clips[0]["path"] if _last_goal_clips else None
    # CheckboxGroup 默认勾选保留的进球
    kept_choices = [checkbox_choices[i] for i in sorted(_kept_goal_indices)
                    if i < len(checkbox_choices)]

    status = (f"✅ 已加载历史记录\n"
              f"视频: {r.get('video_name', '')}\n"
              f"检测时间: {r.get('time', '')}\n"
              f"进球: {len(all_goals)} 个 | 保留: {len(kept)} 个\n"
              f"已生成 {len(_last_goal_clips)} 个预览片段\n"
              f"保留的进球时间: {', '.join([f'{t:.1f}s' for t in kept]) if kept else '全部'}\n"
              f"可预览片段或直接点「生成集锦」剪辑")

    progress(1.0)
    return (preview,
            info_str,
            status,
            first_clip,
            "",
            gr.update(choices=radio_choices, value=default_radio),
            gr.update(choices=checkbox_choices, value=kept_choices))


# 加载 .env 文件中的 API Key
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


# ============ 全局状态 ============
_video_state = {"path": None, "total": 0, "fps": 30.0, "codec": "unknown", "current_frame": 0}
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
# 每个进球片段的分类标签（与 _last_goal_clips 同步）
_last_goal_types = []
# 可选标签列表
_TAG_OPTIONS = ["进球", "三分", "扣篮", "盖帽", "其他"]

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
        return None, "请输入视频文件路径"
    video_path = video_path.strip().strip('"').strip("'")
    if not os.path.exists(video_path):
        return None, f"❌ 文件不存在: {video_path}"
    try:
        info = get_video_info(video_path)
    except Exception as e:
        return None, f"读取失败: {e}"
    _video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                        codec=info["codec"], current_frame=0)
    _calib["clicks"] = []
    _calib["hoop"] = None
    frame = read_frame(video_path, 0, total=info["total"], fps=info["fps"])
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")
    return preview, info_str


def on_load_path(path_input):
    """统一加载入口：自动判断是文件还是文件夹。

    文件 → 单视频模式，加载到预览区，隐藏批量组件
    文件夹 → 批量模式，扫描视频列表，显示批量组件
    返回: (preview_image, info_text, batch_group_visible,
           batch_video_selector, batch_calib_status, click_status,
           run_btn_visible, batch_detect_btn_visible,
           result_cards, result_status)
    """
    _empty_cards = '<div style="color:#888;padding:8px;">暂无检测结果</div>'
    global _batch_files, _batch_calibs, _batch_current_video
    if not path_input or not path_input.strip():
        return (None, "请输入视频文件或文件夹路径",
                gr.update(visible=False), gr.update(choices=[], value=None), "", "",
                gr.update(visible=True), gr.update(visible=False),
                _empty_cards, "")
    path_input = path_input.strip().strip('"').strip("'")
    if not os.path.exists(path_input):
        return (None, f"❌ 路径不存在: {path_input}",
                gr.update(visible=False), gr.update(choices=[], value=None), "", "",
                gr.update(visible=True), gr.update(visible=False),
                _empty_cards, "")

    # 文件夹 → 批量模式
    if os.path.isdir(path_input):
        files = _scan_video_files(path_input)
        if not files:
            return (None,
                f"❌ 文件夹未找到视频文件: {path_input}",
                gr.update(visible=False), gr.update(choices=[], value=None), "", "",
                gr.update(visible=True), gr.update(visible=False),
                _empty_cards, "")
        _batch_files = files
        _batch_calibs = {}
        _batch_current_video = None
        choices = _batch_video_choices()
        n = len(files)
        preview_list = "\n".join([f"  [{i+1}] {os.path.basename(f)}" for i, f in enumerate(files)])
        info_str = (f"📂 批量模式 | {n} 个视频 | 已标定: 0/{n}\n"
                    f"请从下方列表选择视频标定篮筐\n{preview_list}")
        calib_status = f"共 {n} 个视频 | 已标定: 0/{n} | 请选择视频加载到预览区标定"
        return (None, info_str,
                gr.update(visible=True), gr.update(choices=choices, value=choices[0] if choices else None),
                calib_status, "批量模式：请选择视频并标定篮筐",
                gr.update(visible=False), gr.update(visible=True),
                _empty_cards, "")

    # 文件 → 单视频模式
    _batch_files = []
    _batch_calibs = {}
    _batch_current_video = None
    try:
        info = get_video_info(path_input)
    except Exception as e:
        return (None, f"❌ 读取失败: {e}",
                gr.update(visible=False), gr.update(choices=[], value=None), "", "",
                gr.update(visible=True), gr.update(visible=False),
                _empty_cards, "")
    _video_state.update(path=path_input, total=info["total"], fps=info["fps"],
                        codec=info["codec"], current_frame=0)
    _calib["clicks"] = []
    _calib["hoop"] = None
    _calib["baseline_frame"] = None
    _calib["baseline_idx"] = -1
    frame = read_frame(path_input, 0, total=info["total"], fps=info["fps"])
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")
    return (preview, info_str,
            gr.update(visible=False), gr.update(choices=[], value=None), "", "",
            gr.update(visible=True), gr.update(visible=False),
            _empty_cards, "")


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


def on_image_click(evt: gr.SelectData):
    """点击图片标定篮筐：2 个点形成一个框。"""
    if _video_state["path"] is None:
        return None, "请先加载视频"
    frame_idx = _video_state["current_frame"]
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


def on_run_detect(start_frame, end_frame, ball_conf, min_gap_sec,
                  diff_threshold=15, min_circularity=0.35, min_in_hoop_frames=2,
                  min_blob_area=30, search_margin=80, progress=gr.Progress()):
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
        # 高级参数由 UI 传入，支持运行时调整无需改代码
        detector = GoalDetector(hoop, baseline_frame=_calib["baseline_frame"],
                                min_gap_sec=float(min_gap_sec),
                                diff_threshold=int(diff_threshold),
                                min_blob_area=int(min_blob_area),
                                search_margin=int(search_margin),
                                fusion_mode="visual_only",
                                loose_mode=True,
                                yolo_confirm=True,
                                rolling_baseline_sec=60.0,
                                min_circularity=float(min_circularity),
                                min_in_hoop_frames=int(min_in_hoop_frames))
        method_name = "diff+YOLO双确认"

        # 加载 YOLO 模型（双确认 + 可视化都需要）
        progress(0)
        model, weights_path = get_ball_model()

        # 输出目录改到 E 盘（避免 C 盘中文路径导致 URL 编码问题）
        out_dir = Path(_CACHE_ROOT) / "demo_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        # 用时间戳命名避免并发/重复运行时文件占用
        import time as _t
        _stamp = int(_t.time())

        progress(0.01)
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
            progress(0.7 + 0.25 * gi / max(len(goals), 1))
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

        progress(1.0)
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
                 f"  斑块太宽: {d['reject_size']} | 形状过滤(疑似人): {d['reject_shape']} | 趋势失败: {d['reject_trend']}\n"
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
                  f"━━━ 当前参数 ━━━\n"
                  f"  帧差阈值: {diff_threshold} | 圆形度: {min_circularity:.2f} | "
                  f"进框帧数: {min_in_hoop_frames}\n"
                  f"  最小斑块: {min_blob_area} | 搜索范围: {search_margin}px\n"
                  f"━━━ 诊断 ━━━\n"
                  f"{extra}"
                  f"YOLO 检测到球: {diag['ball_detected']}/{processed} 帧 "
                  f"({diag['ball_detected']/max(processed,1)*100:.0f}%)\n"
                  f"球 x 范围: {bx} | 篮筐 x: [{hoop[0]},{hoop[2]}]\n"
                  f"球 y 范围: {by} | 篮筐 y: [{hoop[1]},{hoop[3]}]\n"
                  f"球在篮筐 x 范围内: {diag['in_x_range']} 帧\n"
                  f"  其中 ABOVE(上方): {diag['above']} | IN_HOOP(框内): {diag['in_hoop']} | "
                  f"BELOW(下方): {diag['below']}")
        # 返回：第一个片段路径 + 结果卡片 + 状态文本
        first_clip = _last_goal_clips[0]["path"] if _last_goal_clips else None
        cards_html = _build_result_cards_html()
        # 保存到历史记录（初始全部进球作为 goals，kept_goals 暂为全部）
        _add_history(_video_state["path"], hoop,
                     detector.goals, detector.goals)
        return (first_clip,
                cards_html,
                status)
    except Exception as e:
        import traceback
        return (None,
                '<div style="color:#888;padding:8px;">暂无检测结果</div>',
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


def _build_result_cards_html():
    """构建卡片式结果列表 HTML，每个片段独立卡片带操作按钮。"""
    if not _last_goal_clips:
        return '<div style="color:#888;padding:8px;">暂无检测结果</div>'
    cards = []
    for i, clip in enumerate(_last_goal_clips):
        ts = clip["ts"]
        t_min, t_sec = int(ts // 60), ts % 60
        start = max(ts - 10, 0)
        end = min(ts + 10, _video_state.get("total", 0) / max(_video_state.get("fps", 1), 1))
        s_min, s_sec = int(start // 60), start % 60
        e_min, e_sec = int(end // 60), end % 60
        type_val = _last_goal_types[i] if i < len(_last_goal_types) and _last_goal_types[i] else "进球"
        is_new = i >= len(_last_goal_clips) - 3
        marker = '<span style="color:#f44;font-size:11px;margin-left:8px;">🔴新片段</span>' if is_new else ''
        cards.append(
            f'<div style="background:#1e1e2a;border-radius:10px;padding:10px 12px;margin-bottom:8px;'
            f'box-shadow:0 2px 8px rgba(0,0,0,0.3);border:1px solid #2a2a3a;">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
            f'<span style="background:#FFB320;color:#000;font-weight:800;font-size:13px;'
            f'width:22px;height:22px;display:flex;align-items:center;justify-content:center;'
            f'border-radius:6px;">{i+1}</span>'
            f'<span style="color:#eee;font-size:14px;font-weight:600;">'
            f'{s_min}:{s_sec:04.1f} ~ {e_min}:{e_sec:04.1f}</span>'
            f'<span style="color:#aaa;font-size:12px;">进球点 {t_min}:{t_sec:04.1f}</span>'
            f'{marker}'
            f'<span style="color:#888;font-size:11px;margin-left:auto;">{type_val}</span>'
            f'</div>'
            f'<div style="display:flex;gap:6px;">'
            f'<button onclick="clipAction(\'preview\',{i})" style="flex:1;padding:5px 0;border:none;border-radius:6px;'
            f'background:#2d4a3e;color:#4ade80;font-size:12px;cursor:pointer;">▶ 预览</button>'
            f'<button onclick="clipAction(\'export\',{i})" style="flex:1;padding:5px 0;border:none;border-radius:6px;'
            f'background:#2d3a4a;color:#60a5fa;font-size:12px;cursor:pointer;">↗ 导出</button>'
            f'<button onclick="clipAction(\'delete\',{i})" style="flex:1;padding:5px 0;border:none;border-radius:6px;'
            f'background:#4a2d2d;color:#f87171;font-size:12px;cursor:pointer;">✕ 删除</button>'
            f'</div></div>'
        )
    return '<div style="display:flex;flex-direction:column;gap:0;">' + ''.join(cards) + '</div>'


def on_keep_all():
    """全选：保留所有片段。"""
    global _kept_goal_indices, _last_goals
    if not _last_goal_clips:
        return ""
    _kept_goal_indices = set(range(len(_last_goal_clips)))
    kept_ts = [c["ts"] for c in _last_goal_clips]
    _last_goals.clear()
    _last_goals.extend(kept_ts)
    total = len(_last_goal_clips)
    return f"✅ 全选 {total} 个片段"


def on_clear_all():
    """全不选：取消所有保留。"""
    global _kept_goal_indices, _last_goals
    _kept_goal_indices = set()
    _last_goals.clear()
    return "❌ 已取消全部保留"


def on_clip_action(raw_val):
    """处理卡片按钮点击：preview_0_1, export_1_2, delete_2_3 等（末尾为计数器）。"""
    if not raw_val or "_" not in raw_val:
        return gr.update(), gr.update(), ""
    parts = raw_val.split("_")
    if len(parts) < 2:
        return gr.update(), gr.update(), ""
    action = parts[0]
    idx_str = parts[1]
    try:
        idx = int(idx_str)
    except ValueError:
        return gr.update(), gr.update(), ""
    if idx < 0 or idx >= len(_last_goal_clips):
        return gr.update(), gr.update(), ""

    if action == "preview":
        path = _last_goal_clips[idx]["path"]
        return path, gr.update(), f"▶️ 正在预览第 {idx + 1} 个片段"
    elif action == "export":
        path = _last_goal_clips[idx]["path"]
        return gr.update(), gr.update(), f"✅ 已导出: {path}"
    elif action == "delete":
        global _kept_goal_indices, _last_goals
        deleted_ts = _last_goal_clips[idx]["ts"]
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
        cards_html = _build_result_cards_html()
        return None, cards_html, f"✅ 已删除第 {idx + 1} 个片段（{deleted_ts:.1f}s）| 剩余 {len(_last_goal_clips)} 个"
    return gr.update(), gr.update(), ""


def on_generate_highlights(pre_roll, post_roll, min_gap, progress=gr.Progress()):
    """根据检测到的进球时间戳生成集锦视频。"""
    if _video_state["path"] is None:
        return None, "❌ 请先加载视频并检测进球"
    if not _last_goals:
        return None, "❌ 没有检测到进球，无法生成集锦（请先点「开始检测」）"

    progress(0.05)
    try:
        out_path = cut_clips(
            _video_state["path"], list(_last_goals),
            pre_roll=int(pre_roll), post_roll=int(post_roll),
            min_gap=int(min_gap),
        )
        if out_path and os.path.exists(out_path):
            progress(1.0)
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
    _empty_cards = '<div style="color:#888;padding:8px;">暂无检测结果</div>'
    if not selected or not _batch_files:
        return None, "", "请先扫描文件夹并选择视频", _empty_cards, ""
    # 解析索引
    try:
        idx_str = selected.split("]")[0].split("[")[1]
        idx = int(idx_str) - 1
    except (ValueError, IndexError):
        return None, "", "解析视频索引失败", _empty_cards, ""
    if idx < 0 or idx >= len(_batch_files):
        return None, "", "视频索引超出范围", _empty_cards, ""

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
        return None, "", f"读取失败: {e}", _empty_cards, ""
    _video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                        codec=info["codec"], current_frame=0)
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
    return preview, info_str, status, _empty_cards, ""


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


def on_batch_detect(ball_conf, min_gap_sec,
                    diff_threshold=15, min_circularity=0.35, min_in_hoop_frames=2,
                    min_blob_area=30, search_margin=80, progress=gr.Progress()):
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

    progress(0)
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

        progress(0.05 + 0.9 * (processed_count - 1) / max(total_to_process, 1))

        # 读取视频信息
        try:
            info = get_video_info(video_path)
        except Exception as e:
            results.append(f"[{vi+1}] {vname} ❌ 读取失败: {e}")
            continue

        fps = info["fps"]
        total = info["total"]

        # 初始化检测器（用该视频的标定数据：篮筐 + 基准帧）
        # 高级参数由 UI 传入，批量识别也支持运行时调整
        detector = GoalDetector(hoop, baseline_frame=baseline_frame,
                                min_gap_sec=float(min_gap_sec),
                                diff_threshold=int(diff_threshold),
                                min_blob_area=int(min_blob_area),
                                search_margin=int(search_margin),
                                fusion_mode="visual_only",
                                loose_mode=True,
                                yolo_confirm=True,
                                rolling_baseline_sec=60.0,
                                min_circularity=float(min_circularity),
                                min_in_hoop_frames=int(min_in_hoop_frames))

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
                        progress(0.05 + 0.9 * sub_prog)
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

    progress(1.0)
    return summary, gr.update(choices=_history_choices())


# ============ Gradio 界面 ============
# 预填已知视频路径
_DEFAULT_VIDEO = r"D:\Downloads\highlights.mp4"

# 自定义主题：纯黑背景 + 暖橙黄 #FFB320 强调色
_custom_theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#fff8eb", c100="#ffeac7", c200="#ffd388", c300="#ffb84a",
        c400="#FFB320", c500="#FFB320", c600="#e69a12", c700="#bf7f0c",
        c800="#996608", c900="#7a5208", c950="#452e03",
    ),
    secondary_hue="orange",
    neutral_hue=gr.themes.Color(
        c50="#1a1a1a", c100="#141414", c200="#0f0f0f", c300="#0a0a0a",
        c400="#080808", c500="#000000", c600="#000000", c700="#000000",
        c800="#000000", c900="#000000", c950="#000000",
    ),
    radius_size="lg",
    font=[gr.themes.GoogleFont("Noto Sans SC"), "system-ui", "sans-serif"],
).set(
    # 主按钮：暖橙黄 #FFB320
    button_primary_background_fill="#FFB320",
    button_primary_background_fill_hover="#ffc94d",
    button_primary_text_color="#000000",
    button_primary_border_color="#FFB320",
    # 次按钮：深灰
    button_secondary_background_fill="#1a1a1a",
    button_secondary_background_fill_hover="#2a2a2a",
    button_secondary_text_color="#ffffff",
    button_secondary_border_color="#444444",
    # 背景纯黑
    body_background_fill="#000000",
    body_text_color="#ffffff",
    block_background_fill="#0a0a0a",
    block_border_color="#222222",
    block_title_text_color="#ffffff",
    block_label_text_color="#dddddd",
    input_background_fill="#0f0f0f",
    input_border_color="#444444",
    border_color_primary="#FFB320",
)

# 自定义 CSS：16:9 横向布局，左右分栏，简约科技风
_custom_css = """
/* 全局：纯黑背景，撑满屏幕 */
.gradio-container { max-width: 100% !important; padding: 12px !important; background: #000 !important; }
body { background: #000 !important; }

/* 顶部标题：紧凑 */
.top-header { text-align: center; margin: 4px 0 8px 0 !important; }
.top-header h1 { color: #FFB320 !important; font-size: 22px !important; margin: 0 !important; }
.top-header p { color: #888 !important; font-size: 12px !important; margin: 2px 0 0 0 !important; }

/* 左右分栏容器 */
.split-row { gap: 10px !important; }

/* 左侧控制面板：深色卡片，纵向堆叠 */
.left-panel {
    background: #0a0a0a !important;
    border: 1px solid #222 !important;
    border-radius: 12px !important;
    padding: 10px !important;
    max-height: calc(100vh - 80px) !important;
    overflow-y: auto !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}
/* 压缩左侧面板内部所有元素间距 */
.left-panel > div,
.left-panel .block,
.left-panel .wrap {
    margin: 0 !important;
    padding: 0 !important;
    gap: 2px !important;
}
.left-panel .gradio-group,
.left-panel .gradio-accordion {
    margin: 2px 0 !important;
    padding: 0 !important;
}
/* 强制左侧面板所有子元素纵向堆叠（覆盖 Gradio 6.x 内部 wrap/block 的横向 flex） */
/* 使用 [style] 选择器覆盖 Gradio 内联 style 中的 flex-direction: row */
.left-panel [style*="flex-direction: row"] {
    flex-direction: column !important;
}
.left-panel [style*="flex-direction:row"] {
    flex-direction: column !important;
}
/* 通用覆盖：所有 wrap/block/form 等容器 */
.left-panel > div,
.left-panel .block,
.left-panel .block .wrap,
.left-panel .form,
.left-panel .gradio-group,
.left-panel .gradio-accordion,
.left-panel .container,
.left-panel .gap,
.left-panel .panel,
.left-panel .header,
.left-panel .wrap {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    flex-wrap: nowrap !important;
}
/* gr.Row 容器保留横向（通过 class 识别） */
.left-panel .row,
[class*="row"] .left-panel {
    flex-direction: row !important;
    flex-wrap: wrap !important;
}
.left-panel::-webkit-scrollbar { width: 6px; }
.left-panel::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

/* 隐藏 clip_target 但保持 DOM 可访问 */
.clip-target-box { position: absolute !important; left: -9999px !important; width: 0 !important; height: 0 !important; overflow: hidden !important; opacity: 0 !important; pointer-events: none !important; }

/* 右侧预览区：深色卡片，固定大区域 */
.right-panel {
    background: #0a0a0a !important;
    border: 1px solid #222 !important;
    border-radius: 12px !important;
    padding: 14px !important;
    min-height: calc(100vh - 100px) !important;
}

/* 步骤标记：数字圆圈 + 文字 */
.step-label {
    display: flex; align-items: center; gap: 8px;
    color: #FFB320 !important; font-size: 14px !important; font-weight: 600 !important;
    margin: 4px 0 2px 0 !important;
}
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%;
    background: #FFB320; color: #000; font-size: 13px; font-weight: 700;
}

/* 居中大号主按钮：开始识别 */
.main-action-btn {
    background: #FFB320 !important; color: #000 !important;
    border: none !important; border-radius: 10px !important;
    font-size: 16px !important; font-weight: 700 !important;
    padding: 10px 0 !important; margin: 6px 0 !important;
    width: 100% !important; box-shadow: 0 0 16px rgba(255,179,32,0.3) !important;
}
.main-action-btn:hover { background: #ffc94d !important; }

/* 功能按钮行：紧凑小按钮 */
.func-row { gap: 6px !important; margin: 2px 0 !important; }
.func-btn {
    background: #1a1a1a !important; color: #fff !important;
    border: 1px solid #444 !important; border-radius: 8px !important;
    font-size: 13px !important; padding: 7px 12px !important;
}
.func-btn:hover { background: #2a2a2a !important; border-color: #FFB320 !important; }

/* 结果列表：紧凑卡片，可滚动 */
.result-list {
    background: #0f0f0f !important; border: 1px solid #222 !important;
    border-radius: 8px !important; padding: 6px !important; margin-top: 4px !important;
}
.result-list-label { color: #FFB320 !important; font-size: 14px !important; font-weight: 600 !important; }
.result-list-container {
    max-height: 340px; overflow-y: auto; padding-right: 6px;
}
.result-list-container::-webkit-scrollbar { width: 6px; }
.result-list-container::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
.result-item {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 10px; border: 1px solid #222; border-radius: 8px;
    margin-bottom: 6px; background: #121212;
}
.result-item.highlight { border-color: #FFB320; }
.result-meta { flex: 1; min-width: 0; }
.result-title { color: #fff; font-size: 14px; font-weight: 600; }
.result-status { font-size: 12px; color: #bbb; }
.result-actions { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.result-select {
    background: #0f0f0f; color: #eee; border: 1px solid #444; border-radius: 6px;
    padding: 4px 8px; font-size: 13px;
}

/* 状态框：等宽字体 */
.status-box textarea,
.status-box .wrap textarea,
.status-box input[type="text"] {
    font-family: 'Consolas', 'SF Mono', monospace !important;
    font-size: 13px !important; color: #eee !important;
    background: #0f0f0f !important;
    -webkit-text-fill-color: #eee !important;
}

/* 折叠面板标题 */
.acc-title { font-size: 13px !important; color: #eee !important; }

/* 全局提升文字对比度（覆盖 Gradio 内部样式）*/
.left-panel label,
.left-panel .label-wrap,
.left-panel .label-wrap > span,
.left-panel .label-wrap > div,
.left-panel span,
.left-panel p,
.left-panel .header {
    color: #ddd !important;
}
.left-panel input,
.left-panel input[type="text"],
.left-panel textarea,
.left-panel select,
.left-panel .wrap textarea,
.left-panel .wrap input {
    color: #eee !important;
    font-size: 13px !important;
    -webkit-text-fill-color: #eee !important;
    background: #0f0f0f !important;
}
/* Gradio 6.x 内部组件文字覆盖 */
.left-panel [data-testid] textarea,
.left-panel [data-testid] input,
.left-panel [data-testid] label,
.left-panel .prose,
.left-panel .prose * {
    color: #eee !important;
    -webkit-text-fill-color: #eee !important;
}
/* Dropdown/Combobox 输入框（Svelte 组件） */
.left-panel input[role="combobox"],
.left-panel input.svelte-1xfsv4t,
.left-panel .wrap input[role="combobox"],
input[role="combobox"],
input[role="combobox"].subdued {
    color: #eee !important;
    -webkit-text-fill-color: #eee !important;
    background: #0f0f0f !important;
    opacity: 1 !important;
}
/* Dropdown 整个选择框容器：加明显边框 */
.left-panel [role="combobox"] ~ *,
.left-panel .wrap:has(input[role="combobox"]),
.left-panel .dropdown-wrap,
input[role="combobox"] {
    border: 1px solid #555 !important;
    border-radius: 8px !important;
}
/* Dropdown 外层容器（Gradio 6.x Svelte 结构） */
.left-panel [data-testid="dropdown"],
.left-panel [data-testid="dropdown"] > div,
input[role="combobox"] {
    border: 1px solid #555 !important;
    border-radius: 8px !important;
    background: #0f0f0f !important;
}
/* Dropdown 内部文字容器 */
.left-panel .item,
.left-panel .dropdown-item,
.left-panel [role="option"] {
    color: #eee !important;
    -webkit-text-fill-color: #eee !important;
}
/* 进度条文字（Gradio 内部 Progress 组件） */
.progress-text,
.progress-bar span,
.progress span,
[class*="progress"] span,
[class*="progress"] p,
[class*="progress"] div,
[class*="ProgressBar"] span,
[class*="ProgressBar"] p,
[class*="ProgressBar"] div,
[data-testid] .progress span,
[data-testid] .progress p {
    color: #eee !important;
    -webkit-text-fill-color: #eee !important;
}

/* 视频预览：固定大区域 */
.preview-video { background: #000 !important; border-radius: 8px !important; }
.preview-video video { max-height: calc(100vh - 260px) !important; }

/* 视频播放控制栏 */
.playback-controls {
    background: #0a0a0a !important;
    border: 1px solid #222 !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    margin-top: 6px !important;
}

/* 预览窗口叠加：三个预览组件叠在同一位置，视觉上是单个窗口 */
.preview-stack {
    position: relative !important;
    min-height: 560px !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
.preview-stack > div {
    position: absolute !important;
    top: 0 !important; left: 0 !important; right: 0 !important;
    border: none !important;
    box-shadow: none !important;
}
.preview-stack > div.hide-container {
    display: none !important;
}
.preview-stack .empty {
    display: none !important;
}
.preview-stack .label-wrap {
    display: none !important;
}
.preview-stack .status-tracker,
.preview-stack [data-testid="status-tracker"] {
    display: none !important;
}

/* 帧选择器：紧凑 */
.frame-slider { margin: 8px 0 !important; }
.frame-slider .head { display: none !important; }
.frame-slider .min_value,
.frame-slider .max_value { display: none !important; }

/* 隐藏多余 label */
.gradio-container .label-wrap { padding: 4px 0 !important; }

/* 全局：确保 span/p/label 文字在深色背景下可见 */
.gradio-container span,
.gradio-container p,
.gradio-container label {
    color: #ddd;
}
"""

with gr.Blocks(title="进球剪辑神器", theme=_custom_theme, css=_custom_css) as demo:
    # ====== 顶部标题栏 ======
    gr.HTML('<div class="top-header"><h1>🏀 进球剪辑神器 v0.1.0（PC桌面端布局）</h1></div>')

    # ====== 左右分栏：左 38% / 右 62% ======
    with gr.Row(equal_height=False, elem_classes="split-row"):
        # ============ 左侧：功能控制面板 ============
        with gr.Column(scale=38, min_width=380, elem_classes="left-panel"):
            # 分步操作按钮组
            gr.HTML('<div class="step-label"><span class="step-num">1</span>加载视频</div>')
            video_path_input = gr.Textbox(
                value=_DEFAULT_VIDEO, show_label=False, lines=1,
                placeholder=r"文件路径或文件夹路径")
            info_text = gr.Textbox(label="视频信息", interactive=False, lines=1,
                                   elem_classes="status-box")
            load_btn = gr.Button("📥 加载视频", variant="primary", elem_classes="func-btn")

            # 批量标定（折叠，仅文件夹模式显示）
            with gr.Accordion("📁 批量标定", open=False, visible=False) as batch_group:
                batch_video_selector = gr.Dropdown(
                    label="视频列表", choices=[], value=None, interactive=True,
                    info="✅=已标定 ⬜=未标定")
                with gr.Row(elem_classes="func-row"):
                    batch_load_btn = gr.Button("📥 加载到预览", size="sm", elem_classes="func-btn")
                    batch_save_calib_btn = gr.Button("💾 保存标定", variant="primary", size="sm",
                                                     elem_classes="func-btn")
                batch_calib_status = gr.Textbox(label="标定进度", interactive=False, lines=1,
                                                elem_classes="status-box")

            gr.HTML('<div class="step-label"><span class="step-num">2</span>框住篮圈（在右侧预览图点击 2 点）</div>')
            with gr.Row(elem_classes="func-row"):
                click_status = gr.Textbox(label="标定状态", interactive=False,
                                          scale=3, elem_classes="status-box")
                reset_btn = gr.Button("🔄 重置", size="sm", scale=1, elem_classes="func-btn")

            run_btn = gr.Button("开始识别", variant="primary", elem_classes="main-action-btn")
            batch_detect_btn = gr.Button("🚀 批量识别", variant="primary",
                                         elem_classes="main-action-btn", visible=False)

            # 辅助功能栏
            with gr.Row(elem_classes="func-row"):
                cut_btn = gr.Button("✂️ 导出合集", size="sm", elem_classes="func-btn")
                export_clip_btn = gr.Button("📤 导出分段", size="sm", elem_classes="func-btn")

            # 检测参数（折叠）
            with gr.Accordion("⚙️ 检测参数", open=False):
                with gr.Row():
                    start_frame = gr.Number(label="起始帧", value=0, precision=0)
                    end_frame = gr.Number(label="结束帧(0=末尾)", value=0, precision=0)
                with gr.Row():
                    ball_conf = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="球检测置信度")
                    min_gap = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="最小进球间隔(秒)")
                with gr.Row():
                    pre_roll = gr.Number(label="进球前(秒)", value=5, precision=0)
                    post_roll = gr.Number(label="进球后(秒)", value=5, precision=0)
                    cut_min_gap = gr.Number(label="合并间隔(秒)", value=8, precision=0)
                with gr.Accordion("🔧 高级参数", open=False):
                    diff_threshold = gr.Slider(5, 40, value=15, step=5, label="帧差阈值")
                    min_circularity = gr.Slider(0.0, 0.8, value=0.35, step=0.05, label="圆形度阈值")
                    min_in_hoop_frames = gr.Slider(1, 6, value=2, step=1, label="进框最少帧数")
                    min_blob_area = gr.Slider(10, 200, value=30, step=10, label="最小斑块面积")
                    search_margin = gr.Slider(20, 150, value=80, step=10, label="搜索范围(像素)")

            # 📋 识别进球片段结果列表
            with gr.Group(elem_classes="result-list"):
                gr.HTML('<div class="result-list-label">📋 识别进球片段结果列表</div>')
                result_cards = gr.HTML(
                    value='<div style="color:#888;padding:8px;">暂无检测结果</div>',
                    label="")
                # 隐藏的 Textbox 用于接收卡片按钮点击事件
                clip_target = gr.Textbox(value="", label="", elem_id="clip_target",
                                         visible=True, interactive=True,
                                         elem_classes="clip-target-box")
                result_status = gr.Textbox(label="检测统计", interactive=False, lines=4,
                                           elem_classes="status-box")

            # 历史记录（折叠）
            with gr.Accordion("📂 历史记录", open=False):
                with gr.Row():
                    history_selector = gr.Dropdown(
                        label="选择历史", choices=_history_choices(), value=None,
                        interactive=True, scale=4, show_label=False)
                    refresh_history_btn = gr.Button("🔄", size="sm", scale=1, elem_classes="func-btn")
                load_history_btn = gr.Button("📂 加载历史", variant="primary", elem_classes="func-btn")
                history_status = gr.Textbox(label="历史状态", interactive=False, lines=2,
                                            elem_classes="status-box")

            # 隐藏组件
            highlights_status = gr.Textbox(label="剪辑状态", interactive=False, lines=1,
                                           elem_classes="status-box", visible=False)
            export_status = gr.Textbox(label="导出状态", interactive=False, lines=1,
                                       elem_classes="status-box", visible=False)
            # 检测结果暂存（用于链式事件）
            _detect_result_state = gr.State(value=None)

        # ============ 右侧：视频预览播放区 ============
        with gr.Column(scale=62, min_width=500, elem_classes="right-panel"):
            # 视频预览窗口（三个组件叠在同一位置）
            with gr.Group(elem_classes="preview-stack"):
                preview_image = gr.Image(
                    label="点击画面 2 个点标定篮筐（左上 + 右下）",
                    type="numpy", interactive=False, height=520,
                    elem_classes="preview-video", visible=True)
                result_video = gr.Video(
                    label="", height=520,
                    elem_classes="preview-video", visible=False, autoplay=True)
                highlights_video = gr.Video(
                    label="", height=520,
                    elem_classes="preview-video", visible=False)

    # ====== 事件绑定 ======
    def _show_image():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    def _show_result():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    def _show_highlights():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    # 帧滑块变化 → 更新预览 + 播放信息
    def on_frame_change(frame_idx):
        img, status = on_preview(frame_idx)
        return img, status

    # 加载视频 → 显示标定图
    def on_load_and_switch(path):
        result = on_load_path(path)
        return list(result) + list(_show_image())
    load_btn.click(on_load_and_switch, inputs=[video_path_input],
                   outputs=[preview_image, info_text,
                            batch_group, batch_video_selector,
                            batch_calib_status, click_status,
                            run_btn, batch_detect_btn,
                            result_cards, result_status,
                            result_video, highlights_video])

    # 点击预览图 → 标定篮筐
    preview_image.select(on_image_click, inputs=[],
                         outputs=[preview_image, click_status])
    reset_btn.click(on_reset_hoop, outputs=[click_status])

    # 开始识别 → 右侧预览（显示进度）→ 左侧结果（无进度）
    def on_run_right(start_f, end_f, ball_c, gap, dt, mc, mhf, mba, sm):
        result = on_run_detect(start_f, end_f, ball_c, gap, dt, mc, mhf, mba, sm)
        # result: (first_clip, cards_html, status)
        first_clip = result[0]
        show = _show_result()  # (preview_hide, result_show, highlights_hide)
        # 返回：右侧组件 + 状态暂存
        return (gr.update(value=first_clip, visible=True) if first_clip else gr.update(),
                show[0], show[2], result)

    def on_run_left(state_val):
        if state_val is None:
            return gr.update(), gr.update()
        return state_val[1], state_val[2]  # cards_html, status

    run_btn.click(on_run_right,
                  inputs=[start_frame, end_frame, ball_conf, min_gap,
                          diff_threshold, min_circularity, min_in_hoop_frames,
                          min_blob_area, search_margin],
                  outputs=[result_video, preview_image, highlights_video,
                           _detect_result_state],
                  show_progress="hidden"
                  ).then(on_run_left,
                         inputs=[_detect_result_state],
                         outputs=[result_cards, result_status])

    # 批量识别
    batch_detect_btn.click(on_batch_detect,
                           inputs=[ball_conf, min_gap,
                                   diff_threshold, min_circularity, min_in_hoop_frames,
                                   min_blob_area, search_margin],
                           outputs=[result_status, history_selector])

    # 导出合集 → 显示集锦视频
    def on_cut_and_switch(pre, post, gap):
        path, status = on_generate_highlights(pre, post, gap)
        if path:
            return (gr.update(value=path, visible=True), status,
                    gr.update(visible=False), gr.update(visible=False))
        return path, status, gr.update(visible=False), gr.update(visible=False)
    cut_btn.click(on_cut_and_switch,
                  inputs=[pre_roll, post_roll, cut_min_gap],
                  outputs=[highlights_video, highlights_status,
                           preview_image, result_video])

    # 卡片按钮操作（通过隐藏 Textbox 接收 JS 事件）
    def on_clip_action_and_switch(raw_val):
        result_video_val, cards_html, status = on_clip_action(raw_val)
        if result_video_val is not None and result_video_val != gr.update():
            return (gr.update(value=result_video_val, visible=True), cards_html, status,
                    gr.update(visible=False), gr.update(visible=False))
        return result_video_val, cards_html, status, gr.update(), gr.update()
    clip_target.change(on_clip_action_and_switch,
                       inputs=[clip_target],
                       outputs=[result_video, result_cards, result_status,
                                preview_image, highlights_video])

    # 批量模式
    def on_batch_load_and_switch(video_name):
        result = on_batch_load_video(video_name)
        return list(result) + list(_show_image())
    batch_load_btn.click(on_batch_load_and_switch, inputs=[batch_video_selector],
                         outputs=[preview_image, info_text, click_status,
                                  result_cards, result_status,
                                  result_video, highlights_video])
    batch_save_calib_btn.click(on_batch_save_calib,
                               outputs=[batch_calib_status, batch_video_selector, click_status])

    # 历史记录
    def on_load_history_and_switch(idx, pre, post, gap):
        result = on_load_history(idx, pre, post, gap)
        preview, info_text_val, hist_status, video = result[:4]
        cards_html = _build_result_cards_html()
        return [preview, info_text_val, hist_status, video,
                cards_html, ""] + list(_show_result())
    load_history_btn.click(on_load_history_and_switch,
                           inputs=[history_selector, pre_roll, post_roll, cut_min_gap],
                           outputs=[preview_image, info_text,
                                    history_status, result_video,
                                    result_cards, result_status,
                                    preview_image, highlights_video])
    refresh_history_btn.click(
        lambda: gr.update(choices=_history_choices()),
        outputs=[history_selector])

    # 注入全局 JS 函数（在页面加载时执行）
    _init_counter_js = "window._clipCounter = 0"
    _inject_js = "window.clipAction = function(action, idx) { var ta = document.querySelector('#clip_target textarea'); if (!ta) return; window._clipCounter++; var val = action + '_' + idx + '_' + window._clipCounter; var s = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set; s.call(ta, val); ta.dispatchEvent(new Event('input', {bubbles: true})); ta.dispatchEvent(new Event('change', {bubbles: true})); }"
    demo.load(js=_init_counter_js)
    demo.load(js=_inject_js)

    # 清理预览区多余边框
    _clean_preview_js = """(function(){
        function clean(){
            var stack = document.querySelector('.preview-stack');
            if(!stack) return;
            stack.querySelectorAll('.label-wrap').forEach(function(e){e.style.display='none';});
            stack.querySelectorAll('[data-testid="status-tracker"]').forEach(function(e){e.style.display='none';});
            stack.querySelectorAll('.empty').forEach(function(e){e.style.display='none';});
            stack.querySelectorAll('.hide-container').forEach(function(e){e.style.display='none';});
            stack.style.border='none';
            stack.style.boxShadow='none';
            stack.style.background='transparent';
            stack.style.padding='0';
        }
        clean();
        setInterval(clean, 500);
    })()"""
    demo.load(js=_clean_preview_js)


if __name__ == "__main__":
    _out_dir = str(Path(_CACHE_ROOT) / "demo_output")
    _gradio_dir = str(Path(_CACHE_ROOT) / "gradio")
    os.makedirs(_out_dir, exist_ok=True)
    os.makedirs(_gradio_dir, exist_ok=True)
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=7871,
                show_error=True, prevent_thread_lock=False,
                max_file_size=5000*1024*1024,
                allowed_paths=[_CACHE_ROOT])
