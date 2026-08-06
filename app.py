"""篮球录像检测可视化界面。

用法:
    E:\\bball-env\\python.exe app.py
然后浏览器打开 http://127.0.0.1:7862

三个 Tab:
  1. 标定与追踪（固定机位免微调）：手动框选篮筐+篮球 → CamShift 追踪 → 进球检测
  2. YOLO 检测可视化：用 YOLOv8 检测球和篮筐（需微调权重）
  3. 完整流程：跑 pipeline.py 输出集锦 mp4
"""
import os
import sys
import cv2
import time
import yaml
import numpy as np
import gradio as gr
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# ============ 缓存目录改到 E 盘（C 盘空间不足）============
_CACHE_ROOT = r"E:\bball_cache"
os.environ["GRADIO_TEMP_DIR"] = os.path.join(_CACHE_ROOT, "gradio")
os.environ["MPLCONFIGDIR"] = os.path.join(_CACHE_ROOT, "matplotlib")
os.environ["ULTRALYTICS_CONFIG_DIR"] = os.path.join(_CACHE_ROOT, "ultralytics")
os.environ["TORCH_HOME"] = os.path.join(_CACHE_ROOT, "torch")
os.environ["PIP_CACHE_DIR"] = os.path.join(_CACHE_ROOT, "pip")
os.environ["HF_HOME"] = os.path.join(_CACHE_ROOT, "huggingface")
for _d in [os.environ[k] for k in ["GRADIO_TEMP_DIR", "MPLCONFIGDIR",
          "ULTRALYTICS_CONFIG_DIR", "TORCH_HOME"]]:
    os.makedirs(_d, exist_ok=True)

# FFprobe 路径
FFPROBE = r"E:\bball-env\Library\bin\ffprobe.exe"

# 公共模块
from video_io import av_open, get_video_info, read_frame, VideoReader
from transcoder import probe_codec, transcode_to_h264
from tracker import GoalDetector
from audio_detector import AudioPeakDetector, probe_audio

# ============ 全局状态 ============
_model_cache = {}
_video_state = {"path": None, "total": 0, "fps": 30.0, "codec": "unknown"}
_ball_conf = 0.25  # YOLO 球检测置信度阈值
_calib = {
    "clicks": [],   # 待确认的点击点 [(x,y), ...]
    "hoop": None,   # (x1,y1,x2,y2)
    "baseline_frame_idx": None,  # 基准帧号（无球的篮筐画面）
}
_audio_state = {
    "detector": None,   # AudioPeakDetector 实例
    "peaks": [],        # 音频峰值时间戳列表
    "stats": None,      # 统计信息
}


def get_model(weights):
    from ultralytics import YOLO
    if weights not in _model_cache:
        m = YOLO(weights)
        # 加载后立即搬到 GPU，避免首次推理在 CPU 上跑
        try:
            import torch
            if torch.cuda.is_available():
                m.to("cuda:0")
        except Exception:
            pass
        _model_cache[weights] = m
    return _model_cache[weights]


def list_weights():
    wdir = ROOT / "weights"
    wdir.mkdir(exist_ok=True)
    pts = sorted(wdir.glob("*.pt"))
    return [str(p) for p in pts] if pts else [str(wdir / "yolov8n.pt")]


def get_ball_model():
    """懒加载球检测 YOLO 模型。

    优先用 weights/ 下的篮球专用微调权重；否则用 yolov8n.pt（COCO 预训练，自动下载）。
    返回 (model, weights_path)。
    """
    wdir = ROOT / "weights"
    weights = None
    if wdir.exists():
        # 优先选名字含 basketball/ball/finetuned 的权重（篮球专用）
        pts = sorted(wdir.glob("*.pt"),
                     key=lambda p: 0 if any(k in p.name.lower()
                                            for k in ("basketball", "finetuned", "ball"))
                                   else 1)
        if pts:
            weights = str(pts[0])
    if weights is None:
        weights = "yolov8n.pt"
    return get_model(weights), weights


def detect_ball_yolo(frame, conf=None, imgsz=1280, augment=False):
    """用 YOLO 检测球，返回 (cx, cy, x1, y1, x2, y2, conf) 或 None。

    兼容 COCO 预训练（sports ball=32）和篮球微调权重（basketball 类）。
    取置信度最高的球框。
    imgsz: 推理分辨率，篮球微调模型训练时用 1280，默认 1280 提升小目标检出。
    augment: TTA（Test Time Augmentation），慢约 3 倍但提升小目标检出率。
    """
    if conf is None:
        conf = _ball_conf
    model, _ = get_ball_model()
    res = model.predict(frame, conf=conf, imgsz=imgsz, device="cuda:0",
                        augment=augment, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    names = res.names
    clses = res.boxes.cls.cpu().numpy().astype(int)
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    best = None
    for i, c in enumerate(clses):
        n = names.get(c, "").lower()
        if "ball" in n or "basketball" in n:
            if best is None or confs[i] > confs[best]:
                best = i
    if best is None:
        return None
    x1, y1, x2, y2 = xyxy[best]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (float(cx), float(cy), float(x1), float(y1), float(x2), float(y2), float(confs[best]))


def get_frame(idx):
    """读取指定帧（BGR）。"""
    if _video_state["path"] is None:
        return None
    return read_frame(_video_state["path"], idx,
                      total=_video_state["total"], fps=_video_state["fps"])


def draw_calib(frame, ball_det=None, net_box=None):
    """在帧上画篮筐框（绿）、篮网区域（蓝）和待确认点击点（黄）。

    ball_det: 可选，YOLO 检测到的球 (cx, cy, x1, y1, x2, y2, conf)，画红框。
    net_box: 可选，篮网区域 (x1,y1,x2,y2)，画蓝框。
    """
    out = frame.copy()
    if _calib["hoop"]:
        x1, y1, x2, y2 = _calib["hoop"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(out, "HOOP", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if net_box is not None:
        nx1, ny1, nx2, ny2 = net_box
        cv2.rectangle(out, (int(nx1), int(ny1)), (int(nx2), int(ny2)), (255, 128, 0), 2)
        cv2.putText(out, "NET", (int(nx1), max(int(ny1) - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
    if ball_det is not None:
        _, _, bx1, by1, bx2, by2, bconf = ball_det
        cv2.rectangle(out, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 3)
        cv2.putText(out, f"BALL {bconf:.2f}", (int(bx1), max(int(by1) - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    for i, (x, y) in enumerate(_calib["clicks"]):
        cv2.circle(out, (x, y), 10, (255, 255, 0), -1)
        cv2.putText(out, str(i + 1), (x - 6, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return out


# ============ Tab1: 标定与追踪 ============
def on_load_video_t1(video):
    """加载视频，读取信息和第一帧预览（不转码，避免阻塞导致超时）。"""
    if video is None:
        return None, gr.update(maximum=1, value=0), "请先上传视频", None
    try:
        info = get_video_info(video)
    except Exception as e:
        return None, gr.update(maximum=1, value=0), f"读取视频失败: {e}", None
    _video_state.update(path=video, total=info["total"], fps=info["fps"],
                        codec=info["codec"])
    codec = info["codec"]
    playable = codec in ("h264", "avc", "avc1", "")
    play_note = "可直接播放" if playable else f"编码 {codec}，预览需转码"
    info_str = (f"{play_note} | {info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {codec}")
    # 预览第一帧
    frame = get_frame(0)
    preview_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    # H.264 直接返回原路径预览；非 H.264 先不返回视频路径（避免浏览器解码失败报错）
    preview_video_path = video if playable else None
    return (preview_video_path, gr.update(maximum=max(info["total"] - 1, 1), value=0),
            info_str, preview_img)


def on_transcode_preview():
    """手动触发转码（避免加载视频时阻塞）。"""
    if _video_state["path"] is None:
        return None, "请先加载视频"
    preview_path, msg = transcode_to_h264(_video_state["path"], FFPROBE)
    return preview_path, msg


def _compute_net_box():
    """根据篮筐框计算篮网区域（篮筐下方 1.5 倍篮筐高度）。"""
    if _calib["hoop"] is None:
        return None
    x1, y1, x2, y2 = _calib["hoop"]
    w = x2 - x1
    h = y2 - y1
    return (x1 - int(w * 0.1), y2, x2 + int(w * 0.1), y2 + int(h * 1.5))


def on_preview_t1(frame_idx):
    frame = get_frame(frame_idx)
    if frame is None:
        return None, "读取帧失败"
    annotated = draw_calib(frame, net_box=_compute_net_box())
    ts = frame_idx / _video_state["fps"]
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), f"预览帧 {frame_idx} ({ts:.1f}s)"


def on_image_click(evt: gr.SelectData, frame_idx):
    """点击图片框选篮筐：2个点形成一个框。"""
    x, y = int(evt.index[0]), int(evt.index[1])
    _calib["clicks"].append((x, y))
    frame = get_frame(frame_idx)
    if frame is None:
        return None, "读取帧失败"

    status = f"点击 ({x},{y})，已收集 {len(_calib['clicks'])}/2 个点"
    if len(_calib["clicks"]) >= 2:
        p1, p2 = _calib["clicks"][:2]
        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
        _calib["hoop"] = (x1, y1, x2, y2)
        status = f"篮筐已标定: ({x1},{y1}) - ({x2},{y2})，可点击「检测球」测试 YOLO"
        _calib["clicks"] = []

    annotated = draw_calib(frame, net_box=_compute_net_box())
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), status


def on_track_test(frame_idx, ball_conf, use_tta=False):
    """在指定帧用 YOLO 检测篮球，检测不到时显示所有结果用于诊断。"""
    global _ball_conf
    _ball_conf = float(ball_conf)
    frame = get_frame(frame_idx)
    if frame is None:
        return None, "读取帧失败"
    ts = frame_idx / _video_state["fps"]
    try:
        model, wpath = get_ball_model()
        res = model.predict(frame, conf=_ball_conf, imgsz=1280, device="cuda:0",
                            augment=use_tta, verbose=False)[0]
    except Exception as e:
        return None, f"YOLO 检测失败: {e}"

    names = res.names
    annotated = frame.copy()
    all_dets = []  # [(cls_name, conf), ...]

    if res.boxes is not None and len(res.boxes) > 0:
        clses = res.boxes.cls.cpu().numpy().astype(int)
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()

        # 找球：名称含 ball/basketball
        ball_idx = None
        for i, c in enumerate(clses):
            n = names.get(c, "").lower()
            all_dets.append(f"{names.get(c, c)}({confs[i]:.2f})")
            if "ball" in n or "basketball" in n:
                if ball_idx is None or confs[i] > confs[ball_idx]:
                    ball_idx = i

        if ball_idx is not None:
            # 画球框（红）
            x1, y1, x2, y2 = xyxy[ball_idx]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.circle(annotated, (cx, cy), 10, (0, 165, 255), 3)
            cv2.putText(annotated, f"BALL {confs[ball_idx]:.2f}",
                        (int(x1), max(int(y1) - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            annotated = draw_calib(annotated)
            det_str = ", ".join(all_dets[:8])
            return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), \
                   f"帧 {frame_idx} ({ts:.1f}s) 检测到球: ({cx},{cy}) conf={confs[ball_idx]:.2f} | 全部: {det_str}"
        else:
            # 没有球类别，画出所有检测结果（黄色），帮助诊断
            for i in range(len(clses)):
                x1, y1, x2, y2 = xyxy[i]
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.putText(annotated, f"{names.get(clses[i], clses[i])} {confs[i]:.2f}",
                            (int(x1), max(int(y1) - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            annotated = draw_calib(annotated)
            det_str = ", ".join(all_dets[:10]) if all_dets else "无任何检测"
            cv2.putText(annotated, "NO BALL - showing all detections",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), \
                   f"帧 {frame_idx} ({ts:.1f}s) 未检测到球类别。YOLO 检测到: {det_str}。模型={wpath}"
    else:
        annotated = draw_calib(annotated)
        cv2.putText(annotated, "NO DETECTIONS AT ALL",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), \
               f"帧 {frame_idx} ({ts:.1f}s) YOLO 未检测到任何目标（conf={_ball_conf}）。模型={wpath}，可降低阈值或换帧"


def on_detect_goals(start_frame, end_frame, batch, ball_conf, approach_dist, confirm_below, min_gap,
                    net_threshold,
                    audio_threshold_factor, fusion_window, fusion_mode):
    """基准帧差法 + 连通域分析 + 音频峰值融合检测进球（固定机位专用）。"""
    if _calib["hoop"] is None:
        yield None, "请先标定篮筐", ""
        return
    if _video_state["path"] is None:
        yield None, "请先加载视频", ""
        return

    fps = _video_state["fps"]
    total = _video_state["total"]
    start_frame = max(0, int(start_frame))
    end_frame = min(total, int(end_frame))
    batch = max(1, int(batch))

    # 获取基准帧（无球的篮筐画面）
    baseline_frame = None
    if _calib["baseline_frame_idx"] is not None:
        baseline_frame = get_frame(int(_calib["baseline_frame_idx"]))
        base_note = f"基准帧: {_calib['baseline_frame_idx']}"
    else:
        # 没设置基准帧，用检测起始帧作为基准（不理想但能跑）
        baseline_frame = get_frame(start_frame)
        base_note = f"基准帧: {start_frame}（未手动设置，用起始帧，建议手动设置无球帧）"

    # 获取音频峰值（如果已分析）
    audio_peaks = _audio_state["peaks"] if _audio_state["peaks"] else None
    audio_note = (f"音频峰值: {len(audio_peaks)} 个 {audio_peaks[:5]}"
                  if audio_peaks else "音频: 未分析（仅用视觉信号）")

    detector = GoalDetector(_calib["hoop"],
                            baseline_frame=baseline_frame,
                            min_gap_sec=float(min_gap),
                            diff_threshold=int(net_threshold),
                            audio_peaks=audio_peaks,
                            fusion_window=float(fusion_window),
                            fusion_mode=fusion_mode)

    log_lines = ["方法: 基准帧差法 + 连通域分析 + 音频融合（固定机位专用）",
                 base_note,
                 audio_note,
                 f"融合模式: {fusion_mode} (窗口 ±{fusion_window}s)",
                 f"篮筐: {_calib['hoop']}",
                 f"搜索区域: {detector.get_debug_info()['search_area']}",
                 f"差分阈值: {net_threshold}",
                 f"范围: 帧 {start_frame}-{end_frame}, 抽帧间隔 {batch}",
                 "提示: 抽帧间隔建议设为1（帧差法需连续帧）"]
    checked = 0
    detected = 0
    t0 = time.time()

    yield None, "开始检测（基准帧差法 + 音频融合）...", "\n".join(log_lines)

    with VideoReader(_video_state["path"]) as reader:
        for fidx, frame in reader.iter_frames(start=start_frame, end=end_frame, batch=batch):
            checked += 1
            # 基准帧差法不需要 YOLO，直接传 None
            goal_ts = detector.feed(None, fidx, fps, frame=frame)
            dbg = detector.get_debug_info()
            if dbg["blob_box"] is not None:
                detected += 1  # 检测到运动斑块

            if goal_ts is not None:
                fused_tag = " [融合]" if goal_ts in detector.fused_goals else ""
                log_lines.append(
                    f"[进球] 帧 {fidx} ({goal_ts:.1f}s) 差分={dbg['diff_ratio']:.1f}{fused_tag}")

            if checked % 50 == 0:
                elapsed = time.time() - t0
                speed = checked / elapsed if elapsed > 0 else 0
                eta = (end_frame - start_frame - checked) / speed if speed > 0 else 0
                status_msg = (f"进度 {checked}/{end_frame-start_frame} 帧 | "
                              f"斑块 {detected} | 视觉 {len(detector.visual_goals)} | "
                              f"融合 {len(detector.fused_goals)} | 总进球 {len(detector.goals)} | "
                              f"差分={dbg['diff_ratio']:.1f} | "
                              f"{speed:.1f} fps | 已用 {elapsed:.0f}s | 预计还需 {eta:.0f}s")
                yield None, status_msg, "\n".join(log_lines)

    # 视频处理完后处理仅音频触发的候选
    audio_only = detector.finalize()
    if audio_only:
        log_lines.append(f"[仅音频] 补充 {len(audio_only)} 个候选: {audio_only}")

    # 画进球时间戳到图上
    frame = get_frame(start_frame)
    summary_img = None
    if frame is not None:
        annotated = draw_calib(frame)
        y0 = 40
        # 融合进球红色，仅视觉/仅音频黄色
        for i, ts in enumerate(detector.goals[:10]):
            color = (0, 0, 255) if ts in detector.fused_goals else (0, 255, 255)
            tag = "融合" if ts in detector.fused_goals else "视觉/音频"
            cv2.putText(annotated, f"Goal {i + 1}: {ts:.1f}s [{tag}]",
                        (20, y0 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)
        summary_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    elapsed = time.time() - t0
    log_lines.append(f"\n=== 总结 ===")
    log_lines.append(f"视觉进球: {len(detector.visual_goals)} 个")
    log_lines.append(f"融合确认: {len(detector.fused_goals)} 个")
    log_lines.append(f"仅音频补充: {len(audio_only)} 个")
    log_lines.append(f"最终进球: {len(detector.goals)} 个")
    log_lines.append(f"耗时: {elapsed:.0f}s")
    log_text = "\n".join(log_lines) if log_lines else "未检测到进球"
    status = (f"完成: 检查 {checked} 帧, 斑块 {detected} 帧, "
              f"视觉 {len(detector.visual_goals)}, 融合 {len(detector.fused_goals)}, "
              f"总进球 {len(detector.goals)} 个, 耗时 {elapsed:.0f}s")
    yield summary_img, status, log_text


def on_analyze_audio(audio_threshold_factor, min_gap_audio):
    """分析视频音频，检测峰值。"""
    if _video_state["path"] is None:
        return "请先加载视频", ""

    has_audio, sr, ch = probe_audio(_video_state["path"], FFPROBE)
    if not has_audio:
        _audio_state["peaks"] = []
        _audio_state["stats"] = None
        _audio_state["detector"] = None
        return "视频无音频流，无法进行音频检测", ""

    detector = AudioPeakDetector(
        threshold_factor=float(audio_threshold_factor),
        min_gap_sec=float(min_gap_audio))
    peaks = detector.analyze(_video_state["path"], ffprobe_path=FFPROBE)
    stats = detector.get_stats()

    _audio_state["detector"] = detector
    _audio_state["peaks"] = peaks
    _audio_state["stats"] = stats

    if not peaks:
        return (f"音频分析完成: 时长 {stats['duration']:.1f}s, "
                f"未检测到峰值（可降低阈值因子重试）"), ""

    peak_strs = [f"{p:.1f}s" for p in peaks[:20]]
    info = (f"音频分析完成: 时长 {stats['duration']:.1f}s, "
            f"背景噪声 {stats['bg_noise']:.1f}, "
            f"峰值阈值 {stats['peak_threshold']:.1f}, "
            f"检测到 {len(peaks)} 个峰值\n"
            f"峰值时间: {', '.join(peak_strs)}")
    return info, ""


def on_save_calib():
    """保存篮筐标定结果到 config.yaml。"""
    if _calib["hoop"] is None:
        return "请先标定篮筐"
    cfg_path = ROOT / "config.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    cfg.setdefault("calibration", {})
    cfg["calibration"]["hoop"] = list(_calib["hoop"])
    cfg["hoop"]["manual_box"] = list(_calib["hoop"])
    cfg_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
    return f"已保存到 config.yaml: hoop={_calib['hoop']}"


def on_reset_calib():
    """重置篮筐标定。"""
    _calib["hoop"] = None
    _calib["clicks"] = []
    return "已重置标定，请重新框选篮筐"


def on_set_baseline(frame_idx):
    """设置基准帧（无球的篮筐画面）。"""
    _calib["baseline_frame_idx"] = int(frame_idx)
    ts = int(frame_idx) / _video_state["fps"] if _video_state["fps"] > 0 else 0
    return f"已设置基准帧: 帧 {int(frame_idx)} ({ts:.1f}s)。请确保此帧画面无球。"


# ============ Tab2: YOLO 检测可视化 ============
def on_detect_yolo(frame_idx, conf, weights, only_ball):
    if _video_state["path"] is None:
        return None, "请先加载视频"
    frame = get_frame(frame_idx)
    if frame is None:
        return None, "读取帧失败"
    from ultralytics import YOLO
    model = get_model(weights)
    classes = [32] if only_ball else None  # COCO 32 = sports ball
    res = model.predict(frame, conf=conf, device="cuda:0",
                        classes=classes, verbose=False)[0]
    annotated = res.plot()
    names = res.names
    cls_counts = {}
    if res.boxes is not None and len(res.boxes) > 0:
        for c in res.boxes.cls.cpu().numpy().astype(int):
            n = names.get(c, str(c))
            cls_counts[n] = cls_counts.get(n, 0) + 1
    info = ", ".join(f"{k}: {v}" for k, v in cls_counts.items()) or "无检测"
    ts = frame_idx / _video_state["fps"]
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), f"帧 {frame_idx} ({ts:.1f}s) | {info}"


# ============ Tab3: 完整流程 ============
def on_run_pipeline(video, pre_roll, post_roll, min_gap):
    """跑完整 pipeline（YOLO 检测球 + 篮筐标定 + 进球判断 + 剪辑）。"""
    if _calib["hoop"] is None:
        return None, "请先在 Tab1 标定篮筐", ""
    if video is None:
        return None, "请先上传视频", ""

    fps = _video_state["fps"]
    total = _video_state["total"]
    detector = GoalDetector(_calib["hoop"])
    log_lines = [f"开始处理: {_video_state['path']}"]
    log_lines.append(f"视频信息: {total} 帧, {fps:.1f} fps, 编码 {_video_state['codec']}")
    log_lines.append(f"篮筐: {_calib['hoop']}")

    checked = 0
    detected = 0
    with VideoReader(_video_state["path"]) as reader:
        for fidx, frame in reader.iter_frames(start=0, end=total, batch=3):
            checked += 1
            try:
                ball_det = detect_ball_yolo(frame)
            except Exception:
                ball_det = None
            # 球检测到 → 取位置；漏检 → 传 None，由 detector 用光流+卡尔曼补全
            pos = (ball_det[0], ball_det[1]) if ball_det is not None else None
            if pos is not None:
                detected += 1
            # 把 frame 传给 detector 用于光流追踪
            goal_ts = detector.feed(pos, fidx, fps, frame=frame)
            if goal_ts is not None:
                log_lines.append(f"[进球] 帧 {fidx} ({goal_ts:.1f}s)")

    log_lines.append(f"\n检测完成: 检查 {checked} 帧, 检测到球 {detected} 帧, 共 {len(detector.goals)} 个进球")

    # 剪辑集锦
    if detector.goals:
        from cutter.ffmpeg_cutter import cut_clips
        out_path = str(ROOT / "highlights.mp4")
        try:
            cut_clips(video, detector.goals,
                      pre_roll=int(pre_roll), post_roll=int(post_roll),
                      min_gap=int(min_gap), output_path=out_path)
            log_lines.append(f"集锦已生成: {out_path}")
            return out_path, f"完成！检测到 {len(detector.goals)} 个进球，集锦已生成", "\n".join(log_lines)
        except Exception as e:
            log_lines.append(f"剪辑失败: {e}")
            return None, f"剪辑失败: {e}", "\n".join(log_lines)
    else:
        return None, "未检测到进球", "\n".join(log_lines)


# ============ UI 构建 ============
css = """
.gradio-container {max-width: 1200px !important}
"""


def build_ui():
    with gr.Blocks(title="篮球录像检测工具", css=css) as app:
        gr.Markdown("# 篮球录像进球检测与自动剪辑")

        with gr.Tab("1. 标定篮筐 + YOLO 检测球"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_in = gr.Video(label="上传视频", sources=["upload"])
                    load_btn = gr.Button("1. 加载视频", variant="primary")
                    video_info = gr.Textbox(label="视频信息", interactive=False)
                    transcode_btn = gr.Button("转码预览（HEVC/不可播放时点此）")
                    preview_video = gr.Video(label="预览（已转码）", visible=True)

                    frame_slider = gr.Slider(0, 1, step=1, label="帧号")
                    preview_btn = gr.Button("2. 预览帧")

                    gr.Markdown("**3. 标定篮筐**：在图片上点击左上角和右下角两个点框选篮筐（固定机位只需标一次）")
                    calib_img = gr.Image(label="点击图片框选篮筐（左上角+右下角）",
                                         type="numpy", interactive=True)
                    calib_status = gr.Textbox(label="标定状态", interactive=False)

                    gr.Markdown("**3.5 设置基准帧**：滑动到无球的篮筐画面，点击按钮设为基准帧（帧差法必需）")
                    baseline_btn = gr.Button("设当前帧为基准帧（无球画面）")
                    baseline_status = gr.Textbox(label="基准帧状态", value="未设置", interactive=False)

                    ball_conf_slider = gr.Slider(0.1, 0.9, value=0.25, step=0.05,
                                                 label="球检测置信度阈值（越低越容易检出，误检也多）")
                    use_tta = gr.Checkbox(False, label="TTA 推理增强（慢约3倍，提升小目标检出率）")
                    with gr.Row():
                        track_btn = gr.Button("4. 检测球（测试）", variant="primary")
                        save_btn = gr.Button("保存标定")
                        reset_btn = gr.Button("重置标定")
                    track_status = gr.Textbox(label="检测状态", interactive=False)

                    with gr.Row():
                        start_frame = gr.Number(0, label="起始帧")
                        end_frame = gr.Number(1000, label="结束帧")
                        batch = gr.Number(3, label="抽帧间隔")
                    with gr.Accordion("音频分析（多信号融合）", open=True):
                        gr.Markdown(
                            "**音频峰值检测**：从视频提取音频，检测进球时的欢呼/swish 声峰值。\n"
                            "- 阈值因子越大越严格（默认 4.0，误报多可调到 5-6）\n"
                            "- 分析后点击「检测进球」会自动融合视觉+音频信号"
                        )
                        with gr.Row():
                            audio_threshold_factor = gr.Number(4.0, label="音频阈值因子(2-8)")
                            min_gap_audio = gr.Number(3.0, label="音频峰最小间隔(秒)")
                        audio_btn = gr.Button("4.5 分析音频", variant="secondary")
                        audio_status = gr.Textbox(label="音频分析结果", interactive=False, lines=3)
                        with gr.Row():
                            fusion_window = gr.Number(2.0, label="融合时间窗口(秒, ±窗口)")
                            fusion_mode = gr.Dropdown(
                                ["or", "and", "fused_only", "visual_only"],
                                value="or", label="融合模式",
                                info="or=任一触发(高召回) | and=双信号(高精度) | "
                                     "fused_only=仅融合 | visual_only=仅视觉")
                    with gr.Accordion("进球检测参数（基准帧差法）", open=True):
                        with gr.Row():
                            net_threshold = gr.Number(25, label="差分阈值(15-40)")
                            min_gap = gr.Number(3.0, label="最小进球间隔(秒)")
                        approach_dist = gr.Number(100, label="兼容参数(无作用)")
                        confirm_below = gr.Checkbox(True, label="兼容参数(无作用)")
                        net_help = gr.Markdown(
                            "**基准帧差法**：每帧与基准帧（无球画面）做差分，找运动物体（球）。\n"
                            "- 球从篮筐**上沿→下沿** = 进球\n"
                            "- 差分阈值越小越敏感（误报多），越大越严格（漏报多）\n"
                            "- **必须先设置基准帧**（无球的篮筐画面）\n"
                            "- **抽帧间隔必须设为1**（帧差法需连续帧）\n"
                            "- **融合模式说明**：or=视觉或音频都算 | and=必须都有 | "
                            "fused_only=仅双信号 | visual_only=仅视觉"
                        )
                    detect_btn = gr.Button("5. 检测进球（CV 多信号融合）", variant="primary")
                    detect_status = gr.Textbox(label="检测结果", interactive=False)
                    detect_log = gr.Textbox(label="进球日志", lines=10, interactive=False)

                with gr.Column(scale=1):
                    summary_img = gr.Image(label="进球汇总图（红=融合，黄=单信号）", interactive=False)

            # 事件绑定
            load_btn.click(on_load_video_t1, [video_in],
                           [preview_video, frame_slider, video_info, calib_img])
            transcode_btn.click(on_transcode_preview, [], [preview_video, video_info])
            preview_btn.click(on_preview_t1, [frame_slider], [calib_img, calib_status])
            calib_img.select(on_image_click, [frame_slider], [calib_img, calib_status])
            track_btn.click(on_track_test, [frame_slider, ball_conf_slider, use_tta],
                            [calib_img, track_status])
            save_btn.click(on_save_calib, [], [calib_status])
            reset_btn.click(on_reset_calib, [], [calib_status])
            baseline_btn.click(on_set_baseline, [frame_slider], [baseline_status])
            audio_btn.click(on_analyze_audio,
                            [audio_threshold_factor, min_gap_audio],
                            [audio_status, detect_log])
            detect_btn.click(on_detect_goals,
                             [start_frame, end_frame, batch, ball_conf_slider,
                              approach_dist, confirm_below, min_gap, net_threshold,
                              audio_threshold_factor, fusion_window, fusion_mode],
                             [summary_img, detect_status, detect_log])

        with gr.Tab("2. YOLO 检测可视化（需微调权重）"):
            with gr.Row():
                with gr.Column():
                    yolo_frame = gr.Slider(0, 1, step=1, label="帧号")
                    yolo_conf = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="置信度阈值")
                    yolo_weights = gr.Dropdown(list_weights(), label="权重文件")
                    only_ball = gr.Checkbox(False, label="仅检测球（COCO 32）")
                    yolo_btn = gr.Button("检测", variant="primary")
                yolo_img = gr.Image(label="检测结果")
            yolo_info = gr.Textbox(label="检测信息", interactive=False)
            yolo_btn.click(on_detect_yolo, [yolo_frame, yolo_conf, yolo_weights, only_ball],
                           [yolo_img, yolo_info])

        with gr.Tab("3. 完整流程（标定+检测+剪辑）"):
            gr.Markdown("### 使用前请先在 Tab1 完成篮筐标定")
            with gr.Row():
                pre_roll = gr.Number(5, label="进球前多少秒")
                post_roll = gr.Number(5, label="进球后多少秒")
                min_gap = gr.Number(8, label="最小间隔（秒）")
            run_btn = gr.Button("开始处理", variant="primary")
            pipeline_status = gr.Textbox(label="处理状态", interactive=False)
            pipeline_log = gr.Textbox(label="处理日志", lines=15, interactive=False)
            highlights = gr.Video(label="集锦视频")
            run_btn.click(on_run_pipeline, [video_in, pre_roll, post_roll, min_gap],
                          [highlights, pipeline_status, pipeline_log])

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7863, inbrowser=True, css=css,
               allowed_paths=[_CACHE_ROOT, r"E:\bball_tmp"])
