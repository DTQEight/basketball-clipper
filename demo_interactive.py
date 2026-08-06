"""chonyy 进球检测交互式 demo（Gradio）。

功能：
  1. 上传视频 → 滑动到含篮筐的帧
  2. 点击画面 2 个点标定篮筐（左上角 + 右下角）
  3. 设置起止帧、置信度、最小进球间隔
  4. 点击「开始检测」→ 生成可视化视频
  5. 界面内直接播放结果

用法:
    E:\\bball-env\\python.exe demo_interactive.py
浏览器打开 http://127.0.0.1:7870
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
import cv2
import numpy as np
import gradio as gr

from video_io import get_video_info, read_frame, VideoReader
from app import get_ball_model
from demo_chonyy import ChonyyGoalDetector, draw_frame
from tracker import GoalDetector
from cutter.ffmpeg_cutter import cut_clips
from vlm_verifier import verify_goals_batch, filter_confirmed_goals

# 加载 .env 文件中的 API Key
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


def draw_frame_diff(frame, blob, hoop, detector, frame_idx, fps):
    """基准帧差法的可视化绘制。"""
    out = frame.copy()
    x1, y1, x2, y2 = hoop
    # 篮筐框（绿）+ 上沿/下沿线
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(out, (x1 - 30, y1), (x2 + 30, y1), (0, 255, 255), 1)  # 上沿 黄
    cv2.line(out, (x1 - 30, y2), (x2 + 30, y2), (255, 0, 255), 1)  # 下沿 紫
    cv2.putText(out, "HOOP", (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # 搜索区域（蓝虚线框）
    cv2.rectangle(out, (detector.search_x1, detector.search_y1),
                  (detector.search_x2, detector.search_y2), (255, 200, 0), 1)
    # 运动斑块（红）
    if blob is not None:
        bx1, by1, bx2, by2 = blob
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.putText(out, "MOVING", (bx1, max(by1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # 状态信息
    ts = frame_idx / fps
    above = "Y" if detector.blob_above_hoop else "N"
    info_lines = [
        f"Frame: {frame_idx} ({ts:.1f}s)",
        f"Method: DIFF (baseline)",
        f"Above: {above} | Goals: {len(detector.goals)}",
        f"diff_ratio: {detector.last_diff_ratio:.1f}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(out, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    # 进球时刻闪烁红色边框
    if detector.goals and abs(ts - detector.goals[-1]) < 1.0:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 0, 255), 8)
    return out

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


def on_run_detect(start_frame, end_frame, ball_conf, min_gap_sec, method, progress=gr.Progress()):
    """运行进球检测并生成可视化视频。

    method: "chonyy" 状态机（依赖球 y 轨迹顺序）
            "diff"   基准帧差法（检测运动物体穿越篮筐，适合底角视角）
    """
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

    use_diff = (method == "diff")
    if use_diff and _calib["baseline_frame"] is None:
        return None, "❌ 基准帧差法需要基准帧，请重新点击 2 个点标定篮筐"

    try:
        # 初始化检测器
        if use_diff:
            detector = GoalDetector(hoop, baseline_frame=_calib["baseline_frame"],
                                    min_gap_sec=float(min_gap_sec),
                                    fusion_mode="visual_only")
            method_name = "基准帧差法 (diff)"
        else:
            detector = ChonyyGoalDetector(hoop, min_gap_sec=float(min_gap_sec))
            method_name = "chonyy 状态机"

        # 加载 YOLO（chonyy 需要，diff 不需要但加载以备可视化）
        progress(0, desc="加载 YOLO 模型...")
        model, weights_path = get_ball_model()

        # 输出目录改到 E 盘（避免 C 盘中文路径导致 URL 编码问题）
        out_dir = Path(_CACHE_ROOT) / "demo_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "demo_result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        info = get_video_info(_video_state["path"])
        out_writer = cv2.VideoWriter(out_path, fourcc, fps,
                                     (info["width"], info["height"]))

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

        # 用 VideoReader 顺序读取（一次打开，避免每帧重新 open/seek，提速几十倍）
        reader = VideoReader(_video_state["path"])
        try:
            for fidx, frame in reader.iter_frames(start=start, end=end, batch=1):
                # YOLO 检测球（chonyy 用，diff 仅作诊断对比）
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

                # 喂入检测器 + 可视化
                if use_diff:
                    detector.feed(None, fidx, fps, frame=frame)
                    blob = detector.last_blob_box
                    if blob is not None:
                        diag["blob_detected"] += 1
                    annotated = draw_frame_diff(frame, blob, hoop, detector, fidx, fps)
                else:
                    detector.feed(ball_pos, fidx, fps)
                    annotated = draw_frame(frame, ball_pos, hoop, detector, fidx, fps)
                out_writer.write(annotated)

                processed += 1
                if processed % 30 == 0:
                    progress(processed / n_frames,
                             desc=f"检测中 {processed}/{n_frames} | 进球: {len(detector.goals)}")
        finally:
            reader.close()
            out_writer.release()

        # 基准帧差法结束后调用 finalize（处理仅音频候选，这里无音频所以无影响）
        if use_diff:
            detector.finalize()

        elapsed = time.time() - t0

        # 转码为 H.264 以便浏览器播放（用 subprocess 替代 os.system，更可靠）
        progress(0.99, desc="转码为 H.264...")
        try:
            import imageio_ffmpeg
            import subprocess
            ff = imageio_ffmpeg.get_ffmpeg_exe()
            h264_path = str(out_dir / "demo_h264.mp4")
            subprocess.run([ff, "-y", "-i", out_path, "-c:v", "libx264",
                            "-crf", "23", "-movflags", "+faststart", h264_path],
                           creationflags=0x08000000, capture_output=True, timeout=600)
            if os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
                os.remove(out_path)
                out_path = h264_path
        except Exception as e:
            pass

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
        extra = ""
        if use_diff:
            d = detector.diag
            extra = (f"运动斑块检测: {diag['blob_detected']}/{processed} 帧 "
                     f"({diag['blob_detected']/max(processed,1)*100:.0f}%)\n"
                     f"━━━ diff 拒绝原因 ━━━\n"
                     f"  到达上方: {d['cross_above']} | 在框内: {d['in_hoop']} | 到达下方: {d['cross_below']}\n"
                     f"  冷却跳过: {d['reject_cooldown']} | 无斑块: {d['reject_no_blob']}\n"
                     f"  无上方直接到下方: {d['reject_no_above']} | x不在范围: {d['reject_in_x']}\n"
                     f"  斑块太宽: {d['reject_size']} | 趋势失败: {d['reject_trend']}\n"
                     f"  侧向进筐成功: {d['side_goal']} | 超时匹配成功: {d['timeout_goal']}\n")
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
        return out_path, status
    except Exception as e:
        import traceback
        return None, f"❌ 检测失败: {e}\n\n{traceback.format_exc()}"


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


def on_vlm_verify(api_key, min_confidence, progress=gr.Progress()):
    """用 VLM 对 CV 检测到的候选进球做二次验证。"""
    if _video_state["path"] is None:
        return "❌ 请先加载视频并检测进球", ""
    if not _last_goals:
        return "❌ 没有候选进球（请先点「开始检测」）", ""
    if not api_key or not api_key.strip():
        return "❌ 请输入 Mimo API Key", ""

    api_key = api_key.strip()
    n = len(_last_goals)
    progress(0.02, desc=f"开始 VLM 验证（{n} 个候选）...")

    log_lines = [f"VLM 二次验证开始（{n} 个候选进球）"]
    log_lines.append(f"API: Mimo v2.5 | 窗口: ±3秒 | 最低置信度: {min_confidence}")
    log_lines.append("")

    confirmed = []

    def on_progress(current, total, result):
        pct = current / total
        ts = result["timestamp"]
        status = "✅进球" if result["is_goal"] else "❌否决"
        conf = result["confidence"]
        reason = result.get("reason", "")[:50]
        err = result.get("error", "")
        line = f"[{current}/{total}] {ts:.1f}s → {status} (conf={conf:.2f})"
        if err:
            line += f" ⚠️{err[:30]}"
        elif reason:
            line += f" {reason}"
        log_lines.append(line)
        if result["is_goal"] and conf >= min_confidence:
            confirmed.append(ts)
        progress(pct, desc=f"VLM 验证中 {current}/{total}")

    results = verify_goals_batch(
        _video_state["path"], list(_last_goals), api_key,
        progress_callback=on_progress,
    )

    confirmed = filter_confirmed_goals(results, min_confidence=min_confidence)

    log_lines.append("")
    log_lines.append(f"━━━ VLM 验证结果 ━━━")
    log_lines.append(f"候选: {n} 个 | 确认进球: {len(confirmed)} 个 | "
                     f"否决: {n - len(confirmed)} 个")
    if confirmed:
        log_lines.append("确认的进球时间: " +
                         ", ".join([f"{t:.1f}s" for t in confirmed]))

    # 更新全局进球列表为 VLM 确认的结果（供「生成集锦」使用）
    _last_goals.clear()
    _last_goals.extend(confirmed)

    progress(1.0, desc="VLM 验证完成")
    status = (f"✅ VLM 验证完成\n"
              f"候选 {n} 个 → 确认 {len(confirmed)} 个 → 否决 {n - len(confirmed)} 个\n"
              f"已更新进球列表，可点「生成集锦」剪辑确认的进球")
    return status, "\n".join(log_lines)


# ============ Gradio 界面 ============
# 预填已知视频路径
_DEFAULT_VIDEO = r"D:\Downloads\highlights.mp4"

with gr.Blocks(title="篮球进球检测交互式 Demo") as demo:
    gr.Markdown("# 🏀 篮球进球检测交互式 Demo")
    gr.Markdown("输入视频路径 → 滑动到篮筐画面 → 点击 2 点标定篮筐 → 选择算法 → 开始检测")

    with gr.Row():
        with gr.Column(scale=1):
            video_path_input = gr.Textbox(
                label="1. 视频文件路径（避免上传大文件中断）",
                value=_DEFAULT_VIDEO,
                placeholder=r"例如: D:\Downloads\xxx.mp4",
                lines=1)
            load_btn = gr.Button("📥 加载视频", variant="primary")
            info_text = gr.Textbox(label="视频信息", interactive=False)

            frame_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1,
                                     label="2. 帧选择器（滑动到篮筐画面）")
            preview_image = gr.Image(label="3. 点击画面 2 个点标定篮筐（左上 + 右下）",
                                     type="numpy", interactive=False)
            click_status = gr.Textbox(label="标定状态", interactive=False)
            reset_btn = gr.Button("🔄 重置篮筐标定")

            with gr.Row():
                start_frame = gr.Number(label="起始帧", value=0, precision=0)
                end_frame = gr.Number(label="结束帧 (0=到末尾)", value=600, precision=0)
            with gr.Row():
                ball_conf = gr.Slider(0.1, 0.9, value=0.3, step=0.05,
                                      label="球检测置信度")
                min_gap = gr.Slider(1.0, 10.0, value=3.0, step=0.5,
                                    label="最小进球间隔(秒)")
            method = gr.Radio(
                choices=["chonyy", "diff"],
                value="diff",
                label="检测算法",
                info="chonyy=状态机(需球从上方穿越) | diff=基准帧差法(适合底角视角)")

            run_btn = gr.Button("🚀 开始检测", variant="primary")

        with gr.Column(scale=1):
            result_video = gr.Video(label="检测结果视频")
            result_status = gr.Textbox(label="检测统计", interactive=False, lines=12)

            gr.Markdown("---\n### ✂️ 自动剪辑集锦")
            gr.Markdown("基于上方检测到的进球时间戳，自动裁剪片段并拼接成集锦 mp4")
            with gr.Row():
                pre_roll = gr.Number(label="进球前保留(秒)", value=5, precision=0)
                post_roll = gr.Number(label="进球后保留(秒)", value=5, precision=0)
                cut_min_gap = gr.Number(label="最小合并间隔(秒)", value=8, precision=0)
            cut_btn = gr.Button("✂️ 生成集锦", variant="primary")
            highlights_video = gr.Video(label="集锦视频")
            highlights_status = gr.Textbox(label="剪辑状态", interactive=False, lines=4)

            gr.Markdown("---\n### 🤖 VLM 大模型二次验证（推荐）")
            gr.Markdown("CV 检测的候选进球 → VLM 逐个确认 → 过滤误报。"
                        "解决 CV 漏检/误报问题，提高准确率")
            vlm_api_key = gr.Textbox(
                label="Mimo API Key",
                value=os.environ.get("MIMO_API_KEY", ""),
                type="password",
                placeholder="tp-xxxxx",
                info="从 https://platform.xiaomimimo.com 获取")
            vlm_conf = gr.Slider(0.0, 1.0, value=0.5, step=0.1,
                                 label="最低确认置信度")
            vlm_btn = gr.Button("🤖 VLM 验证候选进球", variant="primary")
            vlm_status = gr.Textbox(label="验证状态", interactive=False, lines=3)
            vlm_log = gr.Textbox(label="验证日志", interactive=False, lines=12)

    # 事件绑定
    load_btn.click(on_load_video, inputs=[video_path_input],
                   outputs=[preview_image, frame_slider, info_text])
    frame_slider.change(on_preview, inputs=[frame_slider],
                        outputs=[preview_image, click_status])
    preview_image.select(on_image_click, inputs=[frame_slider],
                         outputs=[preview_image, click_status])
    reset_btn.click(on_reset_hoop, outputs=[click_status])
    run_btn.click(on_run_detect,
                  inputs=[start_frame, end_frame, ball_conf, min_gap, method],
                  outputs=[result_video, result_status])
    cut_btn.click(on_generate_highlights,
                  inputs=[pre_roll, post_roll, cut_min_gap],
                  outputs=[highlights_video, highlights_status])
    vlm_btn.click(on_vlm_verify,
                  inputs=[vlm_api_key, vlm_conf],
                  outputs=[vlm_status, vlm_log])


if __name__ == "__main__":
    _out_dir = str(Path(_CACHE_ROOT) / "demo_output")
    _gradio_dir = str(Path(_CACHE_ROOT) / "gradio")
    os.makedirs(_out_dir, exist_ok=True)
    os.makedirs(_gradio_dir, exist_ok=True)
    demo.launch(server_name="127.0.0.1", server_port=7870,
                show_error=True, prevent_thread_lock=False,
                max_file_size=5000*1024*1024,  # 5GB 上限
                allowed_paths=[_CACHE_ROOT])  # 允许整个 bball_cache（含 gradio 会话子目录）
