"""检测/剪辑/批量业务逻辑。

从原 demo_nicegui.py 抽离，通过 services.state 模块访问/修改运行时状态，
UI 层只需调用本模块的函数并传入参数。
"""
import os
import time
import json
from pathlib import Path

import cv2
import numpy as np

from . import state
from . import video_utils
from .video_utils import frame_to_base64, scan_video_files

# 项目根目录（basketball-clipper/）
_ROOT = Path(__file__).parent.parent.resolve()
import sys
sys.path.insert(0, str(_ROOT))

from video_io import get_video_info, read_frame, VideoReader
from app import get_ball_model, get_device
from tracker import GoalDetector
from cutter.ffmpeg_cutter import cut_clips, _build_encode_args


# ============ 单视频操作 ============

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
    state.video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                             codec=info["codec"], current_frame=0,
                             width=info["width"], height=info["height"])
    state.calib["clicks"] = []
    state.calib["hoop"] = None
    state.calib["baseline_frame"] = None
    state.calib["baseline_idx"] = -1
    frame = read_frame(video_path, 0, total=info["total"], fps=info["fps"])
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")
    return preview, info_str


def preview_frame(frame_idx):
    """预览指定帧。"""
    if state.video_state["path"] is None:
        return None, "请先加载视频"
    frame = read_frame(state.video_state["path"], int(frame_idx),
                       total=state.video_state["total"], fps=state.video_state["fps"])
    if frame is None:
        return None, "读取帧失败"
    out = frame.copy()
    if state.calib["hoop"]:
        x1, y1, x2, y2 = state.calib["hoop"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(out, "HOOP", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.line(out, (x1 - 30, y1), (x2 + 30, y1), (0, 255, 255), 1)
        cv2.line(out, (x1 - 30, y2), (x2 + 30, y2), (255, 0, 255), 1)
    for i, (x, y) in enumerate(state.calib["clicks"]):
        cv2.circle(out, (x, y), 10, (255, 255, 0), -1)
        cv2.putText(out, str(i + 1), (x - 6, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    ts = int(frame_idx) / state.video_state["fps"]
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), f"帧 {frame_idx} ({ts:.1f}s)"


def click_calibrate(x, y):
    """点击标定篮筐。"""
    if state.video_state["path"] is None:
        return None, "请先加载视频"
    frame_idx = state.video_state["current_frame"]
    state.calib["clicks"].append((x, y))
    status = f"点击 ({x},{y})，已收集 {len(state.calib['clicks'])}/2 个点"
    if len(state.calib["clicks"]) >= 2:
        p1, p2 = state.calib["clicks"][:2]
        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
        state.calib["hoop"] = (x1, y1, x2, y2)
        base_frame = read_frame(state.video_state["path"], int(frame_idx),
                               total=state.video_state["total"], fps=state.video_state["fps"])
        if base_frame is not None:
            state.calib["baseline_frame"] = base_frame.copy()
            state.calib["baseline_idx"] = int(frame_idx)
        status = f"篮筐已标定: ({x1},{y1}) - ({x2},{y2}) | 基准帧: 第 {int(frame_idx)} 帧"
        state.calib["clicks"] = []
    frame = read_frame(state.video_state["path"], int(frame_idx),
                       total=state.video_state["total"], fps=state.video_state["fps"])
    if frame is None:
        return None, status
    out = frame.copy()
    if state.calib["hoop"]:
        x1, y1, x2, y2 = state.calib["hoop"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(out, "HOOP", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.line(out, (x1 - 30, y1), (x2 + 30, y1), (0, 255, 255), 1)
        cv2.line(out, (x1 - 30, y2), (x2 + 30, y2), (255, 0, 255), 1)
    for i, (px, py) in enumerate(state.calib["clicks"]):
        cv2.circle(out, (px, py), 10, (255, 255, 0), -1)
        cv2.putText(out, str(i + 1), (px - 6, py - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), status


def reset_hoop():
    """重置篮筐标定。"""
    state.calib["clicks"] = []
    state.calib["hoop"] = None
    state.calib["baseline_frame"] = None
    state.calib["baseline_idx"] = -1
    return "已重置，请重新点击 2 个点标定篮筐"


# ============ 预览片段生成 ============

def _generate_preview_clips(video_path, goals, start, end, fps, total, stamp,
                            progress_callback=None, cancel_check=None):
    """为进球时间戳列表生成预览片段（480p 低分辨率，用于 UI 内预览）。

    返回 clips 列表 [{"ts": float, "path": str, "idx": int}, ...]
    """
    import imageio_ffmpeg
    import subprocess as _sp

    out_dir = Path(state.CACHE_ROOT) / "demo_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    # 编码参数只检测一次（NVENC 探测是子进程，循环内重复调用会显著拖慢）
    _enc = _build_encode_args(ff, quality="preview")
    clip_half = int(fps * 3)

    clips = []
    for gi, gts in enumerate(goals):
        if cancel_check and cancel_check():
            break
        gframe = int(gts * fps)
        seg_start = max(start, gframe - clip_half)
        seg_end = min(end, gframe + clip_half)
        if seg_end <= seg_start:
            continue
        clip_path = str(out_dir / f"goal_{gi}_{int(gts)}s_{stamp}.mp4")
        seg_start_sec = seg_start / fps
        seg_dur_sec = (seg_end - seg_start) / fps
        try:
            _sp.run([ff, "-y", "-loglevel", "error",
                     "-ss", f"{seg_start_sec:.3f}", "-i", video_path,
                     "-t", f"{seg_dur_sec:.3f}",
                     "-vf", "scale=-2:480"] + _enc +
                    ["-movflags", "+faststart", clip_path],
                    creationflags=state.SBOX, capture_output=True, timeout=60)
            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                clips.append({"ts": gts, "path": clip_path, "idx": gi})
        except Exception as e:
            print(f"[WARN] 预览片段生成失败 ({gts:.1f}s): {e}", flush=True)
        if progress_callback and len(goals) > 0:
            progress_callback(80 + 18 * (gi + 1) / len(goals),
                              f'生成片段 {gi+1}/{len(goals)}')
    return clips


# ============ 检测 ============

def run_detect(start_frame, end_frame, ball_conf, min_gap_sec,
               diff_threshold=15, min_circularity=0.35, min_in_hoop_frames=2,
               min_blob_area=30, search_margin=80, progress_callback=None,
               auto_threshold=True, yolo_step=2):
    """运行进球检测。返回 (结果文本, 是否成功)。"""
    if state.video_state["path"] is None:
        return "❌ 请先加载视频", False
    if state.calib["hoop"] is None:
        return "❌ 请先点击画面标定篮筐", False
    if state.calib["baseline_frame"] is None:
        return "❌ 基准帧差法需要基准帧，请重新标定", False

    def _report(pct, msg):
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    hoop = state.calib["hoop"]
    fps = state.video_state["fps"]
    total = state.video_state["total"]
    start = int(start_frame)
    end = int(end_frame) if end_frame and int(end_frame) > 0 else total
    end = min(end, total)
    if end <= start:
        return "❌ 结束帧必须大于起始帧", False

    _report(5, '初始化检测器...')
    try:
        detector = GoalDetector(hoop, baseline_frame=state.calib["baseline_frame"],
                                min_gap_sec=float(min_gap_sec),
                                diff_threshold=int(diff_threshold),
                                min_blob_area=int(min_blob_area),
                                search_margin=int(search_margin),
                                loose_mode=True,
                                yolo_confirm=True, rolling_baseline_sec=60.0,
                                min_circularity=float(min_circularity),
                                min_in_hoop_frames=int(min_in_hoop_frames),
                                auto_threshold=bool(auto_threshold))
        _report(10, '加载 YOLO 模型...')
        model, _ = get_ball_model()

        t0 = time.time()
        t0_str = time.strftime('%H:%M:%S', time.localtime(t0))
        processed = 0
        n_frames = end - start
        video_dur_min = n_frames / max(fps, 1) / 60.0

        # ============ DEBUG: 开始信息 ============
        print("\n" + "=" * 68)
        print(f"[DETECT START] {t0_str}")
        print(f"  Video       : {state.video_state['path']}")
        print(f"  Resolution  : {state.video_state.get('width', '?')}x{state.video_state.get('height', '?')}  @ {fps:.1f} fps")
        print(f"  Frames      : {start}-{end-1}  ({n_frames} total, {video_dur_min:.1f} min)")
        print(f"  Auto-thresh : {bool(auto_threshold)}")
        print(f"  YOLO step   : every {yolo_step} frames  ({100/yolo_step:.0f}% coverage)")
        print(f"  ball_conf   : {ball_conf}  min_gap : {min_gap_sec}s")
        print(f"  min_circ    : {min_circularity}  in_hoop_f : {min_in_hoop_frames}")
        print("=" * 68, flush=True)
        _debug_last_print_time = t0
        _debug_last_processed = 0

        _report(15, f'开始检测 {n_frames} 帧...')
        reader = VideoReader(state.video_state["path"])
        # 跳帧检测：每 N 帧跑一次 YOLO，跳过的帧复用上一帧 ball_pos（只跑 diff）
        # 篮球下落速度 ~8m/s，30fps 下每帧位移 <0.3m，连续 2 帧丢失不会漏检
        _yolo_step = max(1, int(yolo_step))
        _last_ball_pos = None
        try:
            for fidx, frame in reader.iter_frames(start=start, end=end, batch=1):
                if state.cancel_requested:
                    break
                ball_pos = None
                need_yolo = False
                if auto_threshold and not detector._warmup_done:
                    # 预热期：只收集 P95 统计量，跳过 YOLO
                    _last_ball_pos = None
                else:
                    # 正式检测：每 _yolo_step 帧一次 YOLO，其余帧复用上一帧结果
                    need_yolo = ((processed % _yolo_step) == 0)
                    if need_yolo:
                        try:
                            # classes=[0]：ultralytics 在 NMS 后直接过滤，只保留 basketball 类
                            # 省去 CPU 侧全量 .cpu().numpy() 拷贝 + 遍历查找
                            # 注：basketball_custom.pt 的 names={0:'basketball'}，id 固定为 0
                            res = model.predict(frame, conf=float(ball_conf), imgsz=960,
                                                classes=[0],
                                                device=get_device(), verbose=False)[0]
                            if res.boxes is not None and len(res.boxes) > 0:
                                xyxy = res.boxes.xyxy.cpu().numpy()
                                confs = res.boxes.conf.cpu().numpy()
                                best = int(np.argmax(confs))
                                x1, y1, x2, y2 = xyxy[best]
                                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                                ball_pos = (float(cx), float(cy), float(x1), float(y1),
                                            float(x2), float(y2), float(confs[best]))
                            _last_ball_pos = ball_pos
                        except Exception:
                            pass
                    else:
                        # 跳帧：复用最近一次 YOLO 结果（只在篮筐附近有效）
                        ball_pos = _last_ball_pos
                detector.feed(ball_pos, fidx, fps, frame=frame)
                processed += 1
                # 每 10 帧更新一次进度
                if processed % 10 == 0:
                    pct = 15 + 60 * processed / n_frames
                    if auto_threshold and not detector._warmup_done:
                        phase = '预热'
                    elif need_yolo:
                        phase = '检测'
                    else:
                        phase = '跳帧'
                    _report(pct, f'{phase}帧 {processed}/{n_frames} ({processed*100//n_frames}%)')
                # 定期释放 CUDA 缓存：每 500 帧一次（100 帧太频繁，empty_cache 本身会同步阻塞）
                if processed % 500 == 0:
                    try:
                        import torch as _torch
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                    except Exception:
                        pass
                # ============ DEBUG: 周期进度（每 30 秒打印一次）============
                _now = time.time()
                if (_now - _debug_last_print_time) >= 30.0 and processed < n_frames:
                    _dt = _now - _debug_last_print_time
                    _df = processed - _debug_last_processed
                    _instant_fps = _df / max(_dt, 0.001)
                    _overall_fps = processed / max((_now - t0), 0.001)
                    _pct = processed / max(n_frames, 1)
                    _eta = (n_frames - processed) / max(_overall_fps, 0.1)
                    _phase_label = '预热' if (auto_threshold and not detector._warmup_done) else '检测'
                    print(f"[{time.strftime('%H:%M:%S')}] {_phase_label} {processed:>6d}/{n_frames:<6d} "
                          f"({_pct*100:5.1f}%) | 瞬时 {_instant_fps:>5.0f} f/s  平均 {_overall_fps:>5.1f} f/s | "
                          f"ETA {_eta/60:>4.1f} min | 进球累计 {len(detector.goals):>3d}", flush=True)
                    _debug_last_print_time = _now
                    _debug_last_processed = processed
        finally:
            reader.close()

        _t1 = time.time()
        _detect_elapsed = _t1 - t0
        _report(80, '生成预览片段...')

        # 生成预览片段
        goals = sorted(detector.goals)
        state.last_goal_clips.clear()
        state.kept_goal_indices.clear()

        _stamp = int(time.time())
        state.last_goal_clips.extend(
            _generate_preview_clips(state.video_state["path"], goals, start, end,
                                    fps, total, _stamp, progress_callback=_report,
                                    cancel_check=lambda: state.cancel_requested)
        )

        if state.cancel_requested:
            # 用户取消：清空本次部分生成的片段，不写入历史
            _t2 = time.time()
            state.last_goal_clips.clear()
            state.kept_goal_indices.clear()
            state.last_goals.clear()
            print(f"[{time.strftime('%H:%M:%S')}] [CANCELLED] 已处理 {processed} 帧，取消后退出", flush=True)
            return f"已取消 | 已处理 {processed} 帧", False

        # 检测成功后写入片段缓存并持久化
        if state.last_goal_clips:
            ckey = (state.video_state["path"], tuple(round(g, 3) for g in goals))
            state.clip_cache[ckey] = list(state.last_goal_clips)
            if len(state.clip_cache) > 20:
                state.clip_cache.pop(next(iter(state.clip_cache)))
            state.save_clip_cache()

        _report(100, '完成！')
        state.kept_goal_indices = set(range(len(state.last_goal_clips)))
        state.last_goals.clear()
        state.last_goals.extend(detector.goals)

        d = detector.diag
        total_yolo = d['yolo_confirmed'] + d['yolo_rejected']
        confirm_rate = d['yolo_confirmed'] / max(total_yolo, 1) * 100

        _t2 = time.time()
        _total_elapsed = _t2 - t0
        _preview_elapsed = _t2 - _t1
        _proc_fps = processed / max(_detect_elapsed, 0.001)
        _end_str = time.strftime('%H:%M:%S', time.localtime(_t2))

        # ============ DEBUG: 结束统计 ============
        print("")
        print("=" * 68)
        print(f"[DETECT  END] {_end_str}")
        print(f"  Timing      : detect {_detect_elapsed/60:.1f} min + preview {_preview_elapsed/60:.1f} min = {_total_elapsed/60:.1f} min total")
        print(f"  Speed       : {_proc_fps:.1f} frames/sec  (video {video_dur_min:.1f} min / detect {_detect_elapsed/60:.1f} min = {video_dur_min/max(_detect_elapsed/60,0.001):.2f}x vs realtime)")
        print(f"  Goals       : {len(goals)} detected")
        if goals:
            _timestamps_str = ", ".join(f"{g:.1f}s" for g in goals[:8]) + ("..." if len(goals) > 8 else "")
            print(f"                {_timestamps_str}")
        print(f"  YOLO 确认   : {d['yolo_confirmed']}/{total_yolo} ({confirm_rate:.0f}%)  |  "
              f"上方: {d['cross_above']}  下方: {d['cross_below']}  筐内: {d['in_hoop']}  冷却拒: {d['reject_cooldown']}")
        if detector.auto_threshold:
            if detector._auto_threshold_value is not None:
                if detector._warmup_p95_median is not None:
                    print(f"  AutoThresh  : value={detector._auto_threshold_value}  |  median(P95)={detector._warmup_p95_median:.1f}  +8  clamp[8,50]  |  samples={detector._warmup_sample_count}")
                else:
                    print(f"  AutoThresh  : value={detector._auto_threshold_value}  |  (P95 数据未保存)")
            else:
                print(f"  AutoThresh  : 预热未结束  warmup_done={detector._warmup_done}")
        else:
            print(f"  DiffThresh  : 固定 {diff_threshold}")
        print("=" * 68 + "\n", flush=True)

        status = (f"检测完成 | 处理 {processed} 帧 | 耗时 {_total_elapsed:.0f}s\n"
                  f"进球: {len(detector.goals)} 个 | "
                  f"YOLO确认: {d['yolo_confirmed']}/{total_yolo} ({confirm_rate:.0f}%)")
        if detector.auto_threshold and detector._auto_threshold_value is not None:
            status += f"\n自适应阈值: {detector._auto_threshold_value} (P95+8)"
        state.add_history(state.video_state["path"], hoop, detector.goals, detector.goals,
                          baseline_idx=state.calib["baseline_idx"])
        return status, True
    except Exception as e:
        import traceback
        return f"❌ 检测失败: {e}\n{traceback.format_exc()}", False


def clip_action(action, idx):
    """处理卡片按钮操作。"""
    if idx < 0 or idx >= len(state.last_goal_clips):
        return None, ""
    if action == "preview":
        return state.last_goal_clips[idx]["path"], f"▶ 正在预览第 {idx+1} 个片段"
    elif action == "export":
        return state.last_goal_clips[idx]["path"], f"已导出: {state.last_goal_clips[idx]['path']}"
    elif action == "delete":
        ts = state.last_goal_clips[idx]["ts"]
        del state.last_goal_clips[idx]
        new_kept = set()
        for old_i in sorted(state.kept_goal_indices):
            if old_i < idx:
                new_kept.add(old_i)
            elif old_i > idx:
                new_kept.add(old_i - 1)
        state.kept_goal_indices = new_kept
        kept_ts = [state.last_goal_clips[i]["ts"]
                   for i in sorted(state.kept_goal_indices) if i < len(state.last_goal_clips)]
        state.last_goals.clear()
        state.last_goals.extend(kept_ts)
        return None, f"已删除第 {idx+1} 个片段（{ts:.1f}s）| 剩余 {len(state.last_goal_clips)} 个"
    return None, ""


def generate_highlights(pre_roll, post_roll, min_gap, progress_callback=None):
    """生成集锦视频。"""
    if state.video_state["path"] is None:
        return None, "❌ 请先加载视频并检测进球"
    if not state.last_goals:
        return None, "❌ 没有检测到进球"
    try:
        out_path = cut_clips(state.video_state["path"], list(state.last_goals),
                             pre_roll=int(pre_roll), post_roll=int(post_roll),
                             min_gap=int(min_gap),
                             progress_callback=progress_callback)
        if out_path and os.path.exists(out_path):
            n = len(state.last_goals)
            return out_path, f"集锦已生成（{n} 个进球片段）\n输出: {out_path}"
        return None, "❌ 集锦生成失败"
    except Exception as e:
        import traceback
        return None, f"❌ 剪辑失败: {e}\n{traceback.format_exc()}"


def on_load_history(idx_choice, progress_callback=None):
    """从历史记录加载。"""
    def _report(pct, msg):
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    _report(5, '读取历史记录...')
    if idx_choice is None:
        return None, "请先选择一条历史记录", ""
    records = state.load_history()
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
    state.video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                             codec=info["codec"], current_frame=0,
                             width=info["width"], height=info["height"])
    hoop = r.get("hoop")
    if hoop and len(hoop) == 4:
        state.calib["hoop"] = tuple(int(v) for v in hoop)
        state.calib["clicks"] = []
        # 用保存的标定帧号读取基准帧；旧记录无该字段则回退到第 0 帧
        saved_baseline_idx = int(r.get("baseline_idx", 0))
        base_frame = read_frame(video_path, saved_baseline_idx,
                                total=info["total"], fps=info["fps"])
        if base_frame is not None:
            state.calib["baseline_frame"] = base_frame.copy()
            state.calib["baseline_idx"] = saved_baseline_idx
        else:
            state.calib["baseline_frame"] = None
            state.calib["baseline_idx"] = -1
    kept = [float(t) for t in r.get("kept_goals", [])]
    all_goals = [float(t) for t in r.get("goals", [])]
    state.last_goals.clear()
    state.last_goals.extend(kept if kept else all_goals)

    fps = info["fps"]
    total = info["total"]
    _stamp = int(time.time())
    state.last_goal_clips.clear()
    state.kept_goal_indices.clear()

    _report(30, f'生成 {len(all_goals)} 个预览片段...')
    cache_key = (video_path, tuple(round(t, 3) for t in all_goals))
    cached = state.clip_cache.get(cache_key)
    if cached and all(os.path.exists(c["path"]) for c in cached):
        # 命中缓存：直接复用已生成的片段，跳过 ffmpeg
        state.last_goal_clips.extend(list(cached))
        _report(30, f'命中缓存，复用 {len(state.last_goal_clips)} 个片段')
    elif all_goals:
        state.last_goal_clips.extend(
            _generate_preview_clips(video_path, all_goals, 0, total,
                                    fps, total, _stamp, progress_callback=_report)
        )
        # 写入缓存并持久化
        if state.last_goal_clips:
            state.clip_cache[cache_key] = list(state.last_goal_clips)
            if len(state.clip_cache) > 20:
                state.clip_cache.pop(next(iter(state.clip_cache)))
            state.save_clip_cache()

    kept_set = set(kept)
    for i, clip in enumerate(state.last_goal_clips):
        if clip["ts"] in kept_set or not kept:
            state.kept_goal_indices.add(i)

    frame = read_frame(video_path, 0, total=total, fps=fps)
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")
    status = (f"已加载历史记录\n视频: {r.get('video_name', '')}\n"
              f"进球: {len(all_goals)} 个 | 保留: {len(kept)} 个\n"
              f"已生成 {len(state.last_goal_clips)} 个预览片段")
    return preview, info_str, status


# ============ 文件夹批量模式 ============

def on_batch_load_video(selected):
    """批量模式：加载选中的视频，并应用该视频已保存的标定。"""
    if not selected or not state.batch_files:
        return None, "", "请先扫描文件夹并选择视频"
    video_path = selected
    state.batch_current_video = video_path
    if video_path in state.batch_calibs:
        cal = state.batch_calibs[video_path]
        state.calib["hoop"] = cal["hoop"]
        state.calib["baseline_frame"] = cal["baseline_frame"]
        state.calib["baseline_idx"] = cal["baseline_idx"]
        state.calib["clicks"] = []
    else:
        state.calib["hoop"] = None
        state.calib["baseline_frame"] = None
        state.calib["baseline_idx"] = -1
        state.calib["clicks"] = []
    try:
        info = get_video_info(video_path)
    except Exception as e:
        return None, "", f"读取失败: {e}"
    state.video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                             codec=info["codec"], current_frame=0,
                             width=info["width"], height=info["height"])
    frame = read_frame(video_path, 0, total=info["total"], fps=info["fps"])
    if frame is not None and state.calib["hoop"]:
        x1, y1, x2, y2 = state.calib["hoop"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, "HOOP", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")
    status = (f"已加载: {os.path.basename(video_path)}\n"
              f"{'已标定' if video_path in state.batch_calibs else '未标定，请点击画面 2 个点标定'}")
    return preview, info_str, status


def on_batch_save_calib():
    """保存当前标定到当前批量视频。"""
    if state.batch_current_video is None:
        return "请先从列表选择视频"
    if state.calib["hoop"] is None or state.calib["baseline_frame"] is None:
        return "请先标定篮筐"
    state.batch_calibs[state.batch_current_video] = {
        "hoop": state.calib["hoop"],
        "baseline_frame": state.calib["baseline_frame"].copy(),
        "baseline_idx": state.calib["baseline_idx"],
    }
    n_calib = len(state.batch_calibs)
    n_total = len(state.batch_files)
    status = f"已保存: {os.path.basename(state.batch_current_video)} | 已标定: {n_calib}/{n_total}"
    if n_calib >= n_total:
        status += "，全部标定完成，可点击「批量识别」"
    return status


def run_batch_detect(start_frame, end_frame, ball_conf, min_gap_sec,
                     diff_threshold=15, min_circularity=0.35, min_in_hoop_frames=2,
                     min_blob_area=30, search_margin=80, progress_callback=None,
                     auto_threshold=True, yolo_step=2):
    """批量识别：遍历文件夹内全部视频逐个检测，每个视频独立写入历史。

    返回 (状态文本, 是否成功)。状态文本逐条列出每个视频的结果，
    未标定/读取失败/检测失败都会单独说明，不再静默跳过。
    """
    if not state.batch_files:
        return "请先加载文件夹", False
    # 当前视频若已标定但未点「保存标定」，批量前自动保存，避免漏处理
    cur = state.batch_current_video
    if (cur and cur in state.batch_files and cur not in state.batch_calibs
            and state.calib["hoop"] is not None and state.calib["baseline_frame"] is not None):
        state.batch_calibs[cur] = {
            "hoop": state.calib["hoop"],
            "baseline_frame": state.calib["baseline_frame"].copy(),
            "baseline_idx": state.calib["baseline_idx"],
        }
    lines = []
    n_ok = 0
    total_goals = 0
    n_total = len(state.batch_files)
    cancelled = False
    _batch_t0 = time.time()
    # ============ DEBUG: BATCH 开始 ============
    print("\n" + "#" * 68)
    print(f"[BATCH START] {time.strftime('%H:%M:%S')}  |  {n_total} videos")
    print(f"  Auto-thresh : {bool(auto_threshold)}")
    print(f"  YOLO step   : every {yolo_step} frames")
    print(f"  ball_conf   : {ball_conf}  min_gap : {min_gap_sec}s")
    print("#" * 68, flush=True)
    for i, video_path in enumerate(state.batch_files):
        if state.cancel_requested:
            cancelled = True
            break
        name = os.path.basename(video_path)
        print(f"\n>>> [{i+1}/{n_total}] {name} <<<", flush=True)
        if video_path not in state.batch_calibs:
            lines.append(f"✗ {name}: 未标定，跳过")
            print(f"    ↳ SKIP (未标定)", flush=True)
            continue
        cal = state.batch_calibs[video_path]
        try:
            info = get_video_info(video_path)
        except Exception as e:
            lines.append(f"✗ {name}: 读取失败 ({e})")
            print(f"    ↳ READ FAIL: {e}", flush=True)
            continue
        state.video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                                  codec=info["codec"], current_frame=0,
                                  width=info["width"], height=info["height"])
        state.calib["hoop"] = cal["hoop"]
        state.calib["baseline_frame"] = cal["baseline_frame"]
        state.calib["baseline_idx"] = cal["baseline_idx"]
        state.calib["clicks"] = []

        def _cb(pct, msg, vname=name, idx=i):
            if progress_callback:
                try:
                    # 归一化：每个视频占 1/n_total 份，pct 为当前视频的 0-100
                    overall = (idx + max(0, min(100, pct)) / 100.0) / n_total * 100.0
                    progress_callback(overall, f'[{idx+1}/{n_total}] {vname} · {msg}')
                except Exception:
                    pass

        try:
            _status, ok = run_detect(start_frame, end_frame, ball_conf, min_gap_sec,
                                     diff_threshold, min_circularity, min_in_hoop_frames,
                                     min_blob_area, search_margin, progress_callback=_cb,
                                     auto_threshold=auto_threshold, yolo_step=yolo_step)
        except Exception as e:
            _status, ok = f"异常: {e}", False
        if state.cancel_requested:
            cancelled = True
            reason = _status.splitlines()[0] if _status else "已取消"
            lines.append(f"⏹ {name}: {reason}")
            print(f"    ↳ CANCELLED: {reason}", flush=True)
            break
        if ok:
            n_ok += 1
            total_goals += len(state.last_goals)
            lines.append(f"✓ {name}: 成功，{len(state.last_goals)} 个进球")
            print(f"    ↳ OK: {len(state.last_goals)} goals", flush=True)
        else:
            reason = _status.splitlines()[0] if _status else "失败"
            lines.append(f"✗ {name}: {reason}")
            print(f"    ↳ FAIL: {reason}", flush=True)
    # ============ DEBUG: BATCH 结束 ============
    _batch_elapsed = time.time() - _batch_t0
    _end_time = time.strftime('%H:%M:%S')
    print("")
    print("#" * 68)
    print(f"[BATCH  END ] {_end_time}  |  total {_batch_elapsed/60:.1f} min")
    print(f"  Result: {n_ok}/{n_total} success  |  {total_goals} goals total")
    if cancelled:
        print(f"  Status: CANCELLED")
    print("#" * 68 + "\n", flush=True)
    if cancelled:
        msg = f"已取消 | 已完成 {n_ok}/{n_total} 个视频 | 共 {total_goals} 个进球"
    else:
        msg = f"批量识别完成: {n_ok}/{n_total} 个视频 | 共 {total_goals} 个进球"
    return msg + "\n" + "\n".join(lines), n_ok > 0
