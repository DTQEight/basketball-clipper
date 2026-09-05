"""检测/剪辑/批量业务逻辑。

从原 demo_nicegui.py 抽离，通过 services.state 模块访问/修改运行时状态，
UI 层只需调用本模块的函数并传入参数。
"""
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# 项目根目录（basketball-clipper/）—— 必须在扁平模块导入之前注入 sys.path
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from . import state
from . import video_utils
from video_io import get_video_info, read_frame, VideoReader
from app import get_ball_model, get_device, get_ball_class_ids
from tracker import GoalDetector
from cutter.ffmpeg_cutter import cut_clips, build_encode_args

log = logging.getLogger("detection")

# 预览片段为进球时刻 ±N 秒（demo_nicegui 卡片时间戳显示与此保持同一来源）
PREVIEW_CLIP_HALF_SEC = 3.0


# ============ 单视频操作 ============

def load_video(video_path, task_token=0):
    """加载视频，返回预览帧和信息字符串。

    作为单视频模式的入口，切换视频时**必须同步清空上一个视频的检测 state**
    （进球列表/预览片段/保留索引），避免 UI 层依赖某分支才清空导致旧数据残留。
    task_token: 非零时锁由本函数持有并在 finally 释放（锁归任务本体）。
    """
    try:
        if not video_path or not video_path.strip():
            return None, "请输入视频文件路径"
        video_path = video_path.strip().strip('"').strip("'")
        if not os.path.exists(video_path):
            return None, f"❌ 文件不存在: {video_path}"
        try:
            info = get_video_info(video_path)
        except Exception as e:
            return None, f"读取失败: {e}"
        # 切换新视频：先清空上一个视频的检测结果（无论当前视频是否能最终成功 read_frame，
        # 只要路径合法 → state.video_state 会更新 → 旧进球列表就应清空）
        state.last_goal_clips.clear()
        state.last_goals.clear()
        state.kept_goal_indices.clear()
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
    finally:
        if task_token:
            state.release_task(task_token)


def _draw_calib_overlay(frame, hoop, clicks):
    """在 BGR 帧上叠加篮筐框 + 点击标记，返回新帧（BGR 上绘制，调用方再转 RGB）。"""
    out = frame.copy()
    if hoop:
        x1, y1, x2, y2 = hoop
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(out, "HOOP", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.line(out, (x1 - 30, y1), (x2 + 30, y1), (0, 255, 255), 1)
        cv2.line(out, (x1 - 30, y2), (x2 + 30, y2), (255, 0, 255), 1)
    for i, (x, y) in enumerate(clicks):
        cv2.circle(out, (x, y), 10, (255, 255, 0), -1)
        cv2.putText(out, str(i + 1), (x - 6, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return out


def preview_frame(frame_idx):
    """预览指定帧。"""
    if state.video_state["path"] is None:
        return None, "请先加载视频"
    frame = read_frame(state.video_state["path"], int(frame_idx),
                       total=state.video_state["total"], fps=state.video_state["fps"])
    if frame is None:
        return None, "读取帧失败"
    out = _draw_calib_overlay(frame, state.calib["hoop"], state.calib["clicks"])
    ts = int(frame_idx) / state.video_state["fps"]
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), f"帧 {frame_idx} ({ts:.1f}s)"


def click_calibrate(x, y):
    """点击标定篮筐。"""
    if state.video_state["path"] is None:
        return None, "请先加载视频"
    frame_idx = state.video_state["current_frame"]
    # 整个函数只读一次目标帧：基准帧与叠加显示复用同一份
    # （旧实现读两次 read_frame，各自 open+seek，大视频标定手感明显变慢）
    frame = read_frame(state.video_state["path"], int(frame_idx),
                       total=state.video_state["total"], fps=state.video_state["fps"])
    state.calib["clicks"].append((x, y))
    status = f"点击 ({x},{y})，已收集 {len(state.calib['clicks'])}/2 个点"
    if len(state.calib["clicks"]) >= 2:
        p1, p2 = state.calib["clicks"][:2]
        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
        state.calib["hoop"] = (x1, y1, x2, y2)
        if frame is not None:
            state.calib["baseline_frame"] = frame  # read_frame 返回全新数组，无需 copy
            state.calib["baseline_idx"] = int(frame_idx)
        status = f"篮筐已标定: ({x1},{y1}) - ({x2},{y2}) | 基准帧: 第 {int(frame_idx)} 帧"
        state.calib["clicks"] = []
    if frame is None:
        return None, status
    out = _draw_calib_overlay(frame, state.calib["hoop"], state.calib["clicks"])
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

    片段之间相互独立，用小线程池并行跑 ffmpeg（NVENC 限 2 路、软编 3 路），
    相比串行逐个生成约提速 2.5-3 倍；50 球场景从 ~2 分钟降到 ~40 秒。
    该阶段 GPU 推理已结束，不与检测抢资源。
    返回 clips 列表 [{"ts": float, "path": str, "idx": int}, ...]（按进球顺序排序）
    """
    import imageio_ffmpeg
    import subprocess as _sp
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not goals:
        return []
    if cancel_check and cancel_check():
        return []

    out_dir = Path(state.CACHE_ROOT) / "demo_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    # 编码参数只检测一次（NVENC 探测是子进程，循环内重复调用会显著拖慢）
    _enc = build_encode_args(ff, quality="preview")
    _is_nvenc = "h264_nvenc" in _enc
    # GeForce 消费卡驱动限制同时 2-5 路 NVENC 会话，保守用 2；软编受 CPU 核数约束用 3
    _workers = 2 if _is_nvenc else 3
    clip_half = int(fps * PREVIEW_CLIP_HALF_SEC)
    # 软编回退参数：NVENC 运行时失败（驱动/会话配额）时单段重切用
    _enc_x264 = None if not _is_nvenc else build_encode_args(ff, quality="preview", use_nvenc=False)

    def _cut_cmd(clip_path, seg_start_sec, seg_dur_sec, enc_args):
        return [ff, "-y", "-loglevel", "error",
                "-ss", f"{seg_start_sec:.3f}", "-i", video_path,
                "-t", f"{seg_dur_sec:.3f}",
                "-vf", "scale=-2:480"] + enc_args + \
               ["-movflags", "+faststart", clip_path]

    def _run_cut(clip_path, seg_start_sec, seg_dur_sec, enc_args):
        """跑一次 ffmpeg 切片。失败抛异常（含 stderr 尾部）。"""
        # NVENC 会话配额（消费卡限 2 路）：与集锦 cut_clips 共用信号量排队
        if "h264_nvenc" in enc_args:
            with state.nvenc_semaphore:
                _r = _sp.run(_cut_cmd(clip_path, seg_start_sec, seg_dur_sec, enc_args),
                             creationflags=state.SBOX, capture_output=True,
                             text=True, timeout=60)
        else:
            _r = _sp.run(_cut_cmd(clip_path, seg_start_sec, seg_dur_sec, enc_args),
                         creationflags=state.SBOX, capture_output=True,
                         text=True, timeout=60)
        if _r.returncode != 0:
            tail = (_r.stderr or "")[-1500:].strip()
            raise RuntimeError(f"ffmpeg exit {_r.returncode}: {tail}")

    def _cut_one(gi, gts):
        """切单个片段，成功返回 clip dict，失败/取消返回 None（异常不外抛）。

        NVENC 失败时自动回退软编重切一次，降低 clips < goals（预览失败的
        真实进球从集锦中二次丢失）的概率。
        """
        if cancel_check and cancel_check():
            return None
        gframe = int(gts * fps)
        seg_start = max(start, gframe - clip_half)
        seg_end = min(end, gframe + clip_half)
        if seg_end <= seg_start:
            return None
        clip_path = str(out_dir / f"goal_{gi}_{int(gts)}s_{stamp}.mp4")
        seg_start_sec = seg_start / fps
        seg_dur_sec = (seg_end - seg_start) / fps
        enc_args = _enc
        for _attempt in range(2):  # 第 2 次 = NVENC 失败后软编重试
            try:
                _run_cut(clip_path, seg_start_sec, seg_dur_sec, enc_args)
                if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                    return {"ts": gts, "path": clip_path, "idx": gi}
                log.warning(f"[WARN] 预览片段生成空文件 ({gts:.1f}s)，跳过")
                return None
            except Exception as e:
                if _attempt == 0 and _enc_x264 is not None:
                    log.warning(f"[WARN] 预览片段 NVENC 失败 ({gts:.1f}s)，回退软编重切: {e}")
                    enc_args = _enc_x264
                    continue
                log.warning(f"[WARN] 预览片段生成失败 ({gts:.1f}s): {e}")
                # 清理失败残留的半截文件（旧实现留着 0 字节文件占目录）
                try:
                    if os.path.exists(clip_path):
                        os.remove(clip_path)
                except OSError:
                    pass
                return None
        return None

    clips = []
    _done = 0
    _clips_t0 = time.time()
    # 进度回调统一在提交线程（本线程）内触发，保持单线程调用语义，与旧串行版一致
    with ThreadPoolExecutor(max_workers=_workers) as pool:
        futs = [pool.submit(_cut_one, gi, gts) for gi, gts in enumerate(goals)]
        for fut in as_completed(futs):
            clip = fut.result()
            if clip is not None:
                clips.append(clip)
            _done += 1
            if progress_callback:
                # 预计剩余时间：按片段完成速率线性外推
                _eta_sec = ((len(goals) - _done) / max(_done, 1)
                            * max(time.time() - _clips_t0, 0.001))
                progress_callback(80 + 18 * _done / len(goals),
                                  f'生成片段 {_done}/{len(goals)} | 预计剩余 {_eta_sec:.0f}s')
    # as_completed 完成顺序乱，按进球序恢复，保证卡片时间戳顺序稳定
    clips.sort(key=lambda c: c["idx"])
    return clips


# ============ 检测 ============

def run_detect(start_frame, end_frame, ball_conf, min_gap_sec,
               diff_threshold=15, min_circularity=0.35, min_in_hoop_frames=2,
               min_blob_area=30, search_margin=80, progress_callback=None,
               auto_threshold=True, yolo_step=2, skip_yolo_no_motion=False,
               task_token=0):
    """运行进球检测。返回 (结果文本, 是否成功)。

    task_token: UI 侧 try_acquire_task 返回的 token。
    传入时锁由本函数持有并在 finally 释放（锁归任务本体：UI 协程在页面
    刷新/断开时被取消，io_bound 线程无法取消继续跑，若由 UI release
    会出现"锁已释放、本线程还在写 state"的并发窗口）。
    """
    def _release_lock():
        if task_token:
            state.release_task(task_token)

    if state.video_state["path"] is None:
        _release_lock()
        return "❌ 请先加载视频", False
    if state.calib["hoop"] is None:
        _release_lock()
        return "❌ 请先点击画面标定篮筐", False
    if state.calib["baseline_frame"] is None:
        _release_lock()
        return "❌ 基准帧差法需要基准帧，请重新标定", False
    # 不做 CPU 降级：无 CUDA 时直接拒绝检测（CPU 推理慢约 10 倍，
    # 静默降级会让用户误以为服务正常而空等数小时）
    _device = get_device()
    if _device == "cpu":
        _release_lock()
        return "❌ 未检测到可用 CUDA，请检查显卡驱动/CUDA 环境后重启服务（不支持 CPU 推理）", False

    def _report(pct, msg):
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    hoop = state.calib["hoop"]
    fps = state.video_state["fps"]
    total = state.video_state["total"]
    start = max(0, int(start_frame))
    end = int(end_frame) if end_frame and int(end_frame) > 0 else total
    end = min(max(start + 1, end), total)
    if end <= start:
        _release_lock()
        return "❌ 结束帧必须大于起始帧", False

    # ===== 断点续识别辅助量（必须在恢复逻辑改写 start 之前确定）=====
    # _orig_start: 完整检测区间的原始起始帧。断点恢复会把 start 改写为断点帧，
    #   但预览片段边界（恢复点之前的进球也要有片段）和统计口径
    #   （video_duration_sec / processed_frames 等）都必须用完整区间，不能被恢复点截断
    # _resume_frame: 下一帧待处理帧号。feed 完一帧后更新为 fidx+1，
    #   三处 checkpoint 保存点统一存它：存"已 feed 的帧号"会导致恢复时该帧被
    #   二次 feed（blob_in_hoop_frames 同帧 +2 等状态机污染）
    # _prev_elapsed: 断点之前的累计检测耗时（秒），恢复后 proc_fps /
    #   speed_vs_realtime / elapsed_sec 用全程口径，与从头跑可比
    _orig_start = start
    _resume_frame = start
    _prev_elapsed = 0.0
    processed = 0

    # ===== 入口一次性快照本次检测的全部输入（P1） =====
    # 旧实现生命周期内 4 次读取 state.video_state["path"]（预热 reader / 正式 reader /
    # 预览片段 / 写历史），批量检测运行数十分钟，期间 UI 切换视频会改写全局 path，
    # 导致同一 run_detect 的不同阶段取到不同视频 → 历史记录与实际检测视频错位。
    video_path = state.video_state["path"]
    baseline_frame = state.calib["baseline_frame"]
    baseline_idx = state.calib["baseline_idx"]
    video_width = state.video_state.get("width")
    video_height = state.video_state.get("height")

    # ===== 断点续识别：构建参数指纹 + 检查已有 checkpoint =====
    _cp_params = {
        "hoop": hoop, "baseline_idx": baseline_idx, "fps": fps,
        "start_frame": start, "end_frame": end,
        "ball_conf": ball_conf, "min_gap_sec": min_gap_sec,
        "diff_threshold": int(diff_threshold),
        "min_circularity": float(min_circularity),
        "min_in_hoop_frames": int(min_in_hoop_frames),
        "min_blob_area": int(min_blob_area), "search_margin": int(search_margin),
        "auto_threshold": bool(auto_threshold),
        "yolo_step": int(yolo_step), "skip_yolo_no_motion": bool(skip_yolo_no_motion),
    }
    _checkpoint = state.load_checkpoint(video_path, _cp_params)
    _resuming = _checkpoint is not None
    # H2: 预热未完成的断点视为无效。自适应阈值开启但断点里没有算出的阈值
    # （典型：预热阶段点取消 → 主循环第一帧即 break，保存了 current_frame≈start、
    # _auto_threshold_value=None 的空断点）。若照常续跑：预热 pass 被跳过、
    # 恢复逻辑又拿不到自适应阈值 → 整个视频静默退化成固定阈值，与同参数
    # 从头跑结果不一致且 UI 上自适应开关仍显示开启。删除断点按从头跑（重跑预热）。
    if _resuming and bool(auto_threshold) \
            and _checkpoint.get("detector_state", {}).get("_auto_threshold_value") is None:
        state.delete_checkpoint(video_path, _cp_params)
        _checkpoint = None
        _resuming = False

    _report(5, '初始化检测器...')
    try:
        # 取消短路：预热阶段已被取消时不再加载模型/构建检测器，
        # 直接走取消返回（旧实现仍会跑完模型加载数秒~十几秒才退出）
        if state.cancel_event.is_set():
            state.last_goal_clips.clear()
            state.kept_goal_indices.clear()
            state.last_goals.clear()
            return "已取消（预热阶段）", False
        # ============================================================
        # 步骤A（预热前置 pass）：auto_threshold=True 时先跑前 30s 帧算阈值
        # 预热阶段不跑 YOLO，不判定进球，只收集 P95，算出最终自适应阈值。
        # 然后正式检测用这个阈值 + auto_threshold=False 从帧0完整检测，
        # 避免视频开头进球被预热期跳过。
        # ============================================================
        _warmup_info = None   # 保存预热阶段诊断（auto_threshold_value / median / samples）
        _effective_diff_threshold = int(diff_threshold)
        if bool(auto_threshold) and not _resuming:
            _report(6, '预热：收集前30s帧噪声水平...')
            _warmup_detector = GoalDetector(
                hoop, baseline_frame=baseline_frame,
                min_gap_sec=float(min_gap_sec),
                diff_threshold=int(diff_threshold),
                min_blob_area=int(min_blob_area),
                search_margin=int(search_margin),
                loose_mode=True,
                yolo_confirm=True, rolling_baseline_sec=60.0,
                min_circularity=float(min_circularity),
                min_in_hoop_frames=int(min_in_hoop_frames),
                auto_threshold=True,
                fps=fps)
            # ceil 修正：iter_frames 是半开区间 [start, end)，需要保证区间内存在
            # 满足 (fidx-start)/fps >= 30s 的帧，预热完成判定才能触发。
            # fps=30.0 时 int(900)+1=901 即可（900/30=30.0s 但取不到）；
            # fps=30.04 时 int(901.2)+1=902 → 902/30.04=30.03s ✓
            # 若用 int+1 会得到 901 → 29.99s < 30s，预热永远差 1 帧不触发（ABORT 回退）
            _warmup_end = min(end, start + int(math.ceil(30.0 * max(fps, 1.0))) + 1)
            _warmup_n = max(0, _warmup_end - start)
            _ws = time.time()
            _warmup_reader = VideoReader(video_path)
            try:
                for _wfidx, _wframe in _warmup_reader.iter_frames(start=start, end=_warmup_end, batch=1):
                    # 预热期只跑 diff 收集 P95，ball_pos=None 跳过 YOLO（省显存/耗时）
                    _warmup_detector.feed(None, _wfidx, fps, frame=_wframe)
                    # 预热进度反馈（旧实现停在 6% 无增量，长视频像卡死）
                    if (_wfidx - start) % 150 == 0:
                        _report(6 + 3 * (_wfidx - start) / max(_warmup_n, 1),
                                f'预热采样 {(_wfidx - start)}/{_warmup_n} 帧...')
                    if _warmup_detector._warmup_done:
                        break
                    # 预热循环也响应取消（旧实现不检查，点取消后还要无响应跑完约 900 帧）
                    if state.cancel_event.is_set():
                        break
            finally:
                _warmup_reader.close()
            if _warmup_detector._warmup_done and _warmup_detector._auto_threshold_value is not None:
                _effective_diff_threshold = int(_warmup_detector._auto_threshold_value)
                _we = time.time()
                log.info(f"[WARMUP DONE] 预热 {_we-_ws:.1f}s | "
                         f"阈值 = median(P95)={_warmup_detector._warmup_p95_median:.1f} + 8 = {_effective_diff_threshold} "
                         f"| 采样 {_warmup_detector._warmup_sample_count} 帧")
                _warmup_info = {
                    "auto_threshold_value": _warmup_detector._auto_threshold_value,
                    "warmup_p95_median": _warmup_detector._warmup_p95_median,
                    "warmup_sample_count": _warmup_detector._warmup_sample_count,
                }
            else:
                # 视频短于30s或预热失败，回退到用户传入的 diff_threshold
                _we = time.time()
                log.info(f"[WARMUP ABORT] 视频过短或未完成预热，回退固定阈值 {diff_threshold}")
                _warmup_info = {
                    "auto_threshold_value": None,
                    "warmup_p95_median": (
                        float(np.median(_warmup_detector._warmup_p95s))
                        if _warmup_detector._warmup_p95s else None
                    ),
                    "warmup_sample_count": len(_warmup_detector._warmup_p95s),
                }
            del _warmup_detector

        # ===== 断点续识别：从 checkpoint 恢复阈值/预热信息 =====
        if _resuming:
            _cp_det_state = _checkpoint.get("detector_state", {})
            _cp_auto_thr = _cp_det_state.get("_auto_threshold_value")
            if _cp_auto_thr is not None:
                _effective_diff_threshold = int(_cp_auto_thr)
            _warmup_info = {
                "auto_threshold_value": _cp_det_state.get("_auto_threshold_value"),
                "warmup_p95_median": _cp_det_state.get("_warmup_p95_median"),
                "warmup_sample_count": _cp_det_state.get("_warmup_sample_count", 0),
            }
            # 起始帧 = checkpoint 的下一帧帧号（从该帧继续，不重复 feed 已处理帧）；
            # 原始起始帧保留在 _orig_start：预览片段边界与统计口径仍用完整区间
            _cp_frame = int(_checkpoint.get("current_frame", start))
            if _cp_frame > start:
                start = _cp_frame
            _resume_frame = start
            # 断点前的累计检测耗时（checkpoint extra.elapsed）：
            # 恢复后的 proc_fps / speed_vs_realtime / elapsed_sec 用全程口径
            try:
                _prev_elapsed = float((_checkpoint.get("extra") or {}).get("elapsed") or 0.0)
            except (TypeError, ValueError):
                _prev_elapsed = 0.0
            log.info(f"[RESUME] 从断点恢复: 帧 {start}, 已检测进球 {len(_cp_det_state.get('goals', []))} 个")

        # 步骤B：正式检测器。
        # - auto_threshold=False：使用预热得到的 _effective_diff_threshold 作为固定阈值
        # - 从 start 帧完整检测，前30s不会再跳过进球
        _use_auto_for_detector = False   # 正式检测阶段永远关闭内部预热
        detector = GoalDetector(hoop, baseline_frame=baseline_frame,
                                min_gap_sec=float(min_gap_sec),
                                diff_threshold=_effective_diff_threshold,
                                min_blob_area=int(min_blob_area),
                                search_margin=int(search_margin),
                                loose_mode=True,
                                yolo_confirm=True, rolling_baseline_sec=60.0,
                                min_circularity=float(min_circularity),
                                min_in_hoop_frames=int(min_in_hoop_frames),
                                auto_threshold=_use_auto_for_detector,
                                fps=fps)
        # 把步骤A计算出的自适应阈值信息挂到正式 detector 上，保持旧代码读取逻辑兼容
        if _warmup_info is not None:
            detector._auto_threshold_value = _warmup_info["auto_threshold_value"]
            detector._warmup_p95_median = _warmup_info["warmup_p95_median"]
            detector._warmup_sample_count = _warmup_info["warmup_sample_count"]
            detector.auto_threshold = bool(auto_threshold)   # 保留用户原始意图用于诊断显示/历史写入

        # ===== 断点续识别：恢复检测器状态机（goals/帧号/斑块状态/基准帧等）=====
        if _resuming:
            detector.set_state(_cp_det_state)

        _report(10, '加载 YOLO 模型...')
        model, _weights_path = get_ball_model()
        # 按 model.names 反查球类别索引：classes=[0] 只对自定义单类权重成立，
        # 回退 COCO 权重（yolov8n.pt）时类 0 是 person，硬编码会误把球员当球确认
        _ball_classes = get_ball_class_ids(model, _weights_path)
        if not _ball_classes:
            # names 解析不出球类别 = 权重不可信：拒绝检测而不是把 person 当球确认
            # 同步清空旧结果：同一视频重跑失败时 UI 卡片不应残留上一次的
            # 进球/片段（与 YOLO 熔断、异常路径的清理策略对齐）
            state.last_goal_clips.clear()
            state.kept_goal_indices.clear()
            state.last_goals.clear()
            return "❌ 无法从模型类别表识别球类别（model.names 异常），请检查权重文件", False

        t0 = time.time()
        t0_str = time.strftime('%H:%M:%S', time.localtime(t0))
        # 统计口径：n_frames 为本次运行待处理帧数（ETA/瞬时速率）；
        # _total_frames 为完整区间帧数（进度/历史记录），断点续跑与从头跑可比
        n_frames = end - start
        _done_before = start - _orig_start      # 断点前已处理帧数（从头跑为 0）
        _total_frames = end - _orig_start       # 完整区间帧数（进度/历史统计口径）
        video_dur_min = _total_frames / max(fps, 1) / 60.0

        # ============ DEBUG: 开始信息 ============
        log.info("\n" + "=" * 68)
        log.info(f"[DETECT START] {t0_str}")
        log.info(f"  Video       : {video_path}")
        log.info(f"  Resolution  : {video_width}x{video_height}  @ {fps:.1f} fps")
        log.info(f"  Frames      : {_orig_start}-{end-1}  ({_total_frames} total, {video_dur_min:.1f} min)")
        if bool(auto_threshold):
            # 如实区分预热成功 / 预热失败回退，避免 ABORT 后仍显示"已预热"误导排查
            _warmup_ok = (_warmup_info is not None
                          and _warmup_info.get("auto_threshold_value") is not None)
            if _warmup_ok:
                log.info(f"  Auto-thresh : ON (已预热, 阈值={_effective_diff_threshold}, P95 中位={_warmup_info.get('warmup_p95_median')})")
            else:
                log.info(f"  Auto-thresh : ON (预热未完成, 回退固定阈值 {_effective_diff_threshold})")
        else:
            log.info(f"  Auto-thresh : OFF (固定阈值 {diff_threshold})")
        log.info(f"  YOLO step   : every {yolo_step} frames  ({100/yolo_step:.0f}% coverage)")
        if skip_yolo_no_motion:
            log.info(f"  条件跳过    : ON (篮筐无运动时跳过 YOLO)")
        log.info(f"  ball_conf   : {ball_conf}  min_gap : {min_gap_sec}s")
        log.info(f"  min_circ    : {min_circularity}  in_hoop_f : {min_in_hoop_frames}")
        log.info("=" * 68)
        _debug_last_print_time = t0
        _debug_last_processed = 0

        _report(15, f'开始检测 {n_frames} 帧...')
        reader = VideoReader(video_path)
        # 跳帧检测：每 N 帧跑一次 YOLO，跳过的帧复用上一帧 ball_pos（只跑 diff）
        # 篮球下落速度 ~8m/s，30fps 下每帧位移 <0.3m，连续 2 帧丢失不会漏检
        _yolo_step = max(1, int(yolo_step))
        # (检测帧号, ball_pos)：复用跳帧结果时保留原始检测帧号，
        # 保持 tracker 时间窗口（±N 帧内球在筐边）的语义准确
        _last_ball = None
        _stat_yolo_called = 0
        _stat_yolo_skipped = 0
        _stat_yolo_failed = 0   # YOLO 推理异常次数（如驱动升级后 CUDA 上下文失效）
        try:
            for fidx, frame in reader.iter_frames(start=start, end=end, batch=1):
                if state.cancel_event.is_set():
                    break
                ball_pos = None
                ball_frame = None    # ball_pos 对应的真实检测帧号
                need_yolo = False
                _yolo_skipped = False  # 条件跳过标记（用于进度显示）
                # 预热已在前置 pass 完成，正式阶段从帧 0 直接跑检测（不再跳过前30s YOLO/进球）
                # 正式检测：每 _yolo_step 帧一次 YOLO，其余帧复用上一帧结果
                need_yolo = ((processed % _yolo_step) == 0)
                # ROI 每帧至多算一次：条件跳过判定与 feed 共用
                # （旧实现 has_motion_near_hoop 与 feed 内部各算一次，~1-2ms/帧纯浪费）
                _pending_roi = detector.compute_roi(frame) if (need_yolo and skip_yolo_no_motion) else None
                if need_yolo and skip_yolo_no_motion:
                    # 条件跳过：篮筐区域无运动像素时跳过 YOLO（省 ~60ms）
                    if not detector.has_motion_near_hoop(frame, frame_roi=_pending_roi):
                        need_yolo = False
                        ball_pos = None
                        _last_ball = None
                        _yolo_skipped = True
                        _stat_yolo_skipped += 1
                if need_yolo:
                    _stat_yolo_called += 1
                    try:
                        # classes 过滤：ultralytics 在 NMS 后直接过滤，只保留球类
                        # 省去 CPU 侧全量 .cpu().numpy() 拷贝 + 遍历查找。
                        # _ball_classes 按 model.names 反查（自定义权重=[0]，
                        # COCO 回退权重=[32] sports ball），不再硬编码 [0]
                        # device 用循环外缓存的 _device：运行中设备不会变化，
                        # 每次推理重新 import torch + is_available 属纯冗余（全程 ~2 万次）
                        res = model.predict(frame, conf=float(ball_conf), imgsz=960,
                                            classes=_ball_classes,
                                            device=_device, verbose=False)[0]
                        if res.boxes is not None and len(res.boxes) > 0:
                            xyxy = res.boxes.xyxy.cpu().numpy()
                            confs = res.boxes.conf.cpu().numpy()
                            best = int(np.argmax(confs))
                            x1, y1, x2, y2 = xyxy[best]
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            ball_pos = (float(cx), float(cy), float(x1), float(y1),
                                        float(x2), float(y2), float(confs[best]))
                        _last_ball = (fidx, ball_pos)
                    except Exception as e:
                        # 不能静默吞掉：CUDA 失效（如升级显卡驱动后未重启服务）会
                        # 持续抛异常 → 球位置永远为空 → 所有进球被"YOLO确认"拒绝 → 0 进球空跑全程
                        _stat_yolo_failed += 1
                        if _stat_yolo_failed == 1:
                            import traceback
                            log.error(f"[YOLO ERROR] 首次推理失败: {e}\n{traceback.format_exc()}")
                        # 失败率过高时立即中止（旧实现跑完全程才检查，
                        # CUDA 失效时仍会空跑几十分钟才报错）
                        if (_stat_yolo_called >= 10
                                and _stat_yolo_failed >= _stat_yolo_called * 0.5):
                            log.error(f"[YOLO FAIL] 推理失败率过高，提前中止: "
                                      f"{_stat_yolo_failed}/{_stat_yolo_called}")
                            break
                else:
                    # 跳帧：复用最近一次 YOLO 结果（只在篮筐附近有效），
                    # 帧号用原始检测帧号（位置是 1-2 帧前的，不能用当前帧号）
                    if _last_ball is not None:
                        ball_frame, ball_pos = _last_ball
                detector.feed(ball_pos, fidx, fps, frame=frame,
                              ball_frame=ball_frame, frame_roi=_pending_roi)
                processed += 1
                # 该帧已完整 feed：恢复点 = 下一帧（存 fidx 会让恢复时重复 feed 此帧）
                _resume_frame = fidx + 1
                # ===== 断点续识别：每 300 帧保存一次 checkpoint =====
                # 存 _resume_frame（下一帧帧号，而非 processed 计数）
                if processed % 300 == 0:
                    state.save_checkpoint(
                        video_path, _cp_params, _resume_frame,
                        detector.get_state(),
                        extra={
                            "processed": processed,
                            "yolo_called": _stat_yolo_called,
                            "yolo_skipped": _stat_yolo_skipped,
                            "yolo_failed": _stat_yolo_failed,
                            "elapsed": _prev_elapsed + (time.time() - t0),
                        })
                # 每 10 帧更新一次进度
                if processed % 10 == 0:
                    pct = 15 + 60 * (_done_before + processed) / _total_frames
                    if need_yolo:
                        phase = '检测'
                    elif _yolo_skipped:
                        phase = '跳过'
                    else:
                        phase = '跳帧'
                    # 预计剩余时间：按当前处理速率线性外推（跳过/跳帧比例稳定时准确）
                    _eta_min = ((n_frames - processed) / max(processed, 1)
                                * max(time.time() - t0, 0.001)) / 60.0
                    _report(pct, f'{phase}帧 {_done_before + processed}/{_total_frames} '
                                 f'({(_done_before + processed) * 100 // _total_frames}%) | '
                                 f'预计剩余 {_eta_min:.1f} 分钟')
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
                    _phase_label = '检测'
                    log.info(f"[{time.strftime('%H:%M:%S')}] {_phase_label} {processed:>6d}/{n_frames:<6d} "
                             f"({_pct*100:5.1f}%) | 瞬时 {_instant_fps:>5.0f} f/s  平均 {_overall_fps:>5.1f} f/s | "
                             f"ETA {_eta/60:>4.1f} min | 进球累计 {len(detector.goals):>3d}")
                    _debug_last_print_time = _now
                    _debug_last_processed = processed
        finally:
            reader.close()

        # YOLO 失败率报警：推理环境损坏（如驱动升级后旧进程 CUDA 上下文失效）时，
        # 结果不可信（所有候选进球都会被 YOLO 确认拒绝），直接报错让用户重启服务
        # （循环内已提前中止，这里统一返回错误；两个条件保持一致）
        if _stat_yolo_called >= 10 and _stat_yolo_failed >= _stat_yolo_called * 0.5:
            _fail_pct = _stat_yolo_failed / _stat_yolo_called * 100
            log.error(f"[YOLO FAIL] 推理失败率过高: {_stat_yolo_failed}/{_stat_yolo_called} "
                      f"({_fail_pct:.0f}%)，本次结果不可信，已中止")
            state.last_goal_clips.clear()
            state.kept_goal_indices.clear()
            state.last_goals.clear()
            return (f"❌ YOLO 推理失败率过高 ({_stat_yolo_failed}/{_stat_yolo_called})，"
                    f"疑似 CUDA 环境失效（如升级显卡驱动后未重启服务），请重启服务后重试", False)

        _t1 = time.time()
        _detect_elapsed = _t1 - t0
        _report(80, '生成预览片段...')

        # 生成预览片段
        goals = sorted(detector.goals)
        state.last_goal_clips.clear()
        state.kept_goal_indices.clear()

        _stamp = int(time.time())
        state.last_goal_clips.extend(
            # 片段边界用完整区间 _orig_start（而非恢复后的 start）：
            # 恢复点之前检测到的进球若距恢复点超过 3 秒，片段会被
            # _cut_one 的边界钳制直接丢弃（进球卡片无预览、集锦缺球）
            _generate_preview_clips(video_path, goals, _orig_start, end,
                                    fps, total, _stamp, progress_callback=_report,
                                    cancel_check=state.cancel_event.is_set)
        )

        if state.cancel_event.is_set():
            # 用户取消：删除本次已生成的片段文件，清空内存列表，不写入历史
            _t2 = time.time()
            for _c in list(state.last_goal_clips):
                try:
                    os.remove(_c["path"])
                except OSError:
                    pass
            state.last_goal_clips.clear()
            state.kept_goal_indices.clear()
            state.last_goals.clear()
            # ===== 断点续识别：取消时保存最终 checkpoint（用户可从此处继续）=====
            # 仅在本次确有帧已 feed（processed > 0）时保存：预热阶段取消
            # （主循环第一帧即 break）或视频无可解码帧时没有可恢复的进度，
            # 保存"空断点"只会让下次启动弹出无意义的续跑提示、且续跑后
            # 拿不到自适应阈值（H2）
            if processed > 0:
                state.save_checkpoint(
                    video_path, _cp_params, _resume_frame,
                    detector.get_state(),
                    extra={
                        "processed": processed,
                        "yolo_called": _stat_yolo_called,
                        "yolo_skipped": _stat_yolo_skipped,
                        "yolo_failed": _stat_yolo_failed,
                        "elapsed": _prev_elapsed + _detect_elapsed,
                    })
                log.info(f"[{time.strftime('%H:%M:%S')}] [CANCELLED] 已处理 {processed} 帧，取消后退出（可断点续跑）")
                return f"已取消 | 已处理 {processed} 帧（可断点续跑）", False
            log.info(f"[{time.strftime('%H:%M:%S')}] [CANCELLED] 已处理 {processed} 帧，取消后退出")
            return f"已取消 | 已处理 {processed} 帧", False

        # 检测成功后写入片段缓存并持久化（key 统一走 clip_cache_key，排序+round）
        if state.last_goal_clips:
            state.put_clip_cache(state.clip_cache_key(video_path, goals),
                                 state.last_goal_clips)

        # clips < goals 时显式提示：预览失败的进球是真实的，
        # 用户应知晓集锦将缺少这些球（旧实现静默缩水）
        _missing_previews = len(goals) - len(state.last_goal_clips)

        _report(100, '完成！')
        # ===== 断点续识别：检测成功，清理该视频全部 checkpoint =====
        # 不只删当前参数指纹的一个：残留其他参数的旧断点会让"发现未完成检测"
        # 弹窗每次必现，且用户点"继续"后因指纹不匹配实际从头跑（UI 承诺与
        # 行为不一致）。检测已成功，任何参数组合的旧断点都已无意义
        state.delete_checkpoint(video_path)
        state.kept_goal_indices = set(range(len(state.last_goal_clips)))
        state.last_goals.clear()
        state.last_goals.extend(detector.goals)

        d = detector.diag
        total_yolo = d['yolo_confirmed'] + d['yolo_rejected']
        confirm_rate = d['yolo_confirmed'] / max(total_yolo, 1) * 100

        _t2 = time.time()
        # 全程口径：断点续跑时叠加断点前累计耗时，与从头跑同字段可比（L2）
        _total_elapsed = _prev_elapsed + (_t2 - t0)
        _preview_elapsed = _t2 - _t1
        _detect_total = _prev_elapsed + _detect_elapsed
        _proc_fps = (_done_before + processed) / max(_detect_total, 0.001)
        _end_str = time.strftime('%H:%M:%S', time.localtime(_t2))

        # ============ DEBUG: 结束统计 ============
        log.info("")
        log.info("=" * 68)
        log.info(f"[DETECT  END] {_end_str}")
        log.info(f"  Timing      : detect {_detect_total/60:.1f} min + preview {_preview_elapsed/60:.1f} min = {_total_elapsed/60:.1f} min total")
        log.info(f"  Speed       : {_proc_fps:.1f} frames/sec  (video {video_dur_min:.1f} min / detect {_detect_total/60:.1f} min = {video_dur_min/max(_detect_total/60,0.001):.2f}x vs realtime)")
        log.info(f"  Goals       : {len(goals)} detected")
        if goals:
            _timestamps_str = ", ".join(f"{g:.1f}s" for g in goals[:8]) + ("..." if len(goals) > 8 else "")
            log.info(f"                {_timestamps_str}")
        if skip_yolo_no_motion:
            _yolo_total_scheduled = _stat_yolo_called + _stat_yolo_skipped
            _skip_rate = _stat_yolo_skipped / max(_yolo_total_scheduled, 1) * 100
            log.info(f"  YOLO 调用   : 实际推理 {_stat_yolo_called} 次  |  条件跳过 {_stat_yolo_skipped} 次  ({_skip_rate:.0f}% 跳过率)")
        if _stat_yolo_failed > 0:
            log.info(f"  YOLO 异常   : {_stat_yolo_failed}/{_stat_yolo_called} 次推理失败（详见 [YOLO ERROR] 日志）")
        log.info(f"  YOLO 确认   : {d['yolo_confirmed']}/{total_yolo} ({confirm_rate:.0f}%)  |  "
                 f"上方: {d['cross_above']}  下方: {d['cross_below']}  筐内: {d['in_hoop']}  冷却拒: {d['reject_cooldown']}")
        if detector.auto_threshold:
            if detector._auto_threshold_value is not None:
                if detector._warmup_p95_median is not None:
                    log.info(f"  AutoThresh  : value={detector._auto_threshold_value}  |  median(P95)={detector._warmup_p95_median:.1f}  +8  clamp[8,50]  |  samples={detector._warmup_sample_count}")
                else:
                    log.info(f"  AutoThresh  : value={detector._auto_threshold_value}  |  (P95 数据未保存)")
            else:
                log.info(f"  AutoThresh  : 预热未完成，实际使用固定阈值 {detector.diff_threshold}")
        else:
            log.info(f"  DiffThresh  : 固定 {diff_threshold}")
        log.info("=" * 68 + "\n")

        status = (f"检测完成 | 处理 {processed} 帧 | 耗时 {_total_elapsed:.0f}s\n"
                  f"进球: {len(detector.goals)} 个 | "
                  f"YOLO确认: {d['yolo_confirmed']}/{total_yolo} ({confirm_rate:.0f}%)")
        if _missing_previews > 0:
            status += f"\n⚠ {_missing_previews} 个进球的预览片段生成失败（集锦将缺少这些球）"
        if detector.auto_threshold and detector._auto_threshold_value is not None:
            status += f"\n自适应阈值: {detector._auto_threshold_value} (P95+8)"
        # ===== 准备历史记录写入数据 =====
        # auto_threshold 传入值 = 用户UI开关的意图；即使固定阈值模式（UI关闭）但预热算出了值，也一并保存便于诊断
        _diff_for_history = (
            'auto' if auto_threshold else
            (detector._auto_threshold_value if (detector._auto_threshold_value is not None) else diff_threshold)
        )
        # auto_threshold_value 的最终值：
        #   只要用户 UI 开了自动阈值且预热成功（_warmup_info is not None），
        #   就直接用 _effective_diff_threshold（预热算出的最终 int 阈值），
        #   不再依赖 detector 内部属性（主循环 feed 可能覆盖 _auto_threshold_value）。
        if auto_threshold and _warmup_info is not None:
            _auto_thr_for_history = int(_effective_diff_threshold)
        else:
            _auto_thr_for_history = detector._auto_threshold_value
        _warmup_p95_for_history = (
            detector._warmup_p95_median
            if (detector._warmup_p95_median is not None)
            else (_warmup_info.get("warmup_p95_median") if _warmup_info else None)
        )
        _warmup_count_for_history = (
            detector._warmup_sample_count
            if (detector._warmup_sample_count and detector._warmup_sample_count > 0)
            else (_warmup_info.get("warmup_sample_count") if _warmup_info else 0)
        )
        _batch_idx = None
        _batch_total = None
        if state.batch_files and state.batch_current_video in state.batch_files:
            _batch_total = len(state.batch_files)
            try:
                _batch_idx = state.batch_files.index(state.batch_current_video) + 1
            except ValueError:
                _batch_idx = None
        _start_abs = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))
        _end_abs = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_t2))
        # YOLO 跳过统计（即使未开条件跳过也保存，便于对比）
        _yolo_total_sched = _stat_yolo_called + _stat_yolo_skipped
        _yolo_skip_rate_pct = _stat_yolo_skipped / max(_yolo_total_sched, 1) * 100
        # YOLO 确认/否决统计
        _total_yolo_path = d['yolo_confirmed'] + d['yolo_rejected']
        _confirm_rate_pct = d['yolo_confirmed'] / max(_total_yolo_path, 1) * 100
        # 速度 vs 实时倍数（全程口径：视频时长与检测耗时都含断点前部分）
        _speed_vs_realtime = video_dur_min / max(_detect_total / 60.0, 0.001)

        _saved_record = state.add_history(video_path, hoop, detector.goals,
                          baseline_idx=baseline_idx,
                          ball_conf=ball_conf,
                          min_gap_sec=min_gap_sec,
                          diff_threshold=_diff_for_history,
                          auto_threshold=auto_threshold,
                          yolo_step=yolo_step,
                          skip_yolo_no_motion=skip_yolo_no_motion,
                          min_circularity=min_circularity,
                          min_in_hoop_frames=min_in_hoop_frames,
                          min_blob_area=min_blob_area,
                          search_margin=search_margin,
                          elapsed_sec=_total_elapsed,
                          batch_idx=_batch_idx,
                          batch_total=_batch_total,
                          detect_start_time=_start_abs,
                          detect_end_time=_end_abs,
                          # ===== 视频元信息 =====
                          video_fps=fps,
                          video_width=video_width,
                          video_height=video_height,
                          video_total_frames=total,
                          video_duration_sec=video_dur_min * 60.0,
                          # ===== 处理速度指标（全程口径，断点续跑与从头跑可比）=====
                          processed_frames=_done_before + processed,
                          proc_fps=_proc_fps,
                          speed_vs_realtime=_speed_vs_realtime,
                          # ===== YOLO 跳过统计 =====
                          yolo_called=_stat_yolo_called,
                          yolo_cond_skipped=_stat_yolo_skipped,
                          yolo_skip_rate_pct=_yolo_skip_rate_pct,
                          # ===== YOLO 确认/否决 =====
                          yolo_confirmed=d['yolo_confirmed'],
                          yolo_rejected=d['yolo_rejected'],
                          yolo_confirm_rate_pct=_confirm_rate_pct,
                          # ===== 进球路径细分 =====
                          cross_above=d['cross_above'],
                          cross_below=d['cross_below'],
                          in_hoop=d['in_hoop'],
                          reject_cooldown=d['reject_cooldown'],
                          # ===== 自适应阈值详情 =====
                          auto_threshold_value=_auto_thr_for_history,
                          warmup_p95_median=_warmup_p95_for_history,
                          warmup_sample_count=_warmup_count_for_history)
        if _saved_record is None:
            # 磁盘/权限问题导致未落盘：显式告知（旧实现静默，用户下次启动才发现历史缺失）
            status += "\n⚠ 历史记录写入失败（磁盘/权限问题），本次结果未持久化"
        return status, True
    except Exception as e:
        import traceback
        # 异常路径清理旧结果：同一视频重跑失败时，UI 卡片不应残留上一次成功的
        # 进球/片段，避免用户把旧结果误当本次结果（与 YOLO 熔断路径对齐）
        state.last_goal_clips.clear()
        state.kept_goal_indices.clear()
        state.last_goals.clear()
        # ===== 断点续识别：异常时尽量保存 checkpoint（若已进入主循环）=====
        # processed 提前初始化为 0：异常发生在主循环前（模型加载/构建检测器）
        # 时 processed == 0，不保存无进度的空断点；_resume_frame 同理已初始化，
        # 此处不会再因 fidx 未定义抛 NameError（旧实现的 'fidx' in dir() 防了一半）
        try:
            if 'detector' in dir() and processed > 0:
                state.save_checkpoint(
                    video_path, _cp_params, _resume_frame,
                    detector.get_state(),
                    extra={"processed": processed,
                           "elapsed": _prev_elapsed + (time.time() - t0)})
        except Exception:
            pass
        return f"❌ 检测失败: {e}\n{traceback.format_exc()}", False
    finally:
        # 锁归任务本体：无论成功/失败/取消，线程真正结束时才释放。
        # UI 侧不再 release（页面刷新取消 UI 协程时，本线程仍持有锁直到跑完）
        _release_lock()


def clip_action(action, idx, video_path=None, person=None):
    """处理卡片按钮操作。

    video_path=None: 单视频模式，操作全局 last_goal_clips（原逻辑不变）。
    video_path 非空: 流水线快照模式，操作 batch_results[video_path] 内的数据，
                     不触碰全局 state（后台批量检测运行中也可安全调用）。
    person: set_person 动作的人物名（"" = 清除分类）。
    """
    # ===== 选择数据源：快照模式 or 全局模式 =====
    if video_path is not None:
        snap = state.batch_results.get(video_path)
        if not snap:
            return None, ""
        clips = snap["clips"]
        kept = snap["kept"]
    else:
        clips = state.last_goal_clips
        kept = state.kept_goal_indices

    if idx < 0 or idx >= len(clips):
        return None, ""
    if action == "preview":
        return clips[idx]["path"], f"▶ 正在预览第 {idx+1} 个片段"
    elif action == "export":
        return clips[idx]["path"], f"已导出: {clips[idx]['path']}"
    elif action == "set_person":
        # 人物分类：clip["person"] = 名字（""=清除）；持久化到历史 labels.persons
        clip = clips[idx]
        ts = clip["ts"]
        name = (str(person) or "").strip()
        clip["person"] = name or None
        try:
            state.update_history_labels(
                video_path if video_path else state.video_state["path"],
                kept_ts_list=None, deleted_ts_list=None,
                person_map={ts: name},
            )
        except Exception:
            pass
        # 跨视频复用：登记进全局人物名单（最近使用在前；空名不登记）
        if name:
            try:
                state.add_person(name)
            except Exception:
                pass
        # 已分类片段的 √ 标记语义不受影响，kept 集合无需变更
        shown = name if name else "已清除分类"
        msg = f"第 {idx+1} 个片段（{ts:.1f}s）→ {shown}"
        return None, msg
    elif action in ("mark_keep", "mark_reject"):
        # √/× 仅做标记，不删除片段（列表保持完整，导出集锦只取 √）
        target = "keep" if action == "mark_keep" else "reject"
        clip = clips[idx]
        # toggle：再次点击同标记 = 取消
        clip["mark"] = None if clip.get("mark") == target else target
        clip["mark_source"] = "manual" if clip["mark"] else None
        ts = clip["ts"]
        # kept 集合 = √ 标记的索引（导出集锦/历史标签都以 mark 为准）
        kept.clear()
        kept.update(i for i, c in enumerate(clips) if c.get("mark") == "keep")
        # 标签飞轮：√ → kept_ts_list（正样本），× → deleted_ts_list（负样本）
        kept_ts = [c["ts"] for c in clips if c.get("mark") == "keep"]
        reject_ts = [c["ts"] for c in clips if c.get("mark") == "reject"]
        try:
            state.update_history_labels(
                video_path if video_path else state.video_state["path"],
                kept_ts_list=kept_ts,
                deleted_ts_list=reject_ts,
            )
        except Exception:
            pass
        sym = {"keep": "√ 确认", "reject": "× 误报"}.get(clip["mark"], "已取消标记")
        n_keep = len(kept_ts)
        n_reject = len(reject_ts)
        msg = (f"第 {idx+1} 个片段（{ts:.1f}s）{sym} | "
               f"√ {n_keep} · × {n_reject} · 待标 {len(clips) - n_keep - n_reject}")
        return None, msg
    return None, ""


# 人物筛选哨兵值：仅导出"已分类"（任意人物）的片段，不含未分类片段。
# 用双下划线包裹降低与真实人名撞车的概率；UI 下拉以 detection.PERSON_FILTER_CLASSIFIED 为键。
PERSON_FILTER_CLASSIFIED = "__classified__"


def _export_goals(clips, all_goals, person=None):
    """导出集锦的进球时间戳筛选：

    person 非空 → 只导出该人物的片段；person 为 PERSON_FILTER_CLASSIFIED
                  哨兵 → 只导出任意已分类的片段（不含未分类）。筛选后再叠加
                  √/× 标记规则；
    有 √ 标记 → 只导出 √ 的；
    无 √ 但有 × → 导出未标记的（× 排除在外）；
    完全没标记 → 导出全部（老行为兼容）。
    """
    pool = clips
    if person == PERSON_FILTER_CLASSIFIED:
        pool = [c for c in clips if c.get("person")]
        if not pool:
            return []
    elif person:
        pool = [c for c in clips if c.get("person") == person]
        if not pool:
            return []
    keep_ts = [float(c["ts"]) for c in pool if c.get("mark") == "keep"]
    if keep_ts:
        return sorted(keep_ts)
    if any(c.get("mark") == "reject" for c in pool):
        return sorted(float(c["ts"]) for c in pool if c.get("mark") != "reject")
    if person:
        # 人物/已分类筛选下无任何标记 → 只导筛选池内的
        return sorted(float(c["ts"]) for c in pool)
    return [float(t) for t in all_goals]


def _filter_desc(person_filter):
    """导出筛选的状态栏描述：「小明」/「已分类」/ ''（全部）。"""
    if not person_filter:
        return ""
    if person_filter == PERSON_FILTER_CLASSIFIED:
        return "「已分类」"
    return f"「{person_filter}」"


def _person_highlights_path(src_video, person):
    """按人物导出的输出路径：{视频名}-{人物}-highlights.mp4。

    person 为 PERSON_FILTER_CLASSIFIED 哨兵 → {视频名}-已分类-highlights.mp4
    （与默认全部导出、各人物导出互不覆盖）。
    人名做文件名安全化：Windows 非法字符剔除 + 截断 30 字符（防超长人名
    撑爆路径限制）+ 全非法时回落 "person"。
    不带人物名的话，多个人物的个人集锦会互相覆盖同一个 {视频名}-highlights.mp4。
    """
    _vname = os.path.splitext(os.path.basename(src_video))[0]
    if person == PERSON_FILTER_CLASSIFIED:
        _safe = "已分类"
    else:
        _safe = "".join(c for c in person if c not in '\\/:*?"<>|').strip()[:30]
        _safe = _safe or "person"
    return os.path.join(state.CACHE_ROOT, "demo_output",
                        f"{_vname}-{_safe}-highlights.mp4")


def generate_highlights(pre_roll, post_roll, min_gap, progress_callback=None,
                        video_path=None, task_token=0, person_filter=None):
    """生成集锦视频。

    video_path=None: 单视频模式，用全局 video_state + last_goals（原逻辑不变）。
    video_path 非空: 流水线快照模式，用 batch_results[video_path] 的 goals 生成，
                     批量检测运行中也可调用（NVENC 硬编与 CUDA 推理是 GPU 独立单元，可并行）。
                     - hl_busy 小锁由本函数持有并在 finally 释放（生命周期归任务本体：
                       页面刷新取消 UI 协程时旧线程仍在跑，UI 侧提前重置会让新页面
                       对同一视频再次启动集锦，两线程写同一输出文件）
                     - 取消走独立的 hl_cancel_event：批量检测的「取消」不应连带杀死集锦
    task_token: 非零时全局任务锁由本函数持有并在 finally 释放（锁归任务本体）。
    person_filter: 非空 → 只导出该人物分类的片段（按人物导出集锦）。
    """
    _pipeline = video_path is not None
    try:
        if _pipeline:
            if state.hl_busy["on"]:
                return None, "❌ 该视频的集锦正在生成中，请稍候"
            state.hl_busy["on"] = True
        if video_path is not None:
            snap = state.batch_results.get(video_path)
            if not snap:
                return None, "❌ 该视频没有检测结果"
            clips = snap["clips"]
            goals = _export_goals(clips, list(snap["goals"]), person_filter)
            src = video_path
        else:
            if state.video_state["path"] is None:
                return None, "❌ 请先加载视频并检测进球"
            goals = _export_goals(state.last_goal_clips, list(state.last_goals), person_filter)
            src = state.video_state["path"]
        if not goals:
            return None, f"❌ {_filter_desc(person_filter)}没有可导出的进球片段"
        # 按人物导出：文件名带人物名，多个人物集锦互不覆盖
        _out_path = _person_highlights_path(src, person_filter) if person_filter else None
        # 流水线模式取消走独立事件：批量「取消」不连带杀死集锦
        _cancel = state.hl_cancel_event.is_set if _pipeline else state.cancel_event.is_set
        out_path = cut_clips(src, goals,
                             pre_roll=int(pre_roll), post_roll=int(post_roll),
                             min_gap=int(min_gap),
                             progress_callback=progress_callback,
                             cancel_check=_cancel,
                             output_path=_out_path)
        if out_path and os.path.exists(out_path):
            return out_path, (f"集锦已生成{_filter_desc(person_filter)}"
                              f"（{len(goals)} 个进球片段）\n输出: {out_path}")
        if _cancel():
            return None, "已取消集锦生成"
        return None, "❌ 集锦生成失败"
    except Exception as e:
        import traceback
        return None, f"❌ 剪辑失败: {e}\n{traceback.format_exc()}"
    finally:
        if _pipeline:
            state.hl_busy["on"] = False
        if task_token:
            state.release_task(task_token)


def _clips_from_record(r):
    """从历史记录轻量重建 (ts, mark, person) 片段视图（不生成预览）。

    整场导出只需要进球时间戳 + 标记 + 人物分类，不需要预览片段文件；
    直接从记录的 goals + labels 重建，避免逐视频跑 ffmpeg。
    """
    labels = r.get("labels") or {}
    kept = {float(t) for t in (labels.get("kept") or [])}
    deleted = {float(t) for t in (labels.get("deleted") or [])}
    persons = {}
    for k, v in (labels.get("persons") or {}).items():
        try:
            persons[float(k)] = v
        except (TypeError, ValueError):
            continue
    clips = []
    for t in r.get("goals", []):
        # 标签键（kept/deleted/persons）统一存 round(ts,3)，goals 是全精度：
        # 与 _on_load_history_impl 相同的舍入匹配，否则 31.2789 对不上 31.279
        ts = round(float(t), 3)
        clips.append({"ts": ts,
                      "mark": "keep" if ts in kept else
                              ("reject" if ts in deleted else None),
                      "person": persons.get(ts)})
    return clips


def _game_highlights_path(videos, person):
    """整场（多视频合并）集锦输出路径：{文件夹名}-[人物]-highlights.mp4。

    文件夹名取全部视频的公共父目录名（如 2026.09.04）；
    无公共目录（跨盘符等）回落首视频名 + "-game"。
    人物名安全化规则与单视频导出一致。
    """
    folder = ""
    try:
        dirs = {os.path.dirname(os.path.abspath(v)) for v in videos}
        if len(dirs) == 1:
            folder = os.path.basename(dirs.pop())
    except Exception:
        folder = ""
    if not folder:
        folder = os.path.splitext(os.path.basename(videos[0]))[0] + "-game"
    _safe_folder = "".join(c for c in folder if c not in '\\/:*?"<>|').strip()[:40]
    _safe_folder = _safe_folder or "game"
    if person == PERSON_FILTER_CLASSIFIED:
        _safe = "已分类"
    elif person:
        _safe = "".join(c for c in person if c not in '\\/:*?"<>|').strip()[:30]
        _safe = _safe or "person"
    else:
        _safe = ""
    suffix = f"-{_safe}-highlights.mp4" if _safe else "-highlights.mp4"
    return os.path.join(state.CACHE_ROOT, "demo_output", f"{_safe_folder}{suffix}")


def generate_highlights_fullgame(person_filter, pre_roll, post_roll, min_gap,
                                 progress_callback=None, task_token=0):
    """整场（四节合并）集锦导出：跨批量文件夹内全部视频。

    视频集合 = state.batch_files（当前扫描的文件夹，按文件名排序 = 第1节→第4节）。
    每个视频的片段来源：流水线快照（batch_results，批量运行中优先，最新）
    或历史记录（labels 含 √/× 与人物分类）。同一人物筛选规则与单视频一致
    （有√只导√ → 无√排除× → 无标记导筛选池全部）。
    输出：{文件夹名}-{人物}-highlights.mp4（与各单节导出互不覆盖）。
    """
    _pipeline = state.current_task() == 'batch'
    try:
        if _pipeline:
            if state.hl_busy["on"]:
                return None, "❌ 集锦正在生成中，请稍候"
            state.hl_busy["on"] = True
        # 视频集合：批量模式 = 扫描过的文件夹；
        # 单视频模式（batch_files 空）= 回退到当前视频所在文件夹自动发现同场视频
        # （单视频加载路径会清空 batch_files，用户没扫过文件夹也应能整场导出）
        if state.batch_files:
            videos = sorted(v for v in state.batch_files if v)
        elif state.video_state.get("path"):
            _folder = os.path.dirname(state.video_state["path"])
            videos = video_utils.scan_video_files(_folder)
        else:
            videos = []
        if not videos:
            return None, "❌ 整场导出需要批量视频列表或已加载视频，请先扫描文件夹/加载视频"
        sources = []
        for v in videos:
            if v in state.batch_results:
                snap = state.batch_results[v]
                clips, goals = snap["clips"], list(snap["goals"])
            else:
                r = state.get_record(v)
                if not r:
                    continue  # 该视频无快照也无历史 → 跳过（如某节漏检测）
                clips = _clips_from_record(r)
                goals = [float(t) for t in r.get("goals", [])]
            if not clips:
                continue
            ts_list = _export_goals(clips, goals, person_filter)
            if ts_list:
                sources.append((v, ts_list))
        if not sources:
            return None, f"❌ {_filter_desc(person_filter)}整场没有可导出的进球片段"
        _out_path = _game_highlights_path([v for v, _ in sources], person_filter)
        # 流水线模式取消走独立事件：批量「取消」不连带杀死集锦
        _cancel = state.hl_cancel_event.is_set if _pipeline else state.cancel_event.is_set
        _n_goals = sum(len(t) for _, t in sources)
        out_path = cut_clips(sources, None,
                             pre_roll=int(pre_roll), post_roll=int(post_roll),
                             min_gap=int(min_gap),
                             progress_callback=progress_callback,
                             cancel_check=_cancel,
                             output_path=_out_path)
        if out_path and os.path.exists(out_path):
            return out_path, (f"整场集锦已生成{_filter_desc(person_filter)}"
                              f"（{len(sources)} 个视频，{_n_goals} 个进球片段）\n"
                              f"输出: {out_path}")
        if _cancel():
            return None, "已取消集锦生成"
        return None, "❌ 集锦生成失败"
    except Exception as e:
        import traceback
        return None, f"❌ 剪辑失败: {e}\n{traceback.format_exc()}"
    finally:
        if _pipeline:
            state.hl_busy["on"] = False
        if task_token:
            state.release_task(task_token)


def on_load_history(idx_choice, progress_callback=None, task_token=0):
    """从历史记录加载。

    task_token: 非零时锁由本函数持有并在 finally 释放（锁归任务本体：
    未命中片段缓存时本函数会跑 ffmpeg 生成预览（可达数十秒），
    UI 协程在页面刷新/断开时被取消后线程仍会继续写 state，
    锁必须等线程真正结束才释放）。
    """
    try:
        return _on_load_history_impl(idx_choice, progress_callback)
    finally:
        if task_token:
            state.release_task(task_token)


def _on_load_history_impl(idx_choice, progress_callback):
    """on_load_history 的实际实现（锁由外层 wrapper 管理）。"""
    def _report(pct, msg):
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    _report(5, '读取历史记录...')
    if idx_choice is None:
        return None, "请先选择一条历史记录", ""
    try:
        records = state.load_history()
    except OSError as e:
        return None, f"历史记录暂时无法读取（{e}），请稍后重试", ""
    if idx_choice < 0 or idx_choice >= len(records):
        return None, "历史记录不存在", ""
    r = records[idx_choice]
    video_path = r.get("video", "")
    if not os.path.exists(video_path):
        return None, f"视频文件不存在: {video_path}", ""
    # 加载历史 = 回到单视频模式：清空批量状态，
    # 否则残留的 batch_files/batch_current_video 会让后续单视频检测的历史
    # 被误标 batch_idx/batch_total（UI 侧同步隐藏批量面板）
    state.batch_files = []
    state.batch_calibs = {}
    state.batch_current_video = None

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
            state.calib["baseline_frame"] = base_frame  # read_frame 返回全新数组，无需 copy
            state.calib["baseline_idx"] = saved_baseline_idx
        else:
            state.calib["baseline_frame"] = None
            state.calib["baseline_idx"] = -1
    all_goals = [float(t) for t in r.get("goals", [])]
    state.last_goals.clear()
    state.last_goals.extend(all_goals)

    fps = info["fps"]
    total = info["total"]
    _stamp = int(time.time())
    state.last_goal_clips.clear()
    state.kept_goal_indices.clear()

    _report(30, f'生成 {len(all_goals)} 个预览片段...')
    cache_key = state.clip_cache_key(video_path, all_goals)
    cached = state.clip_cache.get(cache_key)
    cache_hit = bool(cached and all(os.path.exists(c["path"]) for c in cached))
    if cache_hit:
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
            state.put_clip_cache(cache_key, state.last_goal_clips)
    log.info(f"[LOAD] {os.path.basename(video_path)} | "
             f"cache={'HIT' if cache_hit else 'MISS'} | "
             f"clips={len(state.last_goal_clips)}/{len(all_goals)}")

    # 先清空所有 clip 的 mark/mark_source，避免缓存共享引用携带上一次加载的
    # 残留标记（cache 命中时 last_goal_clips 与 clip_cache 共享同一批 dict，
    # 若 get_labels 因瞬态 IO 错误返回空，残留 mark 不会被覆盖，造成误显/漏显）。
    for c in state.last_goal_clips:
        c.pop("mark", None)
        c.pop("mark_source", None)
        c.pop("person", None)

    # 若历史里已有人工标签（kept=√ / deleted=× / persons=人物分类），恢复而非清空
    labels = state.get_labels(video_path)
    deleted_set = set(labels["deleted"]) if labels.get("deleted") else set()
    kept_set = set(labels["kept"]) if labels.get("kept") else set()
    persons_map = labels.get("persons") or {}
    log.info(f"[LOAD] labels: kept={len(kept_set)} deleted={len(deleted_set)} "
             f"persons={len(persons_map)} label_time={labels.get('label_time')}")
    if deleted_set or kept_set:
        kept_indices = []
        n_match_keep = 0
        n_match_reject = 0
        for idx, c in enumerate(state.last_goal_clips):
            ts = round(float(c["ts"]), 3)
            if ts in deleted_set:
                c["mark"] = "reject"
                c["mark_source"] = "manual"
                n_match_reject += 1
            elif ts in kept_set:
                c["mark"] = "keep"
                c["mark_source"] = "manual"
                kept_indices.append(idx)
                n_match_keep += 1
        state.kept_goal_indices = set(kept_indices)
        log.info(f"[LOAD] matched: keep={n_match_keep} reject={n_match_reject} "
                 f"(unmatched={len(state.last_goal_clips) - n_match_keep - n_match_reject})")
    else:
        state.kept_goal_indices = set(range(len(state.last_goal_clips)))
    # 人物分类回填（键为 round(ts,3) 精确匹配，与 mark 回填同一口径）
    if persons_map:
        n_person = 0
        for c in state.last_goal_clips:
            ts = round(float(c["ts"]), 3)
            if ts in persons_map:
                c["person"] = persons_map[ts]
                n_person += 1
        if n_person:
            log.info(f"[LOAD] persons matched: {n_person}")

    frame = read_frame(video_path, 0, total=total, fps=fps)
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    info_str = (f"{info['total']} 帧 | {info['fps']:.1f} fps | "
                f"{info['width']}x{info['height']} | {info['codec']}")
    n_show = len(state.last_goal_clips)
    n_keep = len(state.kept_goal_indices)
    n_reject = sum(1 for c in state.last_goal_clips if c.get("mark") == "reject")
    if labels.get("label_time") and (n_keep or n_reject):
        status = (f"已加载历史记录\n视频: {r.get('video_name', '')}\n"
                  f"进球: {n_show} 个（恢复上次标记 √ {n_keep} · × {n_reject}）\n"
                  f"已生成 {n_show} 个预览片段")
    else:
        status = (f"已加载历史记录\n视频: {r.get('video_name', '')}\n"
                  f"进球: {len(all_goals)} 个\n"
                  f"已生成 {len(state.last_goal_clips)} 个预览片段")
    return preview, info_str, status


# ============ 文件夹批量模式 ============

def on_batch_load_video(selected, progress_callback=None, task_token=0):
    """批量模式：加载选中的视频，应用该视频已保存的标定。

    若该视频已有历史检测记录（批量识别完成后再点击下拉框），
    自动加载检测结果和预览片段，无需再去历史记录里找。
    task_token: 非零时锁由本函数持有并在 finally 释放（锁归任务本体：
    未命中片段缓存时本函数会跑 ffmpeg 生成预览（可达数十秒），
    UI 协程被取消后线程仍会继续写 state，锁必须等线程结束才释放）。
    """
    try:
        return _on_batch_load_video_impl(selected, progress_callback)
    finally:
        if task_token:
            state.release_task(task_token)


def _on_batch_load_video_impl(selected, progress_callback):
    """on_batch_load_video 的实际实现（锁由外层 wrapper 管理）。"""
    def _report(pct, msg):
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    if not selected or not state.batch_files:
        return None, "", "请先扫描文件夹并选择视频"
    video_path = selected
    state.batch_current_video = video_path
    # —— 切换批量视频：先统一清空上一个视频的进球 state（不论后续是否命中历史记录，都先清再填）——
    # 这样避免「提前 return 分支漏清空」或者「异步生成预览期间 UI 残留旧卡片」。
    # 和单文件模式的 load_video 保持一致的先清后填策略。
    state.last_goal_clips.clear()
    state.last_goals.clear()
    state.kept_goal_indices.clear()
    if video_path in state.batch_calibs:
        cal = state.batch_calibs[video_path]
        state.calib["hoop"] = cal["hoop"]
        # 基准帧按保存的帧号现读（~百 ms，已在 io_bound 线程）：
        # 旧实现 batch_calibs 常驻整帧 BGR（1080p ~6MB/个），
        # 50 个视频批量 ≈300MB 常驻至程序结束，而帧只在检测瞬间用一次。
        # total=0：read_frame 内部 `if total > 0` 才 clamp，传 None 会抛 TypeError；
        # fps=0：read_frame 内部自动读流 fps。超界帧号由 decode 循环自然返回 None 兜底
        base_frame = read_frame(video_path, cal["baseline_idx"],
                                total=0, fps=0)
        state.calib["baseline_frame"] = base_frame
        state.calib["baseline_idx"] = cal["baseline_idx"] if base_frame is not None else -1
        state.calib["clicks"] = []
    else:
        # 跨会话复用标定：重启后 batch_calibs 清空，但该视频的历史记录里
        # 存有 hoop + baseline_idx（上次批量识别时写入）→ 恢复并登记进
        # batch_calibs（状态行显示「已标定」，且可直接跑批量识别无需重标）
        rec = None
        try:
            rec = state.get_record(video_path)
        except Exception:
            rec = None
        hoop = rec.get("hoop") if rec else None
        if hoop and len(hoop) == 4:
            hoop = tuple(int(v) for v in hoop)
            base_idx = int(rec.get("baseline_idx", 0))
            state.batch_calibs[video_path] = {"hoop": hoop, "baseline_idx": base_idx}
            state.calib["hoop"] = hoop
            base_frame = read_frame(video_path, base_idx, total=0, fps=0)
            state.calib["baseline_frame"] = base_frame
            state.calib["baseline_idx"] = base_idx if base_frame is not None else -1
            state.calib["clicks"] = []
            log.info(f"[CALIB] {os.path.basename(video_path)} 从历史记录恢复篮筐标定")
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

    # —— 流水线快照优先：批量识别后点视频，优先从快照水合 ——
    # 快照里保留了人工删减后的 goals/clips，比走历史记录重新生成预览更快，且不丢删减结果
    snap = state.batch_results.get(video_path)
    if snap and snap.get("clips"):
        state.last_goals.extend(snap["goals"])
        state.last_goal_clips.extend([dict(c) for c in snap["clips"]])
        state.kept_goal_indices = set(range(len(state.last_goal_clips)))
        # 人物分类回填：快照 clip 缺 person 时从历史 labels 补（快照在批量
        # 检测刚完成时不含分类，历史里可能已有此前会话保存的分类）
        try:
            _persons = state.get_labels(video_path).get("persons") or {}
            if _persons:
                for c in state.last_goal_clips:
                    if not c.get("person"):
                        p = _persons.get(round(float(c["ts"]), 3))
                        if p:
                            c["person"] = p
        except Exception:
            pass
        status = (f"已加载: {os.path.basename(video_path)}\n"
                  f"{'已标定' if video_path in state.batch_calibs else '未标定'}\n"
                  f"进球: {len(snap['goals'])} 个（含人工删减）\n"
                  f"复用 {len(state.last_goal_clips)} 个预览片段")
        return preview, info_str, status

    # —— 查找历史记录：批量识别完成后，下拉框选视频时自动加载检测结果 ——
    records = state.load_history()
    matched = None
    for r in records:
        if r.get("video") == video_path:
            matched = r
            break

    if matched:
        _report(10, '找到检测记录，加载进球数据...')
        all_goals = [float(t) for t in matched.get("goals", [])]
        # 注：last_goals / last_goal_clips / kept_goal_indices 已在函数入口统一清空，
        # 这里直接 extend，不需要再 .clear() 一次
        state.last_goals.extend(all_goals)

        fps = info["fps"]
        total = info["total"]
        _stamp = int(time.time())

        _report(30, f'生成 {len(all_goals)} 个预览片段...')
        cache_key = state.clip_cache_key(video_path, all_goals)
        cached = state.clip_cache.get(cache_key)
        if cached and all(os.path.exists(c["path"]) for c in cached):
            state.last_goal_clips.extend(list(cached))
            _report(30, f'命中缓存，复用 {len(state.last_goal_clips)} 个片段')
        elif all_goals:
            state.last_goal_clips.extend(
                _generate_preview_clips(video_path, all_goals, 0, total,
                                        fps, total, _stamp, progress_callback=_report)
            )
            if state.last_goal_clips:
                state.put_clip_cache(cache_key, state.last_goal_clips)

        # 加载后全部进球默认保留（筛选结果不持久化）
        state.kept_goal_indices = set(range(len(state.last_goal_clips)))

        # 人物分类回填（√/× 标记在该路径不恢复，人物分类与筛选正交，恢复无害）
        try:
            _persons = state.get_labels(video_path).get("persons") or {}
            if _persons:
                for c in state.last_goal_clips:
                    p = _persons.get(round(float(c["ts"]), 3))
                    if p:
                        c["person"] = p
        except Exception:
            pass

        status = (f"已加载: {os.path.basename(video_path)}\n"
                  f"{'已标定' if video_path in state.batch_calibs else '未标定'}\n"
                  f"进球: {len(all_goals)} 个\n"
                  f"已加载 {len(state.last_goal_clips)} 个预览片段")
    else:
        # 未检测过：函数入口已统一清空 state，这里只写状态文本，无需再清
        status = (f"已加载: {os.path.basename(video_path)}\n"
                  f"{'已标定' if video_path in state.batch_calibs else '未标定，请点击画面 2 个点标定'}")
    return preview, info_str, status


def backfill_batch_calibs_from_history():
    """从历史检测记录回填批量标定（跨会话复用篮筐标定）。

    扫描文件夹时 batch_calibs 清空；若该文件夹此前跑过批量识别，历史记录里
    已存每个视频的 hoop + baseline_idx，直接回填免去重新逐个标定。
    本次会话已保存的标定优先（不被历史覆盖）；无记录/无 hoop 的视频跳过。
    返回回填的视频数量。
    """
    if not state.batch_files:
        return 0
    try:
        records = state.load_history()
    except Exception as e:
        log.warning(f"[CALIB] 历史标定回填失败: {e}")
        return 0
    by_path = {}
    for r in records:
        v = r.get("video")
        if v and v not in by_path:
            by_path[v] = r
    n = 0
    for vp in state.batch_files:
        if vp in state.batch_calibs:
            continue
        r = by_path.get(vp)
        if not r:
            continue
        hoop = r.get("hoop")
        if not hoop or len(hoop) != 4:
            continue
        state.batch_calibs[vp] = {
            "hoop": tuple(int(v) for v in hoop),
            "baseline_idx": int(r.get("baseline_idx", 0)),
        }
        n += 1
    if n:
        log.info(f"[CALIB] 从历史记录回填 {n}/{len(state.batch_files)} 个视频的篮筐标定")
    return n


def on_batch_save_calib():
    """保存当前标定到当前批量视频。

    只存 hoop + baseline_idx（不存整帧 BGR）：50 个视频的整帧常驻 ~300MB，
    而基准帧只在检测启动瞬间用到，检测/加载时按帧号现读。
    """
    if state.batch_current_video is None:
        return "请先从列表选择视频"
    if state.calib["hoop"] is None or state.calib["baseline_frame"] is None:
        return "请先标定篮筐"
    state.batch_calibs[state.batch_current_video] = {
        "hoop": state.calib["hoop"],
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
                     auto_threshold=True, yolo_step=2, skip_yolo_no_motion=False,
                     per_video_callback=None, task_token=0):
    """批量识别：遍历文件夹内全部视频逐个检测，每个视频独立写入历史。

    返回 (状态文本, 是否成功)。状态文本逐条列出每个视频的结果，
    未标定/读取失败/检测失败都会单独说明，不再静默跳过。

    per_video_callback(video_path, goal_count): 每个视频检测成功后回调（UI 打完成标记）。
    结果同时存入 state.batch_results 快照，供流水线模式前台人工确认。
    task_token: 非零时锁由本函数持有并在 finally 释放（锁归任务本体）。
    """
    try:
        return _run_batch_detect_impl(start_frame, end_frame, ball_conf, min_gap_sec,
                                      diff_threshold, min_circularity, min_in_hoop_frames,
                                      min_blob_area, search_margin, progress_callback,
                                      auto_threshold, yolo_step, skip_yolo_no_motion,
                                      per_video_callback)
    finally:
        if task_token:
            state.release_task(task_token)


def _run_batch_detect_impl(start_frame, end_frame, ball_conf, min_gap_sec,
                           diff_threshold, min_circularity, min_in_hoop_frames,
                           min_blob_area, search_margin, progress_callback,
                           auto_threshold, yolo_step, skip_yolo_no_motion,
                           per_video_callback):
    """run_batch_detect 的实际实现（锁由外层 wrapper 管理）。"""
    if not state.batch_files:
        return "请先加载文件夹", False
    # 流水线快照：重跑批量时覆盖旧结果
    state.batch_results.clear()
    # 当前视频若已标定但未点「保存标定」，批量前自动保存，避免漏处理
    cur = state.batch_current_video
    if (cur and cur in state.batch_files and cur not in state.batch_calibs
            and state.calib["hoop"] is not None and state.calib["baseline_frame"] is not None):
        state.batch_calibs[cur] = {
            "hoop": state.calib["hoop"],
            "baseline_idx": state.calib["baseline_idx"],
        }
    lines = []
    n_ok = 0
    total_goals = 0
    n_total = len(state.batch_files)
    cancelled = False
    _batch_t0 = time.time()
    # ============ DEBUG: BATCH 开始 ============
    log.info("\n" + "#" * 68)
    log.info(f"[BATCH START] {time.strftime('%H:%M:%S')}  |  {n_total} videos")
    log.info(f"  Auto-thresh : {bool(auto_threshold)}")
    log.info(f"  YOLO step   : every {yolo_step} frames")
    if skip_yolo_no_motion:
        log.info(f"  条件跳过    : ON (篮筐无运动时跳过 YOLO)")
    log.info(f"  ball_conf   : {ball_conf}  min_gap : {min_gap_sec}s")
    log.info("#" * 68)
    for i, video_path in enumerate(state.batch_files):
        if state.cancel_event.is_set():
            cancelled = True
            break
        name = os.path.basename(video_path)
        log.info(f"\n>>> [{i+1}/{n_total}] {name} <<<")
        state.batch_current_video = video_path  # 同步当前视频，供 run_detect 内写历史时查 batch_idx
        if video_path not in state.batch_calibs:
            lines.append(f"✗ {name}: 未标定，跳过")
            log.info(f"    ↳ SKIP (未标定)")
            continue
        cal = state.batch_calibs[video_path]
        try:
            info = get_video_info(video_path)
        except Exception as e:
            lines.append(f"✗ {name}: 读取失败 ({e})")
            log.info(f"    ↳ READ FAIL: {e}")
            continue
        state.video_state.update(path=video_path, total=info["total"], fps=info["fps"],
                                  codec=info["codec"], current_frame=0,
                                  width=info["width"], height=info["height"])
        # 基准帧按标定保存的帧号现读（batch_calibs 不再常驻整帧，~6MB/视频）
        base_frame = read_frame(video_path, cal["baseline_idx"], total=info["total"], fps=info["fps"])
        if base_frame is None:
            lines.append(f"✗ {name}: 基准帧读取失败（帧 {cal['baseline_idx']}），跳过")
            log.info(f"    ↳ BASELINE READ FAIL @ frame {cal['baseline_idx']}")
            continue
        state.calib["hoop"] = cal["hoop"]
        state.calib["baseline_frame"] = base_frame
        state.calib["baseline_idx"] = cal["baseline_idx"]
        state.calib["clicks"] = []

        def _cb(pct, msg, vname=name, idx=i):
            if progress_callback:
                try:
                    # 归一化：每个视频占 1/n_total 份，pct 为当前视频的 0-100
                    overall = (idx + max(0, min(100, pct)) / 100.0) / n_total * 100.0
                    # 整批预计剩余时间：按已耗时与整体进度线性外推
                    # （前几个视频偏慢时估算偏保守，随进度收敛）
                    _elapsed = time.time() - _batch_t0
                    if overall > 1.0:
                        _eta_min = _elapsed * (100.0 - overall) / overall / 60.0
                        msg = f'{msg} | 整批预计剩余 {_eta_min:.0f} 分钟'
                    progress_callback(overall, f'[{idx+1}/{n_total}] {vname} · {msg}')
                except Exception:
                    pass

        try:
            _status, ok = run_detect(start_frame, end_frame, ball_conf, min_gap_sec,
                                     diff_threshold, min_circularity, min_in_hoop_frames,
                                     min_blob_area, search_margin, progress_callback=_cb,
                                     auto_threshold=auto_threshold, yolo_step=yolo_step,
                                     skip_yolo_no_motion=skip_yolo_no_motion)
        except Exception as e:
            _status, ok = f"异常: {e}", False
        if state.cancel_event.is_set():
            cancelled = True
            reason = _status.splitlines()[0] if _status else "已取消"
            lines.append(f"⏹ {name}: {reason}")
            log.info(f"    ↳ CANCELLED: {reason}")
            break
        if ok:
            n_ok += 1
            total_goals += len(state.last_goals)
            lines.append(f"✓ {name}: 成功，{len(state.last_goals)} 个进球")
            log.info(f"    ↳ OK: {len(state.last_goals)} goals")
            # ===== 流水线快照：深拷贝当前视频结果，前台可立即查看/确认 =====
            # 检测线程只写这个 key，之后永不触碰；前台删卡片只改快照，互不干扰
            state.batch_results[video_path] = {
                "goals": list(state.last_goals),
                "clips": [dict(c) for c in state.last_goal_clips],
                "kept": set(state.kept_goal_indices),
                "finished_at": time.strftime("%H:%M:%S"),
            }
            if per_video_callback:
                try:
                    per_video_callback(video_path, len(state.last_goals))
                except Exception:
                    pass
        else:
            reason = _status.splitlines()[0] if _status else "失败"
            lines.append(f"✗ {name}: {reason}")
            log.info(f"    ↳ FAIL: {reason}")
    # ============ DEBUG: BATCH 结束 ============
    # 批量结束（含取消路径）重置当前视频标记：防止之后切单视频检测时
    # 残留的 batch_current_video 让历史记录误带 batch_idx/batch_total
    state.batch_current_video = None
    _batch_elapsed = time.time() - _batch_t0
    _end_time = time.strftime('%H:%M:%S')
    log.info("")
    log.info("#" * 68)
    log.info(f"[BATCH  END ] {_end_time}  |  total {_batch_elapsed/60:.1f} min")
    log.info(f"  Result: {n_ok}/{n_total} success  |  {total_goals} goals total")
    if cancelled:
        log.info(f"  Status: CANCELLED")
    log.info("#" * 68 + "\n")
    if cancelled:
        msg = f"已取消 | 已完成 {n_ok}/{n_total} 个视频 | 共 {total_goals} 个进球"
    else:
        msg = f"批量识别完成: {n_ok}/{n_total} 个视频 | 共 {total_goals} 个进球"
    return msg + "\n" + "\n".join(lines), n_ok > 0
