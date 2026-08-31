# -*- coding: utf-8 -*-
"""L4 阶段 A 特征提取：对 dataset_v1.json 每个事件做 ±1.5s 密集 YOLO 复检，
计算轨迹/几何/网区能量特征，输出 training/features.jsonl（每行一个事件）。

用法（项目根执行）：
  env\\python.exe training\\extract_features.py --limit 5     # 小样测试
  env\\python.exe training\\extract_features.py               # 全量（支持断点续跑）

特征明细见 _compute_features() docstring。GPU 必须（项目硬性要求），CPU 直接拒绝。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import get_ball_model, get_ball_class_ids, get_device  # noqa: E402
from video_io import VideoReader  # noqa: E402

TRAINING_DIR = PROJECT_ROOT / "training"
DATASET_FILE = TRAINING_DIR / "dataset_v1.json"
FEATURES_FILE = TRAINING_DIR / "features.jsonl"

# 与生产一致的参数
BALL_CONF = 0.3
IMGSZ = 960
WIN_PRE = 1.5   # 事件前窗口（秒）
WIN_POST = 1.5  # 事件后窗口（秒）


def _resolve_video(path_str: str) -> str | None:
    """原路径不存在时回退 D:\\Downloads\\test\\{basename}。"""
    if Path(path_str).exists():
        return path_str
    alt = Path(r"D:\Downloads\test") / Path(path_str).name
    if alt.exists():
        return str(alt)
    return None


def _roi_mean_diff(frames_gray_small: list, i0: int, i1: int, roi) -> float | None:
    """帧区间 [i0,i1) 在 ROI（缩小坐标系）内的相邻帧平均差能量。"""
    if i1 - i0 < 2:
        return None
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        return None  # 空切片会产生 NaN，静默毒化 LightGBM 训练
    vals = []
    for i in range(i0 + 1, i1):
        d = np.abs(frames_gray_small[i].astype(np.int16) -
                   frames_gray_small[i - 1].astype(np.int16))
        vals.append(float(d[y1:y2, x1:x2].mean()))
    return float(np.mean(vals)) if vals else None


def _compute_features(detections, net_pre, net_post, hoop, video_w, video_h, fps):
    """detections: [(t_rel, cx, cy, x1, y1, x2, y2, conf)] 按时间序。
    hoop: [hx1, hy1, hx2, hy2]（像素，原始分辨率坐标系）。
    返回特征 dict（全部为 float/int，可直接喂 LightGBM）。
    """
    hx1, hy1, hx2, hy2 = [float(v) for v in hoop]
    hw = max(hx2 - hx1, 1.0)   # 筐宽
    hh = max(hy2 - hy1, 1.0)   # 筐高
    cx_h = (hx1 + hx2) / 2     # 筐心
    cy_h = (hy1 + hy2) / 2
    n_frames = max(len(detections), 1)
    # 预计窗口帧数（用于 detect_rate 分母，由调用方传实际解码帧数更准）
    f = {}
    f["hoop_w"] = hw
    f["hoop_h"] = hh
    f["hoop_aspect"] = hh / hw
    f["video_w"] = float(video_w)
    f["video_h"] = float(video_h)
    f["video_mpix"] = video_w * video_h / 1e6

    if not detections:
        # 无任何检测：全空特征（模型要能处理）
        for k in ("detect_rate", "in_x_rate", "in_band_rate", "cross_progress",
                  "above_then_below", "cross_clean", "vy_down_rate", "vy_reversals",
                  "vy_tail_down_rate", "post_drop_depth", "ball_h_rel", "ball_h_cv",
                  "conf_mean", "conf_min", "conf_last",
                  "dist_final", "x_off_final",
                  "below_x_std", "below_in_x_rate", "time_below_rate",
                  "x_final_in", "vx_drop_ratio", "vx_tail_mag",
                  "bounce_up_after_band", "min_dist_center"):
            f[k] = 0.0
        f["no_detection"] = 1.0
        f["net_pre"] = net_pre if net_pre is not None else 0.0
        f["net_post"] = net_post if net_post is not None else 0.0
        f["net_ratio"] = 0.0
        return f

    f["no_detection"] = 0.0
    xs = np.array([d[1] for d in detections])
    ys = np.array([d[2] for d in detections])
    confs = np.array([d[7] for d in detections])
    hs = np.array([(d[6] - d[4]) for d in detections])  # bbox 高度

    f["detect_rate"] = len(detections) / n_frames
    # 球心 x 在筐口水平范围（±0.25 筐宽余量）
    in_x = (xs >= hx1 - 0.25 * hw) & (xs <= hx2 + 0.25 * hw)
    f["in_x_rate"] = float(in_x.mean())
    # 球心 y 在筐口竖直带内
    in_band = (ys >= hy1) & (ys <= hy2)
    f["in_band_rate"] = float(in_band.mean())

    # 穿越：上方(hy1-0.2hh 之上) → 带内 → 下方(hy2+0.2hh 之下)，按时序状态机
    prog = 0  # 0=未见 1=above 2=band 3=below
    first_above = first_below = None
    for i, d in enumerate(detections):
        y = d[2]
        if prog == 0 and y < hy1 - 0.2 * hh:
            prog = 1
            first_above = i
        elif prog == 1 and hy1 - 0.2 * hh <= y <= hy2 + 0.2 * hh:
            prog = 2
        elif prog == 2 and y > hy2 + 0.2 * hh:
            prog = 3
            first_below = i
            break
    f["cross_progress"] = float(prog)
    f["above_then_below"] = 1.0 if (prog == 3) else 0.0
    # 时序正确且期间大多在 x 范围内的穿越（更强条件）
    if prog == 3 and first_above is not None and first_below is not None:
        seg = detections[first_above:first_below + 1]
        seg_in_x = np.mean([(d[1] >= hx1 - 0.25 * hw) and (d[1] <= hx2 + 0.25 * hw)
                            for d in seg])
        f["cross_clean"] = float(seg_in_x)
    else:
        f["cross_clean"] = 0.0

    # vy（图像坐标 y 向下增大 → vy>0 即下落）
    ts = [d[0] for d in detections]
    if len(detections) >= 2:
        dys = np.diff(ys)
        f["vy_down_rate"] = float((dys > 0).mean())
        f["vy_reversals"] = float(int(np.sum(np.diff(np.sign(dys)) != 0)))
        # 末段（最后 1/3 检测）是否持续下坠
        tail = dys[len(dys) * 2 // 3:]
        f["vy_tail_down_rate"] = float((tail > 0).mean()) if len(tail) else 0.0
    else:
        f["vy_down_rate"] = 0.0
        f["vy_reversals"] = 0.0
        f["vy_tail_down_rate"] = 0.0
    # 末位置低于筐下沿深度（归一化筐高）
    f["post_drop_depth"] = float(max(ys[-1] - hy2, 0.0) / hh)

    # ===== 底角45度机位特征（方向无关设计，左右底角通用）=====
    # 网区 = 筐下沿 0.2~2.5 筐高（限制在网内/筐下，排除落地弹起阶段）
    zone_lo = hy2 + 0.2 * hh
    zone_hi = hy2 + 2.5 * hh
    net_zone_mask = (ys > zone_lo) & (ys <= zone_hi)

    # 网区横向稳定性：入网被兜住 → x 几乎不动且落在筐正下方；弹框而出 → 横向逃逸
    if net_zone_mask.any():
        bx = xs[net_zone_mask]
        f["below_x_std"] = float(np.std(bx) / hw)
        f["below_in_x_rate"] = float(np.mean((bx >= hx1) & (bx <= hx2)))
    else:
        f["below_x_std"] = 0.0
        f["below_in_x_rate"] = 0.0
    f["time_below_rate"] = float(net_zone_mask.mean())
    f["x_final_in"] = 1.0 if hx1 <= xs[-1] <= hx2 else 0.0

    # 横向速度衰减：穿网被网吸能 → vx 骤减；打铁弹出 → 保持横向速度
    if len(detections) >= 4:
        vxs = np.diff(xs) / np.maximum(np.diff(ts), 1e-6)
        half = max(1, len(vxs) // 2)
        pre_vx = float(np.abs(vxs[:half]).mean())
        post_vx = float(np.abs(vxs[half:]).mean()) if len(vxs[half:]) else 0.0
        f["vx_drop_ratio"] = pre_vx / max(post_vx, 1.0)
        f["vx_tail_mag"] = post_vx / hw
    else:
        f["vx_drop_ratio"] = 0.0
        f["vx_tail_mag"] = 0.0

    # 筐区向上反弹（打铁弹出特征）：球在网区内从最深处回升的高度；
    # 球一旦穿出网区下界（去地板）即停止追踪，地板反弹不算打铁
    prog2 = 0
    max_y = None
    bounce = 0.0
    left_zone = False
    for d in detections:
        yy = d[2]
        if prog2 == 0:
            if hy1 - 0.2 * hh <= yy <= hy2 + 0.2 * hh:
                prog2 = 1
                max_y = yy
        elif not left_zone and max_y is not None:
            if yy > zone_hi:
                left_zone = True
            elif yy > max_y:
                max_y = yy
            else:
                bounce = max(bounce, (max_y - yy) / hh)
    f["bounce_up_after_band"] = float(bounce)

    # 轨迹最贴近筐心的距离（是否穿过筐口圆柱中心）
    f["min_dist_center"] = float(np.min(np.hypot(xs - cx_h, ys - cy_h)) / hw)

    # 球尺寸（相对筐宽）
    f["ball_h_rel"] = float(np.median(hs) / hw)
    f["ball_h_cv"] = float(np.std(hs) / max(np.mean(hs), 1.0))
    f["conf_mean"] = float(confs.mean())
    f["conf_min"] = float(confs.min())
    f["conf_last"] = float(confs[-1])
    # 末检测位置相对筐心（归一化）
    f["dist_final"] = float(np.hypot(xs[-1] - cx_h, ys[-1] - cy_h) / hw)
    f["x_off_final"] = float(abs(xs[-1] - cx_h) / hw)

    f["net_pre"] = net_pre if net_pre is not None else 0.0
    f["net_post"] = net_post if net_post is not None else 0.0
    f["net_ratio"] = (net_post / max(net_pre, 0.5)) if (net_post is not None and net_pre is not None) else 0.0
    return f


def extract(model, ball_classes, device, events, log_every=20):
    """events: dataset 记录列表。返回 (features_list, errors)。"""
    # 按视频分组 + 时间戳排序（减少 seek）
    by_video = {}
    for ev in events:
        resolved = _resolve_video(ev["video"])
        if resolved is None:
            continue
        by_video.setdefault(resolved, []).append(ev)

    out = []
    errors = []
    done = 0
    t0 = time.time()
    for video_path, evs in by_video.items():
        evs.sort(key=lambda e: e["ts"])
        try:
            reader = VideoReader(video_path)
        except Exception as e:
            for ev in evs:
                errors.append((ev["event_id"], f"open fail: {e}"))
            continue
        fps = reader.fps
        total = reader.total
        try:
            for ev in evs:
                ts = float(ev["ts"])
                hoop = ev.get("hoop")
                if not hoop:
                    errors.append((ev["event_id"], "no hoop"))
                    continue
                f0 = max(0, int((ts - WIN_PRE) * fps))
                f1 = min(total, int((ts + WIN_POST) * fps) + 1)
                if f1 - f0 < 8:
                    errors.append((ev["event_id"], "window too small"))
                    continue
                detections = []
                grays = []  # 缩小灰度帧（网区能量用）
                scale = 4
                for fidx, frame in reader.iter_frames(start=f0, end=f1):
                    # YOLO 逐帧（特征提取不复用/不跳帧，要最准序列）
                    try:
                        res = model.predict(frame, conf=BALL_CONF, imgsz=IMGSZ,
                                            classes=ball_classes,
                                            device=device, verbose=False)[0]
                    except Exception as e:
                        errors.append((ev["event_id"], f"yolo: {e}"))
                        res = None
                    if res is not None and res.boxes is not None and len(res.boxes) > 0:
                        xyxy = res.boxes.xyxy.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()
                        b = int(np.argmax(confs))
                        x1, y1, x2, y2 = xyxy[b]
                        detections.append(((fidx - f0) / fps,
                                           (x1 + x2) / 2, (y1 + y2) / 2,
                                           x1, y1, x2, y2, confs[b]))
                    # 缩小灰度
                    small = frame[::scale, ::scale]
                    g = small.mean(axis=2)
                    grays.append(g)
                # 网区（缩小坐标系）：底角45度机位网兜会向侧向+下方摆动，
                # ROI 左右各扩 0.25 筐宽、下沿延伸到 2.0 倍筐高
                hx1, hy1, hx2, hy2 = [float(v) for v in hoop]
                hh_r = hy2 - hy1
                hw_r = hx2 - hx1
                # clamp 必须在原始坐标系做（grays 高度是缩小后的，差 scale 倍）
                y2_orig = min(hy2 + 2.0 * hh_r, grays[0].shape[0] * scale)
                x1_orig = max(0.0, hx1 - 0.25 * hw_r)
                x2_orig = min(grays[0].shape[1] * scale, hx2 + 0.25 * hw_r)
                roi = (int(x1_orig / scale), int(hy2 / scale),
                       int(x2_orig / scale), int(y2_orig / scale))
                n = len(grays)
                pre_end = int((WIN_PRE - 0.5) * fps)  # ts-1.5 ~ ts-0.5 基线
                post_start = int(WIN_PRE * fps)       # ts ~ ts+1.5（后窗延长，捕捉球从网底穿出）
                post_end = n
                net_pre = _roi_mean_diff(grays, max(0, pre_end - int(1.0 * fps)),
                                         pre_end, roi) if pre_end > 2 else None
                net_post = _roi_mean_diff(grays, post_start, post_end, roi)
                feats = _compute_features(detections, net_pre, net_post, hoop,
                                          ev.get("video_width") or 0,
                                          ev.get("video_height") or 0, fps)
                feats["event_id"] = ev["event_id"]
                feats["label"] = ev["label"]
                feats["video"] = ev["video"]
                feats["ts"] = ev["ts"]
                out.append(feats)
                done += 1
                if done % log_every == 0:
                    dt = time.time() - t0
                    print(f"  [{done}/{len(events)}] {dt:.0f}s "
                          f"({done/dt*60:.0f} ev/min)", flush=True)
        finally:
            reader.close()
    return out, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个事件（测试用）")
    ap.add_argument("--out", default=str(FEATURES_FILE))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    device = get_device()
    if device == "cpu":
        print("ERROR: 无 CUDA，拒绝特征提取（GPU 是硬性要求）")
        sys.exit(1)
    print(f"device: {device}")

    dataset = json.loads(Path(args.out).parent.joinpath("dataset_v1.json").read_text(encoding="utf-8")) \
        if False else json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    events = dataset
    if args.limit:
        events = events[:args.limit]

    # 断点续跑：跳过已提取的
    done_ids = set()
    if Path(args.out).exists():
        for line in Path(args.out).read_text(encoding="utf-8").splitlines():
            try:
                done_ids.add(json.loads(line)["event_id"])
            except Exception:
                pass
        events = [e for e in events if e["event_id"] not in done_ids]
        print(f"断点续跑：已完成 {len(done_ids)}，本次处理 {len(events)}")

    if not events:
        print("没有待处理事件。")
        return

    model, weights = get_ball_model()
    ball_classes = get_ball_class_ids(model, weights)
    print(f"weights: {weights}, ball classes: {ball_classes}, events: {len(events)}")

    feats, errors = extract(model, ball_classes, device, events)
    # 追加写
    with open(args.out, "a", encoding="utf-8") as fo:
        for r in feats:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成：新提取 {len(feats)}，累计 {len(done_ids) + len(feats)}")
    if errors:
        print(f"错误 {len(errors)} 个（前 10）：")
        for eid, msg in errors[:10]:
            print(f"  {eid}: {msg}")


if __name__ == "__main__":
    main()
