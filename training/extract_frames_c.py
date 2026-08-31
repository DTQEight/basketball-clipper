# -*- coding: utf-8 -*-
"""方向4+3a 统一抽取：球轨迹 + 球心裁剪 + 全局低分辨率流。

与 extract_frames_b（筐心固定裁剪）的差异：
  - 每事件对 ±1.5s 做密集球检测（每 2 帧一次，batch 推理），缓存球轨迹
  - 16 个 FRAME_OFFS 帧各出两路 224 裁剪：
      x_ball  以该帧（±0.1s 内）球位置为中心的局部块（无球回退筐心裁剪）
      x_glob  整帧 letterbox 到 224（全局上下文流）
  - 轨迹存 tracks_c/{eid}.npz，供轨迹渲染图（方向3a）复用

输出：
  frames_c/{eid}.npz   x_ball/x_glob (16,224,224,3) uint8 BGR
  tracks_c/{eid}.npz   fidx/cx/cy/conf
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from video_io import VideoReader  # noqa: E402
from training.extract_frames_b import (  # noqa: E402
    load_dataset_events, FRAME_OFFS, SIZE, ZOOM)

OUT_DIR = PROJECT_ROOT / "training" / "frames_c"
TRACK_DIR = PROJECT_ROOT / "training" / "tracks_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRACK_DIR.mkdir(parents=True, exist_ok=True)

BALL_ZOOM = 2.6      # 球心裁剪边长 = 2.6×hoop_h（比筐心 3.2 略紧，突出球体）
TRACK_STEP = 2       # 密集球检测抽帧间隔
BALL_CONF = 0.30     # 轨迹抽取用低阈值（漏检比错检代价大，错检由距离门限兜底）
BALL_MIN_CONF = 0.35 # 裁剪采用的最低置信度
MATCH_WIN = 3        # offset 帧与轨迹点的最大帧距（±0.1s@30fps）
BATCH = 16           # YOLO 批量推理帧数（4GB 显存 imgsz640 安全）


def _letterbox(frame: np.ndarray, size: int = SIZE, pad: int = 114) -> np.ndarray:
    """整帧等比缩放到 size×size（留边填充），保留全局空间结构。"""
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), pad, dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def _crop_center(frame: np.ndarray, cx: float, cy: float, side: float,
                 size: int = SIZE, pad: int = 114) -> np.ndarray:
    """以 (cx, cy) 为中心裁 side×side，越界填灰，缩放到 size。"""
    h, w = frame.shape[:2]
    cx1, cy1 = int(cx - side / 2), int(cy - side / 2)
    cx2, cy2 = cx1 + int(side), cy1 + int(side)
    px1, py1 = max(0, cx1), max(0, cy1)
    px2, py2 = min(w, cx2), min(h, cy2)
    out = np.full((int(side), int(side), 3), pad, dtype=np.uint8)
    if px2 > px1 and py2 > py1:
        out[py1 - cy1:py2 - cy1, px1 - cx1:px2 - cx1] = frame[py1:py2, px1:px2]
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)


def crop_hoop_box(frame: np.ndarray, hoop, zoom: float = ZOOM, size: int = SIZE):
    """与 extract_frames_b.crop_hoop 一致的筐心裁剪（无球回退用）。"""
    x1, y1, x2, y2 = [float(v) for v in hoop]
    hc, vc = (x1 + x2) / 2, (y1 + y2) / 2
    hh = max(y2 - y1, 1.0)
    return _crop_center(frame, hc, vc, zoom * hh, size)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from ultralytics import YOLO
    model = YOLO(str(PROJECT_ROOT / "weights" / "basketball_custom.pt"))
    names = {str(v).strip().lower() for v in (model.names or {}).values()}
    if not names & {"basketball", "ball"}:
        print(f"ERROR: 权重类别表不含篮球类: {names}")
        sys.exit(1)
    device = "cuda"

    events = load_dataset_events()
    events = [e for e in events if e.get("hoop")]
    todo = [e for e in events
            if not (OUT_DIR / f"{e['event_id']}.npz").exists()
            or not (TRACK_DIR / f"{e['event_id']}.npz").exists()]
    print(f"events={len(events)} 待抽={len(todo)}", flush=True)
    if not todo:
        return

    t0 = time.time()
    done = 0
    for ev in todo:
        eid = ev["event_id"]
        try:
            reader = VideoReader(ev["resolved"])
        except Exception:
            print(f"[SKIP] open fail: {ev['video']}", flush=True)
            continue
        try:
            fps, total = reader.fps, reader.total
            ts = float(ev["ts"])
            hoop = ev["hoop"]
            hh = max(float(hoop[3]) - float(hoop[1]), 1.0)

            start = max(0, int((ts + FRAME_OFFS[0]) * fps))
            last = min(total - 1, int((ts + FRAME_OFFS[-1]) * fps))
            want = {}
            for i, off in enumerate(FRAME_OFFS):
                fidx = max(0, min(int((ts + off) * fps), total - 1))
                want[fidx] = i

            x_ball = np.zeros((len(FRAME_OFFS), SIZE, SIZE, 3), dtype=np.uint8)
            x_glob = np.zeros((len(FRAME_OFFS), SIZE, SIZE, 3), dtype=np.uint8)
            frames_at = {}       # fidx -> frame（offset 帧）
            pending = []         # (fidx, frame) 待 YOLO
            tr_fidx, tr_cx, tr_cy, tr_conf = [], [], [], []

            # 单趟顺序解码：offset 帧留下裁剪，密集网格帧进 YOLO 队列
            last_decoded = None
            for fidx, frame in reader.iter_frames(start=start, end=last + 1):
                last_decoded = frame
                if fidx in want:
                    frames_at[fidx] = frame
                if (fidx - start) % TRACK_STEP == 0 or fidx in want:
                    pending.append((fidx, frame))
                    if len(pending) >= BATCH:
                        _run_yolo(model, pending, device, tr_fidx, tr_cx, tr_cy, tr_conf)
                        pending = []
            if pending:
                _run_yolo(model, pending, device, tr_fidx, tr_cx, tr_cy, tr_conf)

            tr_fidx = np.asarray(tr_fidx, dtype=np.int32)
            tr_cx = np.asarray(tr_cx, dtype=np.float32)
            tr_cy = np.asarray(tr_cy, dtype=np.float32)
            tr_conf = np.asarray(tr_conf, dtype=np.float32)

            for fidx, i in want.items():
                frame = frames_at.get(fidx)
                if frame is None:
                    frame = last_decoded   # 解码跳帧：用最近已解码帧近似
                if frame is None:
                    continue
                x_glob[i] = _letterbox(frame)
                # ±MATCH_WIN 帧内找置信度最高的球
                m = (np.abs(tr_fidx - fidx) <= MATCH_WIN) & (tr_conf >= BALL_MIN_CONF)
                if m.any():
                    k = np.argmax(np.where(m, tr_conf, -1))
                    x_ball[i] = _crop_center(frame, tr_cx[k], tr_cy[k],
                                             BALL_ZOOM * hh)
                else:
                    x_ball[i] = crop_hoop_box(frame, hoop)

            np.savez_compressed(OUT_DIR / f"{eid}.npz",
                                x_ball=x_ball, x_glob=x_glob)
            np.savez_compressed(TRACK_DIR / f"{eid}.npz",
                                fidx=tr_fidx, cx=tr_cx, cy=tr_cy, conf=tr_conf)
            done += 1
            if done % 50 == 0:
                dt = time.time() - t0
                eta = dt / done * (len(todo) - done)
                print(f"  [{done}/{len(todo)}] {dt / 60:.1f}min "
                      f"({done / dt * 60:.0f} ev/min) ETA {eta / 60:.1f}min",
                      flush=True)
        except Exception as e:
            print(f"[ERR] {eid}: {e}", flush=True)
        finally:
            reader.close()
    print(f"完成 {done}/{len(todo)}，耗时 {(time.time() - t0) / 60:.1f}min")


def _run_yolo(model, pending, device, tr_fidx, tr_cx, tr_cy, tr_conf):
    """batch 推理，每帧只保留最高置信度球框。"""
    imgs = [f for _, f in pending]
    res = model.predict(imgs, conf=BALL_CONF, imgsz=640, classes=[0],
                        device=device, verbose=False)
    for (fidx, _), r in zip(pending, res):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        confs = r.boxes.conf.cpu().numpy()
        k = int(np.argmax(confs))
        x1, y1, x2, y2 = r.boxes.xyxy.cpu().numpy()[k]
        tr_fidx.append(fidx)
        tr_cx.append((x1 + x2) / 2)
        tr_cy.append((y1 + y2) / 2)
        tr_conf.append(float(confs[k]))
    pending.clear()


if __name__ == "__main__":
    main()
