# -*- coding: utf-8 -*-
"""按「部署几何」构建 YOLO 微调样本：整帧样本 + **裁剪块样本**（crop 档配方）。

与 training/build_yolo_dataset.py（整帧配方）的分工：
  · 整帧配方  用于「整帧 / 抹黑画布」推理路径（存储长边 1280，训练 imgsz 768）
  · 本脚本    用于**线上主档**「裁剪接受框」推理路径（训练 imgsz 640，
              球在输入里 ~22-33px），并把同一帧同时产出整帧版供兜底档使用
两份配方的采样策略与输出结构都不同，故并列存在而非合并开关。

背景：现役 basketball_ft.pt 训练时只见过「整帧 1920×1080 letterbox」，
而线上主档是「筐接受框裁出来的竖版小图（如 488×583）@640」。几何/尺度分布不同，
实测真裁剪@640 在 518 个真硬帧上命中 0%（详见 doc/BENCHMARKS.md「③」）。

关键做法（无需人工标注）：
  老师 = 现役模型在**整帧**上的预测（conf=0.05，与 build_yolo_dataset.py 同口径）
  学生样本 = 接受框 +20% 外扩的裁剪块，标签由老师框**投影到裁剪坐标系**
    · 中心落在裁剪块内的老师框 → 保留（坐标减去裁剪左上角）
    · 全部落在块外 → 该样本成为「裁剪画布下的背景样本」（教模型在小画布上别乱报）
每个原始样本同时产出整帧版与裁剪版，训练 imgsz=640 后球在两版里分别约 8px / 22-33px，
正是线上「主档 crop@640」与「兜底 mask@1280」两种推理的尺度。

盲测纪律（关键）：`--blind` 指定的日期**整天**排除出训练与验证，只用于评估。
默认排除 08.31（第一盲测日）与 09.11（第二盲测日）。val 另取事件数最少的 N 场，
仅作训练监控用，不能当作泛化判据。

用法:
  env\\Scripts\\python.exe -u training\\build_yolo_crop_dataset.py            # 全库
  env\\Scripts\\python.exe -u training\\build_yolo_crop_dataset.py --out=yolo_crop_data_10v --videos=10
  env\\Scripts\\python.exe -u training\\build_yolo_crop_dataset.py --blind=08.31
产物: training/<out>/{images,labels}/{train,val} + data.yaml + manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import get_ball_model, get_ball_class_ids          # noqa: E402
from services import detection                              # noqa: E402
from tracker import GoalDetector                            # noqa: E402
from training.build_yolo_dataset import detect_batch        # noqa: E402
from training.extract_frames_b import load_dataset_events   # noqa: E402
from video_io import VideoReader, get_video_info, read_frame  # noqa: E402

POS_OFFS = [-0.30, -0.15, 0.0, 0.15, 0.30]
NEG_OFFS = [-0.20, 0.0, 0.20]
N_EASY, N_BG = 8, 6
JPEG_Q = 88
MARGIN = 0.2          # 与 detection.YOLO_CROP_MARGIN 一致
random.seed(42)

OUT_DIR = PROJECT_ROOT / "training" / "yolo_crop_data"     # 由 --out 覆盖


def _p(rel):
    p = OUT_DIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def crop_spec(hoop, frame_shape):
    """返回裁剪块 (x1,y1,x2,y2)，与线上 _yolo_input 的 'crop' 档完全同几何。"""
    det = GoalDetector(hoop_box=hoop, loose_mode=True, yolo_confirm=True,
                       diff_threshold=25, min_blob_area=30, search_margin=80,
                       rolling_baseline_sec=0, auto_threshold=False, fps=30.0)
    return detection._yolo_input_box(det.yolo_accept_box(), frame_shape, margin=MARGIN)


def project_boxes(boxes, win):
    """把整帧坐标的老师框投影到裁剪块坐标系；中心在块外的丢弃。"""
    x1, y1, x2, y2 = win
    out = []
    for bx1, by1, bx2, by2, c in boxes:
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            out.append((bx1 - x1, by1 - y1, bx2 - x1, by2 - y1, c))
    return out


def _write_labels(rel, boxes, w, h):
    p = _p(rel)
    if not boxes:
        p.write_text("", encoding="utf-8")     # 空文件 = 背景样本
        return
    lines = [f"0 {(a+c)/2/w:.6f} {(b+d)/2/h:.6f} {(c-a)/w:.6f} {(d-b)/h:.6f}"
             for a, b, c, d, _conf in boxes]
    p.write_text("\n".join(lines), encoding="utf-8")


def save_pair(img, boxes, win, split_dir, stem, manifest, meta):
    """同时保存整帧样本与裁剪块样本（各带自己的标签），返回 (n_full, n_crop)。"""
    h, w = img.shape[:2]
    # ---- 整帧版 ----
    cv2.imwrite(str(_p(f"images/{split_dir}/{stem}.jpg")), img,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    _write_labels(f"labels/{split_dir}/{stem}.txt", boxes, w, h)
    manifest.append(dict(meta, file=f"images/{split_dir}/{stem}.jpg", kind="full",
                         frame_wh=[w, h], boxes=[[round(v, 1) for v in b] for b in boxes]))
    # ---- 裁剪版（部署几何）----
    x1, y1, x2, y2 = win
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return 1, 0
    ch, cw = crop.shape[:2]
    cboxes = project_boxes(boxes, win)
    cv2.imwrite(str(_p(f"images/{split_dir}/{stem}_c.jpg")), crop,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    _write_labels(f"labels/{split_dir}/{stem}_c.txt", cboxes, cw, ch)
    manifest.append(dict(meta, file=f"images/{split_dir}/{stem}_c.jpg", kind="crop",
                         frame_wh=[cw, ch], boxes=[[round(v, 1) for v in b] for b in cboxes]))
    return 1, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="yolo_crop_data",
                    help="training/ 下的产物目录名")
    ap.add_argument("--videos", type=int, default=0,
                    help="只用事件数最多的 N 场（0 = 全部）")
    ap.add_argument("--max-ev", type=int, default=0,
                    help="每场最多取多少事件（0 = 不限）")
    ap.add_argument("--blind", default="08.31,09.11",
                    help="盲测日（整天排除出训练与验证），逗号分隔；空串 = 不排除")
    ap.add_argument("--val-n", type=int, default=2,
                    help="取事件数最少的 N 场进 val（仅训练监控用）")
    args = ap.parse_args()

    global OUT_DIR
    OUT_DIR = PROJECT_ROOT / "training" / args.out
    blind = tuple(s.strip() for s in args.blind.split(",") if s.strip())

    t0 = time.time()
    events = load_dataset_events()
    if not events:
        print("ERROR: 无事件"); sys.exit(1)

    by_video = defaultdict(list)
    n_blind = 0
    for e in events:
        name = str(e.get("resolved", ""))
        if any(b in name for b in blind):
            n_blind += 1
            continue                      # 盲测日：整天不进训练/验证
        by_video[e["resolved"]].append(e)
    items = sorted(by_video.items(), key=lambda kv: str(kv[0]))
    if args.videos:
        items = sorted(items, key=lambda kv: -len(kv[1]))[:args.videos]
    # 取事件数最少的 N 场进 val：省训练数据，且盲测日本就在训练集之外
    if len(items) > args.val_n:
        val_names = {v for v, _ in sorted(items, key=lambda kv: len(kv[1]))[:args.val_n]}
    else:
        val_names = set()
    print('事件 %d → 排除盲测日 %s 的 %d 个 → 入选 %d 场 / %d 事件'
          % (len(events), '、'.join(blind) or '（无）', n_blind, len(items),
             sum(len(e) for _, e in items)), flush=True)
    print('val = %s（按事件数最少取 %d 场，仅训练监控用）'
          % (sorted(Path(v).stem for v in val_names), args.val_n), flush=True)

    model, weights = get_ball_model()
    cls = get_ball_class_ids(model, weights)
    print('老师模型: %s  classes=%s' % (Path(weights).name, cls), flush=True)

    manifest, stats = [], defaultdict(int)
    for vidx, (video, evs) in enumerate(items, 1):
        name = Path(video).stem
        split = 'val' if video in val_names else 'train'
        try:
            info = get_video_info(video)
        except Exception as e:
            print('[SKIP] %s: %s' % (name, e)); continue
        fps, total = info['fps'], info['total']
        evs.sort(key=lambda e: e['ts'])
        random.shuffle(evs)
        if args.max_ev:
            evs = evs[:args.max_ev]
        evs = sorted(evs, key=lambda e: e['ts'])
        if not evs:
            print('[SKIP] %s: 无事件' % name); continue
        # 该场标定出的筐（同场各事件基本一致；易例/背景阶段也用它裁块）
        easy_hoop = [float(v) for v in evs[0]['hoop']]

        reader, n_pair, n_crop_lbl = None, 0, 0
        try:
            reader = VideoReader(video)
            for ev in evs:
                hoop = [float(v) for v in ev['hoop']]
                offs = POS_OFFS if ev['label'] == 'pos' else NEG_OFFS
                want = {}
                for off in offs:
                    fi = max(0, min(int((ev['ts'] + off) * fps), total - 1))
                    want[fi] = off
                got = {}
                for fi, frame in reader.iter_frames(start=min(want), end=max(want) + 1):
                    if fi in want:
                        got[fi] = frame
                if not got:
                    continue
                idxs = sorted(got)
                boxes_per = detect_batch(model, cls, [got[i] for i in idxs])
                for fi, boxes in zip(idxs, boxes_per):
                    frame = got[fi]
                    win = crop_spec(hoop, frame.shape)
                    if win is None:
                        continue
                    stem = 'v%03d_%06d' % (vidx, fi)
                    nf, nc = save_pair(frame, boxes, win, split, stem, manifest,
                                       {'source': 'event', 'video': name, 'ts': ev['ts'],
                                        'label': ev['label'], 'hoop': hoop})
                    n_pair += nf
                    n_crop_lbl += nc
        except Exception as e:
            print('[%s] 事件阶段异常: %s' % (name, e))
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass

        # 易例/背景（整帧抽，裁剪按同一几何生成）
        try:
            cand = random.sample(range(total), min(40, total))
            n_easy = n_bg = 0          # 每场各自的配额（不能用全局计数，否则从第二场起就不加了）
            for fi in cand:
                if n_easy >= N_EASY and n_bg >= N_BG:
                    break
                frame = read_frame(video, fi, total=total, fps=fps)
                if frame is None:
                    continue
                bs = detect_batch(model, cls, [frame])[0]
                has = any(b[4] >= 0.4 for b in bs)
                if has and n_easy < N_EASY:
                    n_easy += 1
                    bs = [b for b in bs if b[4] >= 0.4]
                elif (not bs) and n_bg < N_BG:
                    n_bg += 1
                else:
                    continue
                win = crop_spec(easy_hoop, frame.shape)
                if win is None:
                    continue
                stem = 'v%03d_e%06d' % (vidx, fi)
                save_pair(frame, bs, win, split, stem, manifest,
                          {'source': 'easy' if has else 'bg', 'video': name})
        except Exception as e:
            print('[%s] 易例阶段异常: %s' % (name, e))

        stats['pair'] += n_pair
        print('[%d/%d] %-28s split=%s 帧样本 %d（裁剪版含标签 %d）  (%.1f min)'
              % (vidx, len(items), name, split, n_pair, n_crop_lbl,
                 (time.time() - t0) / 60), flush=True)

    (OUT_DIR / 'data.yaml').write_text(
        'path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: basketball\n'
        % OUT_DIR.as_posix(), encoding='utf-8')
    with open(OUT_DIR / 'manifest.jsonl', 'w', encoding='utf-8') as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    n_full = sum(1 for r in manifest if r['kind'] == 'full')
    n_crop = len(manifest) - n_full
    n_crop_bg = sum(1 for r in manifest if r['kind'] == 'crop' and not r['boxes'])
    n_val = sum(1 for r in manifest if '/val/' in r['file'])
    print('\n完成: 图像 %d（整帧 %d / 裁剪 %d，其中裁剪背景 %d）| val %d | 耗时 %.1f min'
          % (len(manifest), n_full, n_crop, n_crop_bg, n_val, (time.time() - t0) / 60))
    print('统计: %s' % dict(stats))
    print('产物: %s' % OUT_DIR)


if __name__ == '__main__':
    main()
