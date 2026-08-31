# -*- coding: utf-8 -*-
"""L4 阶段 A 标注工具（离线 CLI，不依赖 NiceGUI / CUDA / YOLO）。

目标：把 detection_history.json 中每个「已注册进球时间戳」切成 6s 预览片段，
并通过交互式三态标注（正=进球 / 负=误报 / ?=不确定）积累训练标签。

用法（在项目根执行）：
  env\\python.exe training\\annotate.py scan
      扫描历史记录，打印事件总量、已标注进度、缺视频清单
  env\\python.exe training\\annotate.py slice [--video PATH]
      为所有（或指定）视频的历史进球切 480p 预览 mp4，存 training/clips/
  env\\python.exe training\\annotate.py label [--video PATH] [--start N]
      交互式标注。键位：1=真进球 0=误报 ?=不确定  n/下一个  p/上一个  q/退出  g/跳转到编号
  env\\python.exe training\\annotate.py export
      导出 training/dataset_v1.json（A 阶段可用的格式：事件元信息 + 标签）

数据目录固定：E:\\basketball-project\\training\\（C 盘空间紧张时的安全位置）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess as _sp
import sys
import time
from collections import defaultdict
from pathlib import Path

# ===== 路径常量（强绑定 E 盘，不受 BBALL_CACHE_ROOT 环境变量影响） =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"  # 固定：e:\basketball-project\training
CLIPS_DIR = TRAINING_DIR / "clips"
LABELS_FILE = TRAINING_DIR / "labels.jsonl"
DATASET_FILE = TRAINING_DIR / "dataset_v1.json"
HISTORY_FILE = PROJECT_ROOT / "cache" / "detection_history.json"

TRAINING_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ts"}


# 额外的历史来源（用户 NAS 备份里的其他副本；不在盘就静默忽略）
_EXTRA_HISTORY_FILES = [
    Path(r"E:\ad-project\cache\detection_history.json"),
]

# =================== 底层小工具 ===================

def _load_history():
    """读取所有已知历史文件并按 (video_path) 去重：
    同一视频有多条时保留 hoop/goals 字段最丰富的那条（优先带标定的）。
    顺序不保证，调用方在 _collect_events 自行按时间戳排序。
    """
    all_records = []
    sources = [HISTORY_FILE] + _EXTRA_HISTORY_FILES
    for src in sources:
        if not src.exists():
            continue
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
            if isinstance(data, list):
                all_records.extend(data)
                print(f"[HIST] 读取 {src}: {len(data)} 条")
        except Exception as e:
            print(f"[WARN] 读历史 {src} 失败: {e}")
    if not all_records:
        return []
    # 按 video 去重：保留字段最完整（有 hoop + 有 goals 数量多）的一条
    best = {}
    for r in all_records:
        v = r.get("video")
        if not v:
            continue
        score = (bool(r.get("hoop")), len(r.get("goals") or []),
                 bool(r.get("video_width")), bool(r.get("time")))
        prev = best.get(v)
        if prev is None:
            best[v] = r
            continue
        prev_score = (bool(prev.get("hoop")), len(prev.get("goals") or []),
                      bool(prev.get("video_width")), bool(prev.get("time")))
        if score > prev_score:
            best[v] = r
    return list(best.values())


def _video_id(video_path: str) -> str:
    """短 ID：basename + 头 1MB md5 前 8 位，用于 clip/label 文件名去重。"""
    if not os.path.exists(video_path):
        # 不在盘就退化为 basename md5
        return hashlib.md5(os.path.basename(video_path).encode("utf-8")).hexdigest()[:8]
    try:
        with open(video_path, "rb") as f:
            head = f.read(1 << 20)
        return hashlib.md5(head).hexdigest()[:8]
    except Exception:
        return hashlib.md5(video_path.encode("utf-8")).hexdigest()[:8]


def _event_id(video_path: str, ts: float) -> str:
    """事件 ID = 视频短ID + 时间戳（ms 精度），同一视频同进球是确定性的，可反复覆盖写。"""
    vid = _video_id(video_path)
    return f"{vid}_{int(round(float(ts) * 1000)):010d}"


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # 回退：项目自带 ffmpeg / 系统 ffmpeg
        ff = shutil.which("ffmpeg")
        if ff:
            return ff
        raise RuntimeError("找不到 ffmpeg：请安装 imageio-ffmpeg 或系统 ffmpeg")


def _build_preview_encode_args(ff: str):
    """与 detection._generate_preview_clips 相同：480p + 预览码率。"""
    try:
        from cutter.ffmpeg_cutter import _build_encode_args
        return _build_encode_args(ff, quality="preview")
    except Exception:
        # 兜底，硬编码合理参数
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-an"]


def _resolve_video_path(target_path: str) -> str | None:
    r"""历史记录里的路径可能是子文件夹（如 D:\Downloads\7.27\xxx.mp4），
    而用户实际把所有视频平放在 D:\Downloads\test\xxx.mp4。
    优先原路径 → 否则 fallback D:\Downloads\test\{basename}。"""
    if os.path.exists(target_path):
        return target_path
    candidates = [
        Path(r"D:\Downloads\test") / os.path.basename(target_path),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


# =================== 扫描 ===================

def _collect_events(hist, filter_video=None):
    """返回 [(video_path, resolved_path, ts, hoop_info, idx_in_video), ...]"""
    events = []
    seen_files = set()
    for r in hist:
        v = r.get("video")
        if not v:
            continue
        if filter_video and os.path.basename(v) != os.path.basename(filter_video) and v != filter_video:
            continue
        # 去重：同路径保留最新（一般 hist 只有一条）
        if v in seen_files:
            continue
        seen_files.add(v)
        resolved = _resolve_video_path(v)
        goals = r.get("goals") or []
        hoop = r.get("hoop")
        for gi, g in enumerate(goals):
            events.append((v, resolved, float(g), hoop, gi))
    return events


def cmd_scan(_args):
    hist = _load_history()
    events = _collect_events(hist)
    # 已标注事件
    labeled = _load_labels()  # {event_id: {label, ts, video, ...}}

    # 按视频分组统计
    by_video = defaultdict(lambda: {"total": 0, "resolved": 0,
                                     "labeled": {"pos": 0, "neg": 0, "?": 0}})
    missing_videos = []
    for orig_v, resolved, ts, hoop, gi in events:
        info = by_video[orig_v]
        info["total"] += 1
        if resolved:
            info["resolved"] += 1
        eid = _event_id(orig_v, ts)
        if eid in labeled:
            lab = labeled[eid].get("label")
            if lab in ("pos", "neg", "?"):
                info["labeled"][lab] += 1
    # 找不到 resolve 的视频列出来
    missing_videos = sorted({v for v, r, *_ in events if r is None})

    print(f"历史记录: {len(hist)} 条  唯一视频: {len(by_video)}  总事件: {len(events)}")
    n_resolved = sum(i["resolved"] for i in by_video.values())
    print(f"视频已在盘: {sum(1 for v,i in by_video.items() if i['resolved']>0)}/{len(by_video)}  事件可切: {n_resolved}/{len(events)}")
    pos = sum(i["labeled"].get("pos", 0) for i in by_video.values())
    neg = sum(i["labeled"].get("neg", 0) for i in by_video.values())
    unk = sum(i["labeled"].get("?", 0) for i in by_video.values())
    print(f"标注进度: 真={pos}  假={neg}  不确定={unk}  合计={pos+neg+unk}/{len(events)}")
    print()
    print("— 按视频明细 —")
    for v, i in sorted(by_video.items(), key=lambda kv: -kv[1]["total"]):
        lp = i["labeled"].get("pos", 0)
        ln = i["labeled"].get("neg", 0)
        lu = i["labeled"].get("?", 0)
        lpc = i["total"] - (lp + ln + lu)
        print(f"  {'✓' if i['resolved']>0 else '✗'} {v}  total={i['total']}  +{lp}/-{ln}/?{lu}  待标注={lpc}")
    if missing_videos:
        print()
        print("— 缺失视频（需从 NAS 恢复）—")
        for v in missing_videos:
            print(f"  {v}")


# =================== 切片段 ===================

def cmd_slice(args):
    hist = _load_history()
    events = _collect_events(hist, filter_video=args.video)
    events = [(v, r, ts, hoop, gi) for v, r, ts, hoop, gi in events if r]  # 只切在盘的
    if not events:
        print("没有可切片的事件（可能视频都不在盘）")
        return
    ff = _ffmpeg_exe()
    enc_args = _build_preview_encode_args(ff)
    SBOX = 0x08000000 if os.name == "nt" else 0

    done = 0
    skipped = 0
    t0 = time.time()
    # 按视频分组，一个视频顺序切（避免来回开 ffmpeg）
    by_video_events = defaultdict(list)
    for v, resolved, ts, hoop, gi in events:
        by_video_events[(v, resolved)].append((ts, gi))

    for (orig_v, resolved_v), items in by_video_events.items():
        # 用 video_io 取 fps；失败回退 30
        fps = 30.0
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from video_io import get_video_info
            info = get_video_info(resolved_v)
            fps = float(info.get("fps") or fps)
        except Exception:
            pass
        for ts, gi in items:
            out_name = f"{_event_id(orig_v, ts)}.mp4"
            out_path = CLIPS_DIR / out_name
            if out_path.exists() and out_path.stat().st_size > 0:
                skipped += 1
                done += 1
                continue
            start_sec = max(0.0, ts - 3.0)
            end_sec = ts + 3.0
            dur_sec = end_sec - start_sec
            try:
                _sp.run([ff, "-y", "-loglevel", "error",
                         "-ss", f"{start_sec:.3f}", "-i", resolved_v,
                         "-t", f"{dur_sec:.3f}",
                         "-vf", "scale=-2:480"] + enc_args +
                        ["-movflags", "+faststart", str(out_path)],
                        creationflags=SBOX, capture_output=True, timeout=120)
                ok = out_path.exists() and out_path.stat().st_size > 0
                if ok:
                    done += 1
                else:
                    print(f"[FAIL] {orig_v}@{ts:.1f}s 生成失败（空文件）")
            except Exception as e:
                print(f"[ERR]  {orig_v}@{ts:.1f}s: {e}")
        print(f"  切完 {os.path.basename(orig_v)}: {len(items)} 个片段  ({time.time()-t0:.0f}s)")

    print(f"切片完成: 新切 {done-skipped}  已存在跳过 {skipped}  耗时 {time.time()-t0:.0f}s")


# =================== 标签读写 ===================

def _load_labels() -> dict:
    out = {}
    if not LABELS_FILE.exists():
        return out
    for line in LABELS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        eid = rec.get("event_id")
        if eid:
            out[eid] = rec
    return out


def _write_label(rec: dict):
    """幂等写：若该 event_id 已存在，先剔除旧记录再追加新记录（保留单行=最新）。"""
    eid = rec["event_id"]
    if not LABELS_FILE.exists():
        with open(LABELS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return
    tmp_path = LABELS_FILE.with_suffix(".jsonl.tmp")
    written = False
    with open(LABELS_FILE, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
        for line in src:
            try:
                old = json.loads(line)
            except Exception:
                dst.write(line)
                continue
            if old.get("event_id") == eid:
                # 跳过旧的；循环结束后统一在文件末尾写新的
                continue
            dst.write(line)
        dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
    try:
        os.replace(tmp_path, LABELS_FILE)
    except Exception:
        # Windows 跨盘偶尔 fail，退化为重写全量
        shutil.move(str(tmp_path), str(LABELS_FILE))


# =================== 交互标注 ===================

def cmd_label(args):
    hist = _load_history()
    events = _collect_events(hist, filter_video=args.video)
    # 只显示在盘且片段已切好的
    ready = []
    missing_clip = []
    for v, resolved, ts, hoop, gi in events:
        if not resolved:
            continue
        eid = _event_id(v, ts)
        clip_path = CLIPS_DIR / f"{eid}.mp4"
        if clip_path.exists() and clip_path.stat().st_size > 0:
            ready.append({"video": v, "resolved": resolved, "ts": ts,
                          "hoop": hoop, "idx": gi, "event_id": eid,
                          "clip_path": str(clip_path)})
        else:
            missing_clip.append((v, ts))
    if not ready:
        print("没有可标注的事件。先运行 annotate.py slice 切片段。")
        if missing_clip:
            print(f"（有 {len(missing_clip)} 个事件待切）")
        return
    labeled = _load_labels()

    start = max(0, int(args.start or 0))
    i = min(start, len(ready) - 1)

    def _fmtbar():
        lab = labeled.get(ready[i]["event_id"], {}).get("label")
        sym = {"pos": "✓真", "neg": "✗假", "?": "?不"}.get(lab, "未标")
        ev = ready[i]
        prog = f"[{i+1}/{len(ready)}]"
        name = os.path.basename(ev["video"])
        m, s = divmod(ev["ts"], 60)
        return f"{prog} {sym}  {name}  @ {int(m):02d}:{s:05.2f}   clip={ev['clip_path']}"

    print("开始标注。键位：1=真进球  0=误报  ?=不确定  n=下一个  p=上一个  g N=跳转到第N条  o=重开播放器  q=保存并退出")
    print("（每个事件会自动用默认播放器打开 6s 预览片段；看完按键即可）")
    print()
    last_opened_eid = None

    def _open_clip():
        """用系统默认播放器打开当前片段（仅事件变化时，避免 n/p 来回翻重复弹窗）。"""
        nonlocal last_opened_eid
        ev = ready[i]
        if ev["event_id"] == last_opened_eid:
            return
        last_opened_eid = ev["event_id"]
        try:
            if os.name == "nt":
                os.startfile(ev["clip_path"])  # noqa: S606
            else:
                _sp.Popen(["xdg-open", ev["clip_path"]])
        except Exception as e:
            print(f"[WARN] 打开播放器失败（可手动打开）: {e}  clip={ev['clip_path']}")

    _open_clip()
    while True:
        print(_fmtbar())
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出（所有已写标签都在磁盘）")
            return
        if not line:
            continue
        tok = line.split()
        k = tok[0].lower()
        if k == "q":
            print("已保存退出。")
            return
        elif k in ("1", "y", "j"):
            lab = "pos"
        elif k in ("0", "x", "k"):
            lab = "neg"
        elif k in ("?", "u", "s"):
            lab = "?"
        elif k == "o":
            last_opened_eid = None
            _open_clip()
            continue
        elif k in ("n", "enter", ""):
            i = min(len(ready) - 1, i + 1)
            _open_clip()
            continue
        elif k in ("p", "b"):
            i = max(0, i - 1)
            _open_clip()
            continue
        elif k == "g" and len(tok) >= 2:
            try:
                i = max(0, min(len(ready) - 1, int(tok[1]) - 1))
            except ValueError:
                pass
            _open_clip()
            continue
        else:
            print("键位：1 真  0 假  ? 不确定  n  p  o  g N  q")
            continue
        # 写标签
        ev = ready[i]
        rec = {
            "event_id": ev["event_id"],
            "video": ev["video"],
            "ts": round(ev["ts"], 3),
            "hoop": ev["hoop"],
            "idx_in_video": ev["idx"],
            "clip_path": ev["clip_path"],
            "label": lab,
            "label_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            _write_label(rec)
            labeled[ev["event_id"]] = rec
        except Exception as e:
            print(f"[ERR] 写标签失败: {e}")
        # 下一条
        i = min(len(ready) - 1, i + 1)
        _open_clip()


# =================== 导出 A 阶段训练集 ===================

def cmd_export(_args):
    hist = _load_history()
    labeled = _load_labels()
    # UI 标注（detection_history.json 的 labels 块）：kept→pos deleted→neg，
    # 以 (video, ts) 合并，UI 优先（比离线标注新）
    ui_labels = {}
    for r in hist:
        lab = r.get("labels") or {}
        for t in lab.get("kept") or []:
            ui_labels[(r["video"], round(float(t), 3))] = "pos"
        for t in lab.get("deleted") or []:
            ui_labels[(r["video"], round(float(t), 3))] = "neg"

    # 离线标注 → 同 key 结构；UI 覆盖离线
    merged = {}
    for eid, rec in labeled.items():
        key = (rec.get("video"), round(float(rec["ts"]), 3))
        merged[key] = {"label": rec.get("label"), "rec": rec, "src": "offline"}
    for key, lab in ui_labels.items():
        if lab not in ("pos", "neg"):
            continue
        if key in merged:
            merged[key]["label"] = lab
            merged[key]["src"] = "ui-overwrite"
        else:
            merged[key] = {"label": lab, "rec": None, "src": "ui"}

    # 从历史拿视频元信息（fps、分辨率、标定）
    meta_by_video = {}
    for r in hist:
        meta_by_video[r["video"]] = r
    # UI-only 事件：hoop/fps/尺寸从该视频历史记录取
    records = []
    pos = neg = 0
    src_count = defaultdict(int)
    for (video, ts3), item in sorted(merged.items()):
        lab = item["label"]
        if lab not in ("pos", "neg"):
            continue
        rec = item["rec"] or {}
        meta = meta_by_video.get(video, {})
        # ts 取原始精度（离线 rec 优先，UI 用 round 后的）
        ts = rec.get("ts", ts3)
        eid = _event_id(video, ts)
        clip = rec.get("clip_path")
        if clip and not Path(clip).exists():
            clip = None  # clip 只服务离线回看，特征提取读原视频，缺失不阻导出
        records.append({
            "event_id": eid,
            "video": video,
            "ts": ts,
            "clip_path": clip,
            "label": 1 if lab == "pos" else 0,
            "hoop": rec.get("hoop") or meta.get("hoop"),
            "video_fps": meta.get("video_fps"),
            "video_width": meta.get("video_width"),
            "video_height": meta.get("video_height"),
            "video_duration_sec": meta.get("video_duration_sec"),
            "yolo_confirmed_history": meta.get("yolo_confirmed"),
            "yolo_rejected_history": meta.get("yolo_rejected"),
            "detect_time": meta.get("time"),
            "label_time": rec.get("label_time") or (meta.get("labels") or {}).get("label_time"),
            "label_source": item["src"],
        })
        if lab == "pos":
            pos += 1
        else:
            neg += 1
        src_count[item["src"]] += 1
    DATASET_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"导出 dataset_v1.json：正={pos} 负={neg} 总计={len(records)}")
    print(f"标签来源：{dict(src_count)}（UI 优先于离线）")
    print(f"输出: {DATASET_FILE}")


# =================== CLI 入口 ===================

def _build_argparser():
    p = argparse.ArgumentParser(description="篮球进球检测 · L4 标注工具",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scan", help="扫描历史记录，显示可标注/已标注进度")

    ps = sub.add_parser("slice", help="切 480p 预览片段到 training/clips/")
    ps.add_argument("--video", default=None, help="仅处理该视频（路径或 basename）")

    pl = sub.add_parser("label", help="交互式标注")
    pl.add_argument("--video", default=None, help="仅标注该视频")
    pl.add_argument("--start", type=int, default=0, help="从第 N 条事件开始（1-based）")

    sub.add_parser("export", help="导出训练集 dataset_v1.json")
    return p


def main(argv=None):
    parser = _build_argparser()
    args = parser.parse_args(argv)
    # 重定向 stdout 到 utf-8（终端中文安全）
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if args.cmd == "scan":
        cmd_scan(args)
    elif args.cmd == "slice":
        cmd_slice(args)
    elif args.cmd == "label":
        cmd_label(args)
    elif args.cmd == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
