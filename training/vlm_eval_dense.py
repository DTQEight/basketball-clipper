# -*- coding: utf-8 -*-
"""VLM 抽帧加密 A/B：ts±0.3s 从每 0.1s 一帧改成每 1/15s≈0.067s 一帧（16→22 张）。

用法：
  测 qwen（百炼 dashscope，用 VLM_KEY_QWEN 或 VLM_KEY）：
    $env:VLM_KEY_QWEN='sk-xxx' ; env\python.exe training\vlm_eval_dense.py --model qwen
  测 deepseek（用 VLM_KEY_DS 或 VLM_KEY）：
    $env:VLM_KEY_DS='sk-xxx' ; env\python.exe training\vlm_eval_dense.py --model deepseek
  默认两者都跑：
    $env:VLM_KEY='百炼key'; $env:VLM_KEY_DS='deepseek-key'; env\python.exe training\vlm_eval_dense.py
结果分别写到 training\vlm_eval_{model}_dense.jsonl。
完成后用 vlm_eval_dense_report.py 对比原版与加密版的增量。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from video_io import VideoReader  # noqa: E402
from vlm_eval import (PROMPT, parse_answer, compute_oof,  # noqa: E402
                      build_sample, FEATURES_FILE)

TRAINING_DIR = PROJECT_ROOT / "training"

# 原版 16 帧：稀采 + t=±0.5s 内每 0.1s 一帧
# 加密版：保持稀采不变，穿筐窗口 ±0.3s 从 [ -0.2,-0.1,0.0,0.1,0.2,0.3 ]（6 帧）
#         升级为 1/15s ≈ 0.0667s 一帧 → 10 帧，多 4 张（+ 前后加密缓冲区 2 张）
#         总共 16 + 6 = 22 张
DENSE_OFFSETS = [
    -1.5, -1.0, -0.6, -0.3,
    -0.266, -0.2, -0.133, -0.067,
    0.0, 0.067, 0.133, 0.2, 0.267,
    0.333, 0.4, 0.467, 0.533, 0.6, 0.7,
    1.0, 1.4,
]
MAX_W = 640
JPEG_Q = 80
VIDEO_ROOTS = [Path(r"D:\Downloads\test"), Path(r"D:\Downloads")]

MODEL_CONFIGS = {
    "qwen": {
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3-vl-plus",
        "result_file": TRAINING_DIR / "vlm_eval_qwen_dense.jsonl",
        "env_keys": ["VLM_KEY_QWEN", "VLM_KEY"],
        "extra_payload": {},
    },
    "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-v4-flash-vision-exp",
        "result_file": TRAINING_DIR / "vlm_eval_deepseek_dense.jsonl",
        "env_keys": ["VLM_KEY_DS", "VLM_KEY"],
        "extra_payload": {"max_tokens": 400, "thinking": {"type": "disabled"}},
    },
}


def resolve_video(path_str):
    p = Path(path_str)
    if p.exists():
        return str(p)
    name = p.name
    for root in VIDEO_ROOTS:
        cand = root / name
        if cand.exists():
            return str(cand)
    for root in VIDEO_ROOTS:
        if root.exists():
            try:
                for hit in root.rglob(name):
                    return str(hit)
            except OSError:
                continue
    return None


def extract_frames_dense(ev):
    """与 vlm_eval.extract_frames 相同的裁剪放大 + 画绿框，但用加密 DENSE_OFFSETS。"""
    reader = VideoReader(ev["video"])
    fps, total = reader.fps, reader.total
    ts = ev["ts"]
    t0 = max(0.0, ts + DENSE_OFFSETS[0] - 0.05)
    t1 = min(total / fps, ts + DENSE_OFFSETS[-1] + 0.05)
    f0, f1 = int(t0 * fps), int(t1 * fps)
    targets = {round(o, 3): None for o in DENSE_OFFSETS}
    try:
        for fidx, frame in reader.iter_frames(start=f0, end=f1):
            t = fidx / fps
            for o in targets:
                if targets[o] is None and abs(t - ts - o) <= 0.5 / fps + 1e-6:
                    targets[o] = frame
    finally:
        reader.close()
    hoop = ev.get("hoop")
    cx1 = cy1 = cx2 = cy2 = None
    if crop := bool(hoop and len(hoop) == 4):
        hx1, hy1, hx2, hy2 = [float(v) for v in hoop]
        hw, hh = hx2 - hx1, max(hy2 - hy1, 1)
        ccx, ccy = (hx1 + hx2) / 2, (hy1 + hy2) / 2 - hh * 0.5
        half_w, half_h = hw * 2.0, hh * 3.0
        cx1, cy1 = ccx - half_w, ccy - half_h
        cx2, cy2 = ccx + half_w, ccy + half_h
    out = []
    for o in DENSE_OFFSETS:
        frame = targets[round(o, 3)]
        if frame is None:
            continue
        H, W = frame.shape[:2]
        box = None
        if cx1 is not None:
            ix1, iy1 = max(0, int(cx1)), max(0, int(cy1))
            ix2, iy2 = min(W, int(cx2)), min(H, int(cy2))
            if ix2 - ix1 < 60 or iy2 - iy1 < 60:
                ix1, iy1, ix2, iy2 = 0, 0, W, H
            frame = frame[iy1:iy2, ix1:ix2]
            if frame.size == 0:
                continue
            box = (int(hoop[0]) - ix1, int(hoop[1]) - iy1,
                   int(hoop[2]) - ix1, int(hoop[3]) - iy1)
        if box is not None:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                          (0, 255, 0), 2)
        h, w = frame.shape[:2]
        if w > MAX_W:
            frame = cv2.resize(frame, (MAX_W, int(h * MAX_W / w)))
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        if ok:
            out.append((o, base64.b64encode(buf).decode("ascii")))
    return out


def call_vlm(cfg, key, frames):
    content = [{"type": "text", "text": PROMPT}]
    for off, b64 in frames:
        content.append({"type": "text", "text": f"t={off:+.3f}s"})
        content.append({"type": "image_url", "image_url":
                        {"url": f"data:image/jpeg;base64,{b64}"}})
    payload = {
        "model": cfg["model"],
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
        **cfg["extra_payload"],
    }
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(cfg["url"], timeout=240,
                              headers={"Authorization": f"Bearer {key}"},
                              json=payload)
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"] or ""
                return parse_answer(text)
            if r.status_code in (400, 401, 403):
                return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(4 * (attempt + 1))
                continue
            return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
        except Exception as e:
            last_err = str(e)
            time.sleep(4 * (attempt + 1))
    return {"error": last_err}


def report(model_name, results):
    gray = [r for r in results if r["band"] == "gray"]
    ctrl = [r for r in results if r["band"] != "gray"]
    err = [r for r in results if r.get("vlm_error")]

    def acc(sub):
        ok = [r for r in sub if r.get("vlm_verdict") in ("goal", "miss")]
        if not ok:
            return 0.0, 0
        hit = sum(1 for r in ok
                  if (r["vlm_verdict"] == "goal") == (r["label"] == 1))
        return hit / len(ok), len(ok)

    print(f"\n=== {model_name} 加密抽帧版（有效 {len(results) - len(err)}/{len(results)}，"
          f"失败 {len(err)}）===")
    a, n = acc(gray)
    print(f"灰区（LGBM 不敢判的）: 准确率 {a:.1%}（n={n}）")
    for lab in (1, 0):
        sub = [r for r in gray if r["label"] == lab
               and r.get("vlm_verdict") in ("goal", "miss")]
        if sub:
            hit = sum(1 for r in sub
                      if (r["vlm_verdict"] == "goal") == (lab == 1))
            print(f"  {'真进球' if lab else '误报'}: {hit}/{len(sub)} 判对")
    if ctrl:
        a2, n2 = acc(ctrl)
        print(f"两端对照: 准确率 {a2:.1%}（n={n2}）")
    ex = [r for r in gray if 0.2 <= r["oof"] <= 0.8
          and r.get("vlm_verdict") in ("goal", "miss")]
    if ex:
        eh = sum(1 for r in ex if (r["vlm_verdict"] == "goal") == (r["label"] == 1))
        print(f"极灰区(0.2~0.8): {eh}/{len(ex)} = {eh/len(ex):.0%}")
    # 双判×
    dual = [r for r in results if r.get("vlm_verdict") == "miss" and r["oof"] < 0.5]
    if dual:
        dh = sum(1 for r in dual if r["label"] == 0)
        print(f"双判×精确率: {dh}/{len(dual)} = {dh/len(dual):.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "deepseek", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-gray", type=int, default=100)
    ap.add_argument("--n-ctrl", type=int, default=20)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rows = []
    for line in FEATURES_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows_by_id = {r["event_id"]: r for r in rows}
    scores = compute_oof()
    sample = build_sample(scores, rows_by_id, args.n_gray, args.n_ctrl)
    if args.limit:
        sample = sample[:args.limit]

    def get_key(cfg):
        for e in cfg["env_keys"]:
            if os.environ.get(e):
                return os.environ[e]
        return None

    models = [args.model] if args.model != "both" else ["qwen", "deepseek"]
    for mname in models:
        cfg = MODEL_CONFIGS[mname]
        key = get_key(cfg)
        if not key:
            print(f"[SKIP {mname}] 未提供 Key（环境变量: {' 或 '.join(cfg['env_keys'])}）")
            continue
        print(f"\n===== 评测 {mname} / {cfg['model']} 加密抽帧版"
              f" ({len(DENSE_OFFSETS)} 帧) =====")
        done_ids = set()
        results = []
        out_path = cfg["result_file"]
        if out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    done_ids.add(r["event_id"])
        todo = [e for e in sample if e["event_id"] not in done_ids]
        print(f"样本: {len(sample)}（断点 {len(done_ids)} 已完成，本次 {len(todo)}）")
        # 复用 resolve_video：build_sample 里的 video 是 resolve 过的完整路径，OK
        t0 = time.time()
        for i, ev in enumerate(todo):
            try:
                frames = extract_frames_dense(ev)
                if len(frames) < 10:
                    raise RuntimeError(f"抽帧不足({len(frames)})")
                ans = call_vlm(cfg, key, frames)
            except Exception as e:
                ans = {"error": str(e)}
            rec = {**{k: ev[k] for k in
                      ("event_id", "label", "oof", "band", "video", "ts")},
                   "frames": len(frames),
                   "vlm_verdict": ans.get("verdict"),
                   "vlm_confidence": ans.get("confidence"),
                   "vlm_reason": ans.get("reason"),
                   "vlm_error": ans.get("error")}
            results.append(rec)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0 or i == len(todo) - 1:
                dt = time.time() - t0
                print(f"  [{i + 1}/{len(todo)}] {dt:.0f}s "
                      f"({(i + 1) / max(dt, 1) * 60:.0f} ev/min)", flush=True)
        report(mname, results)
        print(f"结果: {out_path}")


if __name__ == "__main__":
    main()
