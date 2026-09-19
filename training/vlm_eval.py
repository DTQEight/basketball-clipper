# -*- coding: utf-8 -*-
"""VLM 灰区仲裁评测：qwen3-vl-plus 对 LightGBM 灰区事件的判读 vs 人工标签。

流程：
  1. 重算 OOF 分数（按视频分组交叉验证，与 train_lgbm 一致），确定三档
  2. 抽样：灰区 0.05<score<0.905 为主 + 自动√/自动× 两端对照
  3. 每事件抽 16 帧画上篮筐标定框 → base64 → dashscope 兼容接口
  4. 对答案：灰区准确率 / 两端对照 / 与 LightGBM 意见分歧分析

用法（项目根，需 VLM_KEY 环境变量）：
  env\\python.exe training\\vlm_eval.py --limit 3        # 先试 3 个
  env\\python.exe training\\vlm_eval.py                 # 全量（灰区100+对照20）
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from video_io import VideoReader  # noqa: E402

TRAINING_DIR = PROJECT_ROOT / "training"
FEATURES_FILE = TRAINING_DIR / "features.jsonl"
RESULTS_FILE = TRAINING_DIR / "vlm_eval_results.jsonl"
OOF_CACHE = PROJECT_ROOT / "cache" / "oof_scores.json"

API_URL = ("https://dashscope.aliyuncs.com/compatible-mode/v1/"
           "chat/completions")
DEFAULT_MODEL = "qwen3-vl-plus"
VIDEO_ROOTS = [Path(r"D:\Downloads\test"), Path(r"D:\Downloads")]

# 抽帧方案：球穿筐约 0.2s，ts±0.5s 密采，两端稀采给上下文
TARGET_OFFSETS = [-1.5, -1.0, -0.6, -0.3,
                  -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                  1.0, 1.4]
MAX_W = 640
JPEG_Q = 80
KEEP_THR, REJECT_THR = 0.905, 0.05
SEED = 42


def resolve_video(path_str: str) -> str | None:
    p = Path(path_str)
    if p.exists():
        return str(p)
    name = p.name
    for root in VIDEO_ROOTS:
        cand = root / name
        if cand.exists():
            return str(cand)
    # 子目录递归找一次（如 D:\Downloads\test\2026.07.11\xxx.mp4）
    for root in VIDEO_ROOTS:
        if root.exists():
            try:
                for hit in root.rglob(name):
                    return str(hit)
            except OSError:
                continue
    return None


def compute_oof():
    """OOF 分数（与 train_lgbm 相同配置），带缓存。"""
    if OOF_CACHE.exists():
        cached = json.loads(OOF_CACHE.read_text(encoding="utf-8"))
        if cached.get("mtime") == FEATURES_FILE.stat().st_mtime:
            return cached["scores"]
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from train_lgbm import load_data, SEED

    rows, feat_keys, X, y, videos = load_data()
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y))
    for tr, va in gkf.split(X, y, groups=videos):
        w = np.ones(len(tr))
        w[y[tr] == 1] = max(1.0, n_neg / max(n_pos, 1))
        clf = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=SEED, verbose=-1)
        clf.fit(X[tr], y[tr], sample_weight=w)
        oof[va] = clf.predict_proba(X[va])[:, 1]
    scores = {rows[i]["event_id"]: round(float(oof[i]), 4)
              for i in range(len(rows))}
    OOF_CACHE.write_text(json.dumps(
        {"mtime": FEATURES_FILE.stat().st_mtime, "scores": scores}),
        encoding="utf-8")
    return scores


def build_sample(scores: dict, rows_by_id: dict, n_gray=100, n_ctrl=20):
    """灰区为主 + 两端对照；只抽视频在盘的事件。

    hoop 从 dataset_v1.json 取（features.jsonl 不存 hoop），
    无有效 hoop 的事件跳过（无法裁剪定位）。
    """
    rng = random.Random(SEED)
    ds = {d["event_id"]: d for d in json.loads(
        (TRAINING_DIR / "dataset_v1.json").read_text(encoding="utf-8"))}
    pool = []
    for eid, sc in scores.items():
        r = rows_by_id[eid]
        hoop = (ds.get(eid) or {}).get("hoop")
        if not hoop or len(hoop) != 4 or max(hoop) <= 0:
            continue
        vp = resolve_video(r["video"])
        if not vp:
            continue
        if REJECT_THR < sc < KEEP_THR:
            band = "gray"
        elif sc >= KEEP_THR:
            band = "auto_keep"
        else:
            band = "auto_reject"
        pool.append({"event_id": eid, "label": int(r["label"]),
                     "oof": sc, "band": band, "video": vp,
                     "ts": float(r["ts"]), "hoop": hoop})
    gray = [e for e in pool if e["band"] == "gray"]
    hi = [e for e in pool if e["band"] == "auto_keep"]
    lo = [e for e in pool if e["band"] == "auto_reject"]
    # 灰区按标签近似比例分层抽样
    gray_pos = [e for e in gray if e["label"] == 1]
    gray_neg = [e for e in gray if e["label"] == 0]
    n_pos = round(n_gray * len(gray_pos) / max(len(gray), 1))
    take = rng.sample(gray_pos, min(n_pos, len(gray_pos))) + \
        rng.sample(gray_neg, min(n_gray - n_pos, len(gray_neg)))
    rng.shuffle(take)
    ctrl = rng.sample(hi, min(n_ctrl // 2, len(hi))) + \
        rng.sample(lo, min(n_ctrl - n_ctrl // 2, len(lo)))
    return take + ctrl


def extract_frames(ev: dict, crop: bool = True) -> list[tuple[float, str]]:
    """返回 [(offset_sec, jpeg_b64), ...]。

    crop=True：以篮筐为中心裁剪放大（球在全景 640px 下太小，VLM 看不清），
    区域约筐宽×4 / 筐高×6，略偏上（捕捉球从上方进入）。
    """
    reader = VideoReader(ev["video"])
    fps, total = reader.fps, reader.total
    ts = ev["ts"]
    t0 = max(0.0, ts + TARGET_OFFSETS[0] - 0.05)
    t1 = min(total / fps, ts + TARGET_OFFSETS[-1] + 0.05)
    f0, f1 = int(t0 * fps), int(t1 * fps)
    targets = {round(o, 2): None for o in TARGET_OFFSETS}
    try:
        for fidx, frame in reader.iter_frames(start=f0, end=f1):
            t = fidx / fps
            for o in targets:
                if targets[o] is None and abs(t - ts - o) <= 0.5 / fps + 1e-6:
                    targets[o] = frame
    finally:
        reader.close()
    hoop = ev.get("hoop")
    # 裁剪区域（原坐标）
    cx1 = cy1 = cx2 = cy2 = None
    if crop and hoop and len(hoop) == 4:
        hx1, hy1, hx2, hy2 = [float(v) for v in hoop]
        hw, hh = hx2 - hx1, max(hy2 - hy1, 1)
        ccx, ccy = (hx1 + hx2) / 2, (hy1 + hy2) / 2 - hh * 0.5
        half_w, half_h = hw * 2.0, hh * 3.0
        cx1, cy1 = ccx - half_w, ccy - half_h
        cx2, cy2 = ccx + half_w, ccy + half_h
    out = []
    for o in TARGET_OFFSETS:
        frame = targets[round(o, 2)]
        if frame is None:
            continue
        H, W = frame.shape[:2]
        box = None
        if cx1 is not None:
            # 裁剪（clamp 到画面内）+ 篮筐框坐标平移
            ix1, iy1 = max(0, int(cx1)), max(0, int(cy1))
            ix2, iy2 = min(W, int(cx2)), min(H, int(cy2))
            if ix2 - ix1 < 60 or iy2 - iy1 < 60:
                ix1, iy1, ix2, iy2 = 0, 0, W, H  # 区域过小退化为全景
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


PROMPT = (
    "你是篮球录像裁判。下面是同一段固定机位（底角45度）比赛录像按时间顺序"
    "排列的帧，绿色方框是预先标定的篮筐位置。请判断：篮球是否从上方穿过"
    "篮筐入网得分。\n"
    "区分要点：真进球=球从筐口上方落入、穿过网下坠；误报=打铁弹出、从筐前/"
    "侧面飞过、擦网不进、碰板弹出、球只是经过筐附近。\n"
    '只输出 JSON：{"verdict":"goal 或 miss","confidence":0到100,"reason":"一句话"}'
)


def call_vlm(key: str, model: str, frames: list[tuple[float, str]]) -> dict:
    content = [{"type": "text", "text": PROMPT}]
    for off, b64 in frames:
        content.append({"type": "text", "text": f"t={off:+.1f}s"})
        content.append({"type": "image_url", "image_url":
                        {"url": f"data:image/jpeg;base64,{b64}"}})
    payload = {"model": model, "temperature": 0,
               "messages": [{"role": "user", "content": content}]}
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(API_URL, timeout=180,
                              headers={"Authorization": f"Bearer {key}"},
                              json=payload)
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"] or ""
                return parse_answer(text)
            if r.status_code in (429, 500, 502, 503):
                last_err = f"HTTP {r.status_code}: {r.text[:150]}"
                time.sleep(3 * (attempt + 1))
                continue
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            last_err = str(e)
            time.sleep(3 * (attempt + 1))
    return {"error": last_err}


def parse_answer(text: str) -> dict:
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[^{}]*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    low = text.lower()
    if "goal" in low and "miss" not in low:
        return {"verdict": "goal", "confidence": None, "reason": text[:80]}
    if "miss" in low:
        return {"verdict": "miss", "confidence": None, "reason": text[:80]}
    return {"error": f"无法解析: {text[:120]}"}


def report(results: list[dict]):
    gray = [r for r in results if r["band"] == "gray"]
    ctrl = [r for r in results if r["band"] != "gray"]
    err = [r for r in results if r.get("vlm_error")]

    def acc(sub):
        ok = [r for r in sub
              if r.get("vlm_verdict") in ("goal", "miss")]
        if not ok:
            return 0.0, 0
        hit = sum(1 for r in ok
                  if (r["vlm_verdict"] == "goal") == (r["label"] == 1))
        return hit / len(ok), len(ok)

    print(f"\n=== VLM 评测（有效 {len(results) - len(err)}/{len(results)}，"
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
        hi = [r for r in ctrl if r["band"] == "auto_keep"]
        lo = [r for r in ctrl if r["band"] == "auto_reject"]
        print(f"两端对照: 准确率 {a2:.1%}（n={n2}；"
              f"LGBM 自动√端 {len(hi)} / 自动×端 {len(lo)}）")
        # 分歧事件（两端上 VLM 与 LGBM 意见相反）最值得关注
        dis = [r for r in hi if r.get("vlm_verdict") == "miss"] + \
              [r for r in lo if r.get("vlm_verdict") == "goal"]
        if dis:
            print(f"  与 LGBM 两端判定相反: {len(dis)} 个：")
            for r in dis:
                print(f"    {r['event_id']} label={'真' if r['label'] else '假'}"
                      f" oof={r['oof']:.3f} vlm={r['vlm_verdict']}"
                      f" {r.get('vlm_reason', '')[:40]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-gray", type=int, default=100)
    ap.add_argument("--n-ctrl", type=int, default=20)
    ap.add_argument("--out", default=str(RESULTS_FILE),
                    help="结果 jsonl 路径（不同模型分开存）")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    key = os.environ.get("VLM_KEY", "")
    if not key:
        print("ERROR: 缺少 VLM_KEY 环境变量")
        sys.exit(1)

    rows = []
    for line in FEATURES_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows_by_id = {r["event_id"]: r for r in rows}
    scores = compute_oof()
    sample = build_sample(scores, rows_by_id, args.n_gray, args.n_ctrl)
    if args.limit:
        sample = sample[:args.limit]
    print(f"评测样本 {len(sample)} 个（灰区为主），模型 {args.model}")

    done_ids = set()
    results = []
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                results.append(r)
                done_ids.add(r["event_id"])
    todo = [e for e in sample if e["event_id"] not in done_ids]
    print(f"断点续跑：已完成 {len(done_ids)}，本次 {len(todo)}")

    t0 = time.time()
    for i, ev in enumerate(todo):
        try:
            frames = extract_frames(ev)
            if len(frames) < 8:
                raise RuntimeError(f"抽帧不足({len(frames)})")
            ans = call_vlm(key, args.model, frames)
        except Exception as e:
            ans = {"error": str(e)}
        rec = {**{k: ev[k] for k in
                  ("event_id", "label", "oof", "band", "video", "ts")},
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
    report(results)
    print(f"\n结果: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
