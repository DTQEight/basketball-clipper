# -*- coding: utf-8 -*-
"""真实录像体检：当前生产模型 vs 旧模型（缓存分） vs 人工标签。

用法：
    env\\python.exe training\\spot_check.py                  # 随机 3 场
    env\\python.exe training\\spot_check.py --n 10           # 随机 10 场
    env\\python.exe training\\spot_check.py --n 0            # 全部可选场次
    env\\python.exe training\\spot_check.py --seed 42        # 固定随机种子（复现同一批）
    env\\python.exe training\\spot_check.py 悍高 07.29       # 只测文件名含关键词的场次
    env\\python.exe training\\spot_check.py --list           # 只列可选场次
    env\\python.exe training\\spot_check.py --n 10 --out r.json

做什么：把 cache/clip_cache.json 里缓存的旧分数留一份，**清掉各臂分**后用当前生产模型
（services.goal_verifier.score_clips）重新打分，再与人工标签三方对比：
自动 √ 的精确率/召回、中间带大小、翻带明细、各臂 AUC（旧 vs 新）。

口径与副作用：
- **只改内存，不写 cache/clip_cache.json**；界面上仍显示旧分，重开记录时才重算
- ⚠ 必须清掉各臂分再打分：`_score_lgbm`/`_score_visual` 对已有分数的片段会直接跳过
  （`todo = [c for c in clips if "score_lgbm" not in c]`），不清就变成「旧分批 + 新口径融合」
- 人工标签取 cache/history，严格口径（只用 kept/deleted；未标注的候选不进精确率/召回）
- 旧分取缓存里的 score / score_lgbm / score_b / score_flow / score_vm（当时的生产口径）
"""
from __future__ import annotations

import argparse
import bisect
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import goal_verifier as gv  # noqa: E402
from services import state  # noqa: E402

CLIP_CACHE = ROOT / "cache" / "clip_cache.json"
ARM_KEYS = {"lgbm": "score_lgbm", "b": "score_b", "flow": "score_flow", "vm": "score_vm"}
STRIP = ("score", "score_lgbm", "score_b", "score_flow", "score_vm",
         "auto", "auto_reject", "verify_ver", "verify_score")


def auc(pairs) -> float:
    """pairs: [(score, label)]，秩和法（含并列）。"""
    pos = sorted(s for s, y in pairs if y == 1)
    neg = sorted(s for s, y in pairs if y == 0)
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for s in pos:
        tot += bisect.bisect_left(neg, s) + 0.5 * sum(1 for x in neg if x == s)
    return tot / (len(pos) * len(neg))


def tally(rows, keep, rej):
    """rows: [(ts, score, truth)]；truth ∈ {√, ×, ?}。"""
    scored = [(t, s, tr) for t, s, tr in rows if s is not None]
    a_k = [(t, s) for t, s, _ in scored if s >= keep]
    a_r = [(t, s) for t, s, _ in scored if s < rej]
    mid = [(t, s) for t, s, _ in scored if rej <= s < keep]
    tp = sum(1 for t, s, tr in scored if s >= keep and tr == "√")
    fp = sum(1 for t, s, tr in scored if s >= keep and tr == "×")
    real_kept = sum(1 for _, _, tr in scored if tr == "√")
    rec = tp / real_kept if real_kept else float("nan")
    return {"auto_k": len(a_k), "tp": tp, "fp": fp, "mid": len(mid),
            "auto_r": len(a_r), "prec": tp / len(a_k) if a_k else float("nan"),
            "rec": rec, "n_pos": real_kept, "n_scored": len(scored)}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("keywords", nargs="*", help="只测文件名含这些关键词的场次")
    ap.add_argument("--n", type=int, default=3, help="随机抽几场（0=全部）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="只列出可选场次")
    ap.add_argument("--out", type=str, default=None, help="另存 JSON 报告")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cache = json.loads(CLIP_CACHE.read_text(encoding="utf-8"))
    recs = {os.path.basename(r["video"]): r for r in state.load_history()}
    pool = []
    for e in cache:
        b = os.path.basename(e["video"])
        r = recs.get(b)
        if not r or not r.get("hoop") or not (e.get("clips") or []):
            continue
        if not any(c.get("score") is not None for c in e["clips"]):
            continue                       # 没有旧分可对比
        if args.keywords and not any(k in b for k in args.keywords):
            continue
        pool.append((b, e, r))
    pool.sort(key=lambda x: x[0])
    print(f"可选场次 {len(pool)}（有旧分数缓存 + 有人工标签）")
    if args.list:
        for b, e, r in pool:
            print(f"  {b:<32} 候选 {len(e['clips']):>3}")
        return

    picks = pool if args.n == 0 else random.sample(pool, min(args.n, len(pool)))
    keep_thr, rej_thr = gv.auto_threshold(), gv.reject_threshold()
    print(f"本次测 {len(picks)} 场 | keep_thr={keep_thr} reject_thr={rej_thr} "
          f"| 权重={gv.ENS_WEIGHTS}\n")

    report, summary = [], []
    for name, entry, rec in picks:
        video = rec["video"]
        hoop = [int(x) for x in rec["hoop"]]
        labels = state.get_labels(video)
        kept = {round(float(t), 3) for t in (labels.get("kept") or [])}
        deleted = {round(float(t), 3) for t in (labels.get("deleted") or [])}
        old_clips = copy.deepcopy(entry["clips"])
        new_clips = copy.deepcopy(entry["clips"])
        for c in new_clips:
            for k in STRIP:                # 关键：清掉各臂分，否则会被 todo 跳过
                c.pop(k, None)
        t0 = time.time()
        n_scored = gv.score_clips(video, new_clips, hoop)
        dt = time.time() - t0
        print(f"{'=' * 76}\n{name}\n  候选 {len(old_clips)}  人工 √{len(kept)} ×{len(deleted)}"
              f"  hoop={hoop}\n  重打分 {n_scored}/{len(new_clips)} 个，{dt:.0f}s")

        rows_old, rows_new = [], []
        for c_o, c_n in zip(old_clips, new_clips):
            ts = round(float(c_o["ts"]), 3)
            truth = "√" if ts in kept else ("×" if ts in deleted else "?")
            rows_old.append((ts, c_o.get("score"), truth))
            rows_new.append((ts, c_n.get("score"), truth))
        o = tally(rows_old, keep_thr, rej_thr)
        n = tally(rows_new, keep_thr, rej_thr)
        print(f"  {'':<8}{'自动√':>7}{'误报':>6}{'精确':>8}{'召回':>8}{'中间带':>8}{'自动×':>7}")
        for tag, r in (("旧模型", o), ("新模型", n)):
            print(f"  {tag:<8}{r['auto_k']:>7}{r['fp']:>6}{r['prec']:>8.3f}{r['rec']:>8.3f}"
                  f"{r['mid']:>8}{r['auto_r']:>7}")

        flips = [(t1, s1, s2, tr, "新进高带" if a2 else "退出高带")
                 for (t1, s1, tr), (t2, s2, tr2) in zip(rows_old, rows_new)
                 for a1, a2 in [(s1 is not None and s1 >= keep_thr,
                                 s2 is not None and s2 >= keep_thr)] if a1 != a2]
        if flips:
            print("  高带变动:")
            for t, s1, s2, tr, what in flips:
                print(f"    ts={t:>8.2f} 旧={s1} → 新={s2}  人工={tr}  {what}")

        arms = {}
        for arm, key in ARM_KEYS.items():
            a_old = [(float(c[key]), 1 if round(float(c["ts"]), 3) in kept else 0)
                     for c in old_clips if c.get(key) is not None
                     and round(float(c["ts"]), 3) in (kept | deleted)]
            a_new = [(float(c[key]), 1 if round(float(c["ts"]), 3) in kept else 0)
                     for c in new_clips if c.get(key) is not None
                     and round(float(c["ts"]), 3) in (kept | deleted)]
            arms[arm] = [round(auc(a_old), 4), round(auc(a_new), 4)]
        print(f"  各臂 AUC（严格人工口径，{len(kept) + len(deleted)} 个已标注候选）: "
              + "  ".join(f"{k} {v[0]}→{v[1]}" for k, v in arms.items()))
        summary.append((name, o, n))
        report.append({"video": name, "candidates": len(old_clips),
                       "kept": len(kept), "deleted": len(deleted),
                       "old": o, "new": n, "arms_auc": arms})

    print(f"\n{'=' * 76}\n汇总（{len(summary)} 场）")
    print(f"  {'视频':<28}{'旧自动√':>9}{'误报':>6}{'新自动√':>9}{'误报':>6}"
          f"{'旧中间带':>9}{'新中间带':>9}")
    for name, o, n in summary:
        print(f"  {name[:26]:<28}{o['auto_k']:>9}{o['fp']:>6}{n['auto_k']:>9}{n['fp']:>6}"
              f"{o['mid']:>9}{n['mid']:>9}")
    so = {k: sum(x[k] for _, x, _ in summary) for k in ("auto_k", "tp", "fp", "mid", "n_pos")}
    sn = {k: sum(x[k] for _, _, x in summary) for k in ("auto_k", "tp", "fp", "mid", "n_pos")}
    print(f"  合计: 旧 自动√{so['auto_k']}（对 {so['tp']} / 误报 {so['fp']}），"
          f"精确 {so['tp'] / max(so['auto_k'], 1):.3f}  召回 {so['tp'] / max(so['n_pos'], 1):.3f}"
          f"  中间带 {so['mid']}")
    print(f"        新 自动√{sn['auto_k']}（对 {sn['tp']} / 误报 {sn['fp']}），"
          f"精确 {sn['tp'] / max(sn['auto_k'], 1):.3f}  召回 {sn['tp'] / max(sn['n_pos'], 1):.3f}"
          f"  中间带 {sn['mid']}")
    print(f"  人工复核量（中间带）: {so['mid']} → {sn['mid']} "
          f"({(sn['mid'] - so['mid']) / max(so['mid'], 1) * 100:+.1f}%)")

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n已写出 {args.out}")


if __name__ == "__main__":
    main()
