# -*- coding: utf-8 -*-
"""DeepSeek-V4-Flash-Vision-Exp 灰区评测脚本（A/B 对照 qwen3-vl-plus）。

基于 vlm_eval.py 改：
  · base_url = https://api.deepseek.com （DeepSeek 官方 OpenAI 兼容）
  · model = deepseek-v4-flash-vision-exp
  · 关闭 thinking（关闭推理节省 token + 时间）
  · Prompt 保持完全一致（公平对比）
  · 抽帧与 qwen 评测脚本完全复用

用法：
  $env:VLM_KEY='sk-xxx' ; env\python.exe training\vlm_eval_ds.py --limit 3
  结果写入 training\vlm_eval_deepseek.jsonl（独立文件，不覆盖 qwen 结果）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vlm_eval import (compute_oof, build_sample, extract_frames,  # noqa: E402
                      parse_answer, PROMPT, FEATURES_FILE)

TRAINING_DIR = PROJECT_ROOT / "training"
RESULTS_FILE = TRAINING_DIR / "vlm_eval_deepseek.jsonl"

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash-vision-exp"


def call_vlm(key: str, model: str, frames) -> dict:
    content = [{"type": "text", "text": PROMPT}]
    for off, b64 in frames:
        content.append({"type": "text", "text": f"t={off:+.1f}s"})
        content.append({"type": "image_url", "image_url":
                        {"url": f"data:image/jpeg;base64,{b64}"}})
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 400,
        # DeepSeek V4 系列默认为思考模式（reasoning），视觉评测关推理
        # 一是省钱（思考 token 贵），二是结论要纯直觉判断（思考过长易幻觉）
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": content}],
    }
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(API_URL, timeout=240,
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


def report(results):
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

    print(f"\n=== DeepSeek V4-Flash-Vision-Exp 评测"
          f"（有效 {len(results) - len(err)}/{len(results)}，失败 {len(err)}）===")
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
        dis = [r for r in hi if r.get("vlm_verdict") == "miss"] + \
              [r for r in lo if r.get("vlm_verdict") == "goal"]
        if dis:
            print(f"  与 LGBM 两端判定相反: {len(dis)} 个：")
            for r in dis:
                print(f"    {r['event_id']} label={'真' if r['label'] else '假'}"
                      f" oof={r['oof']:.3f} vlm={r['vlm_verdict']}"
                      f" {r.get('vlm_reason', '')[:45]}")
    # 对比 qwen（若已有 vlm_eval_results.jsonl）
    qf = TRAINING_DIR / "vlm_eval_results.jsonl"
    if qf.exists():
        qwen = {json.loads(l)["event_id"]: json.loads(l)
                for l in qf.read_text(encoding="utf-8").splitlines() if l.strip()}
        same = [r for r in results if r["event_id"] in qwen
                and r.get("vlm_verdict") in ("goal", "miss")
                and qwen[r["event_id"]].get("vlm_verdict") in ("goal", "miss")]
        if same:
            print(f"\n=== 与 qwen3-vl-plus 同事件对比（n={len(same)}）===")
            ds_hit = sum(1 for r in same
                         if (r["vlm_verdict"] == "goal") == (r["label"] == 1))
            q_hit = sum(1 for r in same
                        if (qwen[r["event_id"]]["vlm_verdict"] == "goal") == (r["label"] == 1))
            print(f"  DeepSeek: {ds_hit}/{len(same)} = {ds_hit/len(same):.0%}")
            print(f"  Qwen   : {q_hit}/{len(same)} = {q_hit/len(same):.0%}")
            agree = sum(1 for r in same
                        if r["vlm_verdict"] == qwen[r["event_id"]]["vlm_verdict"])
            print(f"  两模型结论一致: {agree}/{len(same)} = {agree/len(same):.0%}")
            beat = [r for r in same
                    if (r["vlm_verdict"] == "goal") == (r["label"] == 1)
                    and (qwen[r["event_id"]]["vlm_verdict"] == "goal") != (r["label"] == 1)]
            lost = [r for r in same
                    if (qwen[r["event_id"]]["vlm_verdict"] == "goal") == (r["label"] == 1)
                    and (r["vlm_verdict"] == "goal") != (r["label"] == 1)]
            print(f"  DeepSeek 判对但 qwen 判错: {len(beat)} 个"
                  + ("" if not beat else f"（如 {beat[0]['event_id']} {beat[0].get('vlm_reason','')[:25]}）"))
            print(f"  qwen 判对但 DeepSeek 判错: {len(lost)} 个"
                  + ("" if not lost else f"（如 {lost[0]['event_id']} {lost[0].get('vlm_reason','')[:25]}）"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n-gray", type=int, default=100)
    ap.add_argument("--n-ctrl", type=int, default=20)
    ap.add_argument("--out", default=str(RESULTS_FILE))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    key = os.environ.get("VLM_KEY", "")
    if not key:
        print("ERROR: 缺少 VLM_KEY 环境变量（DeepSeek 官方 Key）")
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
    print(f"评测样本 {len(sample)} 个，模型 {args.model}")

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
    print(f"\n结果文件: {out_path}")


if __name__ == "__main__":
    main()
