# -*- coding: utf-8 -*-
"""单场 AUC + 不同阈值下「自动通过」的精确率/召回率。

用法：
    env\\python.exe training\\eval_thresholds.py "Y:\\全场录像\\2026.09.01\\2026.09.01-3rd.mp4"

依赖 cache/clip_cache.json（含各片段 score）与 cache/history（人工标签）。
未打分的片段自动跳过，不会 KeyError。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import state  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

arg = sys.argv[1] if len(sys.argv) > 1 else None
if not arg:
    sys.exit("用法：env\\Scripts\\python.exe training\\eval_thresholds.py <视频完整路径 或 文件名>")

# 兼容只给文件名：PowerShell 传中文完整路径会因编码失真，按 basename 解析更可靠
data = json.loads((ROOT / "cache" / "clip_cache.json").read_text(encoding="utf-8"))
e = next((i for i in data if i["video"] == arg), None)
if e is None:
    e = next((i for i in data if Path(i["video"]).name == Path(arg).name), None)
if e is None:
    sys.exit(f"clip_cache 里没有这个视频：{arg}")
v = e["video"]
print(f"视频: {v}")
lab = state.get_labels(v)
kept, dele = state.label_sets(lab)   # 人工 ∪ 模型（见 state.label_sets）

y, s = [], []
skipped = 0
for c in e["clips"]:
    ts = round(float(c["ts"]), 3)
    if ts not in kept and ts not in dele:
        continue
    if c.get("score") is None:
        skipped += 1
        continue
    y.append(1 if ts in kept else 0)
    s.append(float(c["score"]))

print(f"已标注且有分数 {len(y)} 个（√ {sum(y)} / × {len(y) - sum(y)}）"
      + (f"  跳过未打分 {skipped} 个" if skipped else ""))
print("本场 AUC =", round(roc_auc_score(y, s), 4))

# 线上现役阈值作为锚点，避免硬编码漂移
from services import goal_verifier as gv  # noqa: E402
thr0 = gv._read_ensemble()
print(f"线上现役 keep_thr = {thr0}")
print(f"\n{'阈值':>7}{'自动√':>7}{'命中':>6}{'误报':>6}{'精确率':>9}{'召回':>9}")
for th in [round(0.58 + 0.02 * i, 2) for i in range(12)] + [thr0]:
    tp = sum(1 for a, b in zip(y, s) if b >= th and a == 1)
    fp = sum(1 for a, b in zip(y, s) if b >= th and a == 0)
    mark = "  ← 现役" if abs(th - thr0) < 1e-9 else ""
    print(f"{th:>7.2f}{tp + fp:>7}{tp:>6}{fp:>6}"
          f"{tp / max(tp + fp, 1):>9.3f}{tp / sum(y):>9.1%}{mark}")
