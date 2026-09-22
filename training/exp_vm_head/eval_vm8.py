# -*- coding: utf-8 -*-
"""未部署实验的复现脚本：对比「VM 臂换时序头前后」在 8 场真实视频上的表现。

结论（2026.09.19）：换头在 OOF 上更好但**真实视频上没有兑现**，故不部署。
详见 services/goal_verifier.py 的 VM 臂段与 doc/CHANGELOG.md 第八节。

数据源 vm8_scores.json：每行 [A, B, Flow, VM_old(LGBM), VM_new(时序头), is_goal]，
 由一次性脚本产出（对 8 场逐片段调用 goal_verifier._score_visual，同时保留换头前的
 score_vm 缓存值）——重跑需要在 goal_verifier 里把 exp_vm_head/model_vm_head.pt 接回
 VM 臂，故这里只保留产物与结论。

用法：env\\Scripts\\python.exe training\\exp_vm_head\\eval_vm8.py [权重a 权重b 权重flow 权重vm]
默认权重取线上现役 A.5 / B2 / Flow1 / VM1。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "vm8_scores.json").read_text(encoding="utf-8"))["data"]
CLEAN = ["2026.09.01-3rd.mp4", "2026.09.01-4th.mp4", "2026.09.03-3rd.mp4",
         "2026.08.31-1st.mp4"]
w = [float(x) for x in sys.argv[1:5]] or [0.5, 2.0, 1.0, 1.0]
print(f"权重 A={w[0]} B={w[1]} Flow={w[2]} VM={w[3]}\n")


def score(rows, vm_idx):
    """vm_idx 3=旧 LGBM 头，4=新时序头，5=两者取平均。"""
    def vm_of(r):
        if vm_idx == 5:
            return (r[3] + r[4]) / 2.0
        return r[vm_idx]
    return [(w[0] * r[0] + w[1] * r[1] + w[2] * r[2] + w[3] * vm_of(r))
            / sum(w) for r in rows]


def subset(names, vm_idx):
    out = []
    for n in names:
        for s, r in zip(score(D[n]["rows"], vm_idx), D[n]["rows"]):
            out.append((s, r[5]))
    return out


print("=" * 104)
print("细阈值扫描：自动 √ 数 / 命中 / 误报 / 精度 / 召回")
print("=" * 104)
for tag, vm_idx in (("旧 VM 头（LGBM）", 3), ("新 VM 头（时序）", 4),
                    ("两头平均", 5)):
    print(f"\n[{tag}]")
    for label, names in (("8 场", list(D)), ("4 场干净", CLEAN)):
        rows = subset(names, vm_idx)
        y = [g for _, g in rows]
        s = [x for x, _ in rows]
        print(f"  {label:<10} AUC={roc_auc_score(y, s):.4f}  候选 {len(rows)}  "
              f"真进球 {sum(y)}")
        for th in [round(0.64 + 0.02 * i, 2) for i in range(9)]:
            sel = [g for x, g in rows if x >= th]
            tp = sum(sel)
            fp = len(sel) - tp
            print(f"      th={th:.2f}: 自动√ {len(sel):>3}  命中 {tp:>3}  误报 {fp:>2}  "
                  f"精度 {tp / max(len(sel),1):.3f}  召回 {tp / max(sum(y),1):.1%}")

print("\n" + "=" * 104)
print("逐场明细（th=0.68）")
print("=" * 104)
print(f"{'视频':<22}{'候选':>5}{'真球':>5} | {'旧VM 自动√/精度/召回':>28} | "
      f"{'新VM 自动√/精度/召回':>28}")
for n in D:
    rows = D[n]["rows"]
    y = [r[5] for r in rows]
    line = f"{n:<22}{len(rows):>5}{sum(y):>5} |"
    for vm_idx in (3, 4):
        sel = [r[5] for s, r in zip(score(rows, vm_idx), rows) if s >= 0.68]
        tp = sum(sel)
        line += (f" {len(sel):>4} {tp / max(len(sel),1):>6.3f} "
                 f"{tp / max(sum(y),1):>7.1%}    |")
    print(line)
