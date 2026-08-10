"""检查标注质量：统计每帧框数分布，找出误检多框帧。"""
import collections
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
LBL = ROOT / "train_data_manual" / "labels"
cnt = collections.Counter()
total = 0
multi = []
for sub in ("train", "val"):
    d = LBL / sub
    if not d.exists():
        continue
    for f in d.glob("*.txt"):
        n = sum(1 for line in open(f, encoding="utf-8") if line.strip())
        cnt[n] += 1
        total += n
        if n > 3:
            multi.append((f.stem, sub, n))

print(f"总框数: {total}, 总帧数: {sum(cnt.values())}")
print("每帧框数分布:")
for k in sorted(cnt):
    print(f"  {k}个框: {cnt[k]}帧")

print(f"\n超过3个框的可疑帧({len(multi)}):")
for s, sub, n in sorted(multi, key=lambda x: -x[2]):
    print(f"  {s} ({sub}): {n}个框")
