# -*- coding: utf-8 -*-
"""重建训练集：把 cache/history 的**标注池**转成训练流水线要求的 training/dataset_v1.json。

为什么要刷新：dataset_v1.json 是 2026.09.18 的冻结快照（831 事件 / 21 场），而标注池
（cache/history 的 labels）已增长到 995 片段 / 25 场。且其中 4 场视频在快照之后被重新
检测过（09.04-2nd 从 59 个候选增至 100 个），这部分片段任何模型都没见过。

口径与旧版完全一致（下游脚本无需改动）：
    event_id = md5(视频路径)[:8] + "_" + f"{round(ts*fps):010d}"
    字段     = event_id / video / ts / label / hoop / video_fps / ...

留出集（不进训练集）：

- **当前策略：除 08.25-bba半场 那场跨机位样本外，全量投入训练**（用户决定）。手上其余
  已标注的比赛日（08.29 / 08.31 / 09.01 / 09.03 / 09.04 / 09.08 / 09.11 / 09.15）一律
  进训练集，换取约 +30% 的样本量；跨机位那场留在 `HOLDOUT_FILES` 里当泛化测试集。
- ⚠️ **拿不到测试日的窗口期**：这段时间里唯一的评估面是 **OOF**（GroupKFold 按比赛日
  分折，仍是日级无偏），但它看不到部署层（裁剪口径、口径指纹、三段分诊实现）——所以
  窗口期内**别做无法回退的大改**（换骨干/改集成方式），改了也没法验证。
- ✅ **测试日 = 下一场新录的比赛**：录完标完后，**务必把它的日期前缀加进
  `HOLDOUT_DAYS`**，否则它会随训练集一起被吃掉。纪律：整日留出、只跑一次、只报告、
  不参与任何调参。
- 机制保留：`HOLDOUT_DAYS` 按日期前缀整日排除（新跑完的分节自动排除），
  `HOLDOUT_FILES` 用于历史上的单场排除。08.29/08.31 曾在其中，现已按上述策略放开。

标签口径（按来源，见 services/state.py 的来源分流表）：

| 来源 | 正样本 | 负样本 |
|---|---|---|
| 人工 | kept（收） | deleted（收） |
| 模型 | auto_kept（默认收，标 `label_source=ui_auto`） | auto_rejected（**不收**） |

不收模型判的 ×：会形成"模型自己判×→自己学"的闭环，且某个被漏判的难例会永久
固化成负样本。模型自动 √ 默认收（正样本本就不够用，高带精度约 0.99），但打了
来源标——需要严格只用人工正样本时，按 `label_source == "ui_manual"` 过滤即可。
本次改动之前写入的记录只有 kept/deleted，无法回溯区分来源，一律记成 ui_manual（不猜）。

event_id 沿用策略：重新检测会让同一进球的时间戳位移 0.03~0.17 秒，帧号随之改变，
event_id 也就变了——若照搬就会白白重抽全部特征。故对同一视频按 REUSE_TOL（0.25 秒）
匹配旧 dataset 的事件并**沿用旧 event_id**（同分片的特征可直接复用）。
容差安全：检测侧 min_gap_sec=2.0，相邻进球至少隔 2 秒，不会误并。

用法：
    env\\Scripts\\python.exe training\\build_dataset.py --dry-run   # 只报 delta，不写文件
    env\\Scripts\\python.exe training\\build_dataset.py             # 覆盖写 dataset_v1.json
    env\\Scripts\\python.exe training\\build_dataset.py --from-blocks   # 旧的 blocks_index 口径
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TR = ROOT / "training"
BLOCKS = ROOT / "dataset_20260918" / "blocks_index.json"
OUT = TR / "dataset_v1.json"
REUSE_TOL = 0.25   # 秒：重检造成的亚帧位移容差（远小于 min_gap_sec=2.0）

# 整日留出：这天的所有分节都不进训练集（按日期前缀匹配，新跑完的分节自动排除）
# ⚠️ 当前为空：手上所有已标注的比赛日都投入训练（用户决定），测试日改用**未来新比赛**。
#    拿到新测试日后务必把它的日期前缀加进这里（例如 "2026.09.2x"），
#    否则它会随训练集一起被吃掉、再也当不了测试日。
HOLDOUT_DAYS = ()
# 单场留出（按文件名）。当前保留 08.25-bba半场 的**整场次（3 场全留）**：机位是中线/边线
# 交点斜正对篮筐，与全部训练数据（底线机位）不同域，是目前唯一的跨机位泛化测试集 ——
# 实测该场集成 AUC 0.967 但自动 √ 召回只有 12.5%（B 臂真球均分 0.205，场内 0.95 AUC 的
# 主力臂在此域崩掉）。三场同机位必须一起留，只留一场的话另两场会把同域样本带进训练集、
# 破坏无偏性。若该机位以后还有新录像，一并加到这里（或改用 HOLDOUT_DAYS 写 "VID_20260825" 前缀）。
# 除非用户明确决定扩域，否则不要把它们放进训练集。
HOLDOUT_FILES = ("VID_20260825_211134.mp4",
                 "VID_20260825_212842.mp4",
                 "VID_20260825_215353.mp4")


def is_holdout(name: str) -> bool:
    """该视频是否留出（不进训练集）：整日留出按**日期前缀**，其余按文件名。

    整日留出必须用前缀匹配——否则同日新跑完的分节会悄悄进池（08.31-2nd/-3rd 就
    差点这样），而"同一天的其他节在训练集里"正是留出集失去无偏性的原因。
    """
    return name.startswith(HOLDOUT_DAYS) or name in HOLDOUT_FILES


def from_blocks():
    items = json.loads(BLOCKS.read_text(encoding="utf-8"))
    out = []
    for it in items:
        out.append({"video": it["video"], "ts": float(it["ts"]),
                    "label": 1 if it["label"] == "kept" else 0,
                    "hoop": [int(x) for x in it["hoop"]], "fps": float(it["fps"])})
    print(f"数据源 blocks_index: {len(out)} 事件")
    return out


def from_history():
    """读 cache/history 的标注池：当前检测候选 × 标签（按来源打标）。

    正样本取「人工 kept ∪ 模型 auto_kept」，负样本只取「人工 deleted」：
    - 模型判 × 的 auto_rejected 刻意不读（会形成"模型自己判×→自己学"的闭环，
      且某个被漏判的难例会永久固化成负样本）
    - 模型自动 √ 的 auto_kept **默认收进来**（正样本本就不够用，且高带精度约
      0.99），但写 `label_source` 区分来源，随时可在下游按来源过滤/加权
    - 本次改动之前写入的历史记录只有 kept/deleted，无法回溯区分来源，一律记成
      ui_manual（不猜）
    """
    sys.path.insert(0, str(ROOT))
    from services import state
    out, skipped = [], []
    n_auto = 0
    for r in state.load_history():
        name = Path(r["video"]).name
        if is_holdout(name):
            skipped.append(name)
            continue
        lab = state.get_labels(r["video"])
        hoop, fps = r.get("hoop"), r.get("video_fps")
        if not hoop or not fps:
            print(f"[SKIP] {name}: 缺 hoop/fps")
            continue

        base = {"video": r["video"], "hoop": [int(x) for x in hoop],
                "fps": float(fps)}
        for t in (lab.get("kept") or []):
            out.append({**base, "ts": float(t), "label": 1,
                        "label_source": "ui_manual"})
        for t in (lab.get("auto_kept") or []):
            out.append({**base, "ts": float(t), "label": 1,
                        "label_source": "ui_auto"})
            n_auto += 1
        for t in (lab.get("deleted") or []):
            out.append({**base, "ts": float(t), "label": 0,
                        "label_source": "ui_manual"})
    print(f"数据源标注池: {len(out)} 事件（其中模型自动 √ {n_auto} 条，"
          f"source=ui_auto）；留出 {len(skipped)} 场: {', '.join(sorted(skipped))}")
    return out


def reuse_event_ids(items):
    """按 REUSE_TOL 把新事件对齐到旧 dataset 的 event_id（同分片特征可直接复用）。

    返回 (reused, fresh, flips)；items 就地写入 event_id 或 None（表示需新抽特征）。
    """
    import bisect
    for it in items:
        it["event_id"] = None
    if not OUT.exists():
        return 0, len(items), []
    idx = {}
    for r in json.loads(OUT.read_text(encoding="utf-8")):
        idx.setdefault(r["video"], []).append((float(r["ts"]), r["event_id"], r["label"]))
    for v in idx:
        idx[v].sort()
    reused, flips = 0, []
    for it in items:
        cand = idx.get(it["video"])
        if not cand:
            continue
        pts = [c[0] for c in cand]
        i = bisect.bisect_left(pts, it["ts"])
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(cand):
                d = abs(cand[j][0] - it["ts"])
                if d <= REUSE_TOL and (best is None or d < best[0]):
                    best = (d, cand[j])
        if best:
            it["event_id"] = best[1][1]
            if best[1][2] != it["label"]:
                flips.append((best[1][1], best[1][2], it["label"]))
            reused += 1
    return reused, len(items) - reused, flips


def to_rows(items):
    rows = []
    vhash = {}
    for it in items:
        v = it["video"]
        if v not in vhash:
            vhash[v] = hashlib.md5(v.encode("utf-8")).hexdigest()[:8]
        eid = it.get("event_id") or "%s_%010d" % (
            vhash[v], int(round(it["ts"] * it["fps"])))
        rows.append({
            "event_id": eid,
            "video": v, "ts": round(it["ts"], 3), "clip_path": None,
            "label": it["label"], "hoop": it["hoop"],
            "video_fps": it["fps"],
            "video_width": None, "video_height": None, "video_duration_sec": None,
            "yolo_confirmed_history": None, "yolo_rejected_history": None,
            "detect_time": None, "label_time": None,
            # 必须带上 from_history 标好的来源（ui_manual / ui_auto）——写死成 "ui"
            # 会丢掉来源，下游就无法按 label_source == "ui_manual" 过滤出「纯人工正
            # 样本」，模型自动 √ 会静默混进真值（本文件开头承诺的过滤口径失效）。
            # 取不到时按 ui_manual（与「不猜」的既有策略一致）。
            "label_source": it.get("label_source") or "ui_manual",
        })
    eids = [r["event_id"] for r in rows]
    dup = len(eids) - len(set(eids))
    if dup:
        print(f"  event_id 重复 {dup} 个（同视频同帧）→ 后者丢弃")
        seen, keep = set(), []
        for r in rows:
            if r["event_id"] in seen:
                continue
            seen.add(r["event_id"])
            keep.append(r)
        rows = keep
    return rows, vhash


def report_delta(rows):
    if not OUT.exists():
        return
    old = json.loads(OUT.read_text(encoding="utf-8"))
    o = {r["event_id"]: r for r in old}
    n = {r["event_id"]: r for r in rows}
    common, added, removed = set(o) & set(n), set(n) - set(o), set(o) - set(n)
    print(f"\n相对现有 dataset_v1.json:")
    print(f"  旧 {len(o)} → 新 {len(n)}   沿用 {len(common)}  新增 {len(added)}  移除 {len(removed)}")
    lab_flip = [k for k in common if o[k]["label"] != n[k]["label"]]
    ts_move = [k for k in common if abs(o[k]["ts"] - n[k]["ts"]) > 1e-6]
    if lab_flip:
        print(f"  标签变化 {len(lab_flip)} 个")
    if ts_move:
        print(f"  ts 微移 {len(ts_move)} 个（应为 0，event_id 含帧号）")
    by_v = {}
    for k in added:
        by_v.setdefault(Path(n[k]["video"]).name, [0, 0])[0] += 1
    for k in removed:
        by_v.setdefault(Path(o[k]["video"]).name, [0, 0])[1] += 1
    if by_v:
        print(f"  {'视频':<26}{'新增':>6}{'移除':>6}")
        for v in sorted(by_v):
            print(f"  {v:<26}{by_v[v][0]:>6}{by_v[v][1]:>6}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只报 delta，不写文件")
    ap.add_argument("--from-blocks", action="store_true",
                    help="用旧的 dataset_20260918/blocks_index.json（复现历史数字用）")
    args = ap.parse_args()

    items = from_blocks() if args.from_blocks else from_history()
    if args.from_blocks:
        rows, vhash = to_rows(items)
    else:
        reused, fresh, flips = reuse_event_ids(items)
        rows, vhash = to_rows(items)
        print(f"event_id 沿用 {reused} 个，需新抽特征 {fresh} 个")
        if flips:
            print(f"[注意] 沿用事件中有 {len(flips)} 个标签与旧记录不一致：{flips[:5]}")
    npos = sum(1 for r in rows if r["label"] == 1)
    print(f"→ 事件 {len(rows)}  正 {npos} / 负 {len(rows) - npos}  视频 {len(vhash)}")
    report_delta(rows)

    if args.dry_run:
        print("\n（--dry-run：未写文件）")
        return
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写出 {OUT}")


if __name__ == "__main__":
    main()
