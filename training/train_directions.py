# -*- coding: utf-8 -*-
"""四方向统一训练与评测（与 compare_ab 完全同折同池，可直接对比）。

方向4  C-ball   球心裁剪帧块 → ResNet18 → TemporalNet
       C-glob   全局低分辨率流 → ResNet18 → TemporalNet
       C-cat    两路拼接(1024) → TemporalNet
方向3  Flow-T   筐心光流幅度序列 → ResNet18 → TemporalNet
       Flow-S   光流汇总统计 → LGBM
       Traj     球轨迹渲染图 → ResNet18 → LGBM
方向1  VM       VideoMAE 768 维 → LGBM
方向2  SimCLR   域内预训练编码器帧特征 → TemporalNet（特征就绪后自动纳入）

阶段：
  --feat    GPU 提 ResNet18 特征（orig + vflip/dark/bright 变体，口径同 B 轮）
  --train   所有可用方向 5 折同池对比 + 集成搜索 → compare_directions.json
  --blind   留出最近比赛日盲测（各方向在非盲数据训练→盲集评估）
            → compare_directions_blind.json
  --deploy  冠军集成部署产物：Flow_T/VM 终模型 + 线上口径阈值标定
            → model_flow_t.pt / model_vm_lgbm.txt / deploy_ensemble.json
            （阈值写进 model_temporal_meta.json 的 ensemble 段，线上直接读）

用法：env\\python.exe training\\train_directions.py --feat --train --blind
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TR = PROJECT_ROOT / "training"
FRAMES_C = TR / "frames_c"
FLOW_B = TR / "flow_b"
TRAJ_C = TR / "traj_c"
VM_FEAT = TR / "vm_feat"
SIMCLR_FEAT = TR / "frames_b_feat_simclr"

FEAT_DIRS = {
    "c_ball": TR / "feat_c_ball",
    "c_glob": TR / "feat_c_glob",
    "flow": TR / "feat_flow",
    "traj": TR / "feat_traj",
}
REPORT_FILE = TR / "compare_directions.json"
BLIND_REPORT = TR / "compare_directions_blind.json"
DETAIL_OUT = TR / "oof_directions.jsonl"


# ================= 各方向数据装载（返回 (T,224,224,3) uint8 BGR） =================

def load_c(kind):
    key = "x_ball" if kind == "ball" else "x_glob"

    def _load(eid, variant):
        p = FRAMES_C / f"{eid}.npz"
        if not p.exists():
            return None
        x = np.load(p)[key]
        if variant != "orig":
            from training.train_temporal import _apply_variant
            x = _apply_variant(x, variant)
        return x
    return _load


def load_flow_maps(eid, variant):
    p = FLOW_B / f"{eid}.npz"
    if not p.exists():
        return None
    mag = np.load(p)["mag"]                    # (15,224,224) uint8
    x = np.repeat(mag[..., None], 3, axis=-1)  # (15,224,224,3)
    if variant != "orig":
        from training.train_temporal import _apply_variant
        x = _apply_variant(x, variant)
    return x


def load_traj(eid, variant):
    p = TRAJ_C / f"{eid}.npz"
    if not p.exists():
        return None
    return np.load(p)["x"][None]               # (1,224,224,3)


def flow_static_features(eid):
    """光流汇总统计（方向3b 静态特征）。"""
    p = FLOW_B / f"{eid}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    mag = d["mag"].astype(np.float32) / 12.0   # 还原像素位移
    frame_mag = mag.mean(axis=(1, 2))
    peak_t = int(np.argmax(frame_mag))
    return [float(d["mag_mean"]) / 12.0, float(d["mag_max"]) / 12.0,
            float(d["mag_std"]) / 12.0,
            float(np.percentile(mag, 90)),
            float((mag > 2.5).mean()),
            float(peak_t / max(len(frame_mag) - 1, 1)),
            float(frame_mag[-3:].mean() / max(frame_mag[:3].mean(), 1e-3))]


# ================= 阶段一：ResNet18 特征提取（通用版） =================

def extract_resnet_features(loader, out_root, variants, batch_frames=64,
                            events=None):
    import torch
    import torchvision.models as tvm

    if events is None:
        from training.extract_frames_b import load_dataset_events
        events = load_dataset_events()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    for variant in variants:
        out_dir = out_root if variant == "orig" else out_root / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [e for e in events
                if loader(e["event_id"], variant) is not None
                and not (out_dir / f"{e['event_id']}.npz").exists()]
        print(f"  [{out_root.name}/{variant}] 待抽: {len(todo)}", flush=True)
        t0 = time.time()
        done = 0
        i = 0
        ev_per_chunk = max(1, batch_frames // 16)
        while i < len(todo):
            chunk = todo[i:i + ev_per_chunk]
            imgs, owners = [], []
            for ev in chunk:
                x = loader(ev["event_id"], variant)
                if x is None:
                    continue
                for f in x:
                    imgs.append(f)
                    owners.append(ev["event_id"])
            if imgs:
                arr = np.stack(imgs).astype(np.float32) / 255.0
                arr = np.ascontiguousarray(arr[..., ::-1].transpose(0, 3, 1, 2))
                with torch.no_grad():
                    t = (torch.from_numpy(arr).to(device) - mean) / std
                    feat = net(t)
                by_ev = {}
                for eid, f in zip(owners, feat.cpu().numpy().astype(np.float16)):
                    by_ev.setdefault(eid, []).append(f)
                for eid, fl in by_ev.items():
                    np.savez_compressed(out_dir / f"{eid}.npz", x=np.stack(fl))
                    done += 1
            i += len(chunk)
            if done and done % 400 < ev_per_chunk:
                print(f"    [{done}/{len(todo)}] {(time.time() - t0) / 60:.1f}min",
                      flush=True)
        print(f"  [{out_root.name}/{variant}] 完成 {done}")


def run_feat():
    from training.train_temporal import AUG_VARIANTS
    variants = ("orig",) + AUG_VARIANTS
    print("== 方向4：球心裁剪特征 ==", flush=True)
    extract_resnet_features(load_c("ball"), FEAT_DIRS["c_ball"], variants)
    print("== 方向4：全局流特征 ==", flush=True)
    extract_resnet_features(load_c("glob"), FEAT_DIRS["c_glob"], variants)
    print("== 方向3：光流序列特征 ==", flush=True)
    extract_resnet_features(load_flow_maps, FEAT_DIRS["flow"], variants)
    print("== 方向3：轨迹图特征 ==", flush=True)
    extract_resnet_features(load_traj, FEAT_DIRS["traj"], ("orig",))


# ================= 特征矩阵装载 =================

def load_feat_matrix(events, root, variant="orig"):
    sub = root if variant == "orig" else root / variant
    if not sub.exists():
        return None
    arrs = []
    for ev in events:
        p = sub / f"{ev['event_id']}.npz"
        if not p.exists():
            return None
        arrs.append(np.load(p)["x"].astype(np.float32))
    return np.stack(arrs)


def build_arms(events, use_aug=True):
    """构建各方向特征矩阵。返回:
    {name: ("temporal", Xs_dict) | ("static", X)}；特征不全的方向省略。
    """
    from training.train_temporal import AUG_VARIANTS
    arms = {}
    variants = ("orig",) + (AUG_VARIANTS if use_aug else ())

    for name, roots in [("C_ball", [FEAT_DIRS["c_ball"]]),
                        ("C_glob", [FEAT_DIRS["c_glob"]]),
                        ("C_cat", [FEAT_DIRS["c_ball"], FEAT_DIRS["c_glob"]])]:
        Xs = {}
        ok = True
        for v in variants:
            mats = [load_feat_matrix(events, r, v) for r in roots]
            if any(m is None for m in mats):
                ok = False
                break
            Xs[v] = np.concatenate(mats, axis=-1)
        if ok:
            arms[name] = ("temporal", Xs)

    Xs_f = {}
    ok = True
    for v in variants:
        m = load_feat_matrix(events, FEAT_DIRS["flow"], v)
        if m is None:
            ok = False
            break
        Xs_f[v] = m
    if ok:
        arms["Flow_T"] = ("temporal", Xs_f)

    fs = [flow_static_features(e["event_id"]) for e in events]
    if all(f is not None for f in fs):
        arms["Flow_S"] = ("static", np.array(fs, dtype=np.float32))

    tr = load_feat_matrix(events, FEAT_DIRS["traj"], "orig")
    if tr is not None:
        arms["Traj"] = ("static", tr[:, -1, :])

    vm = []
    ok = VM_FEAT.exists()
    if ok:
        for e in events:
            p = VM_FEAT / f"{e['event_id']}.npz"
            if not p.exists():
                ok = False
                break
            vm.append(np.load(p)["x"].astype(np.float32).ravel())
    if ok:
        arms["VM"] = ("static", np.stack(vm))

    Xs_s = {}
    ok = SIMCLR_FEAT.exists()
    if ok:
        for v in variants:
            m = load_feat_matrix(events, SIMCLR_FEAT, v)
            if m is None:
                ok = False
                break
            Xs_s[v] = m
    if ok:
        arms["SimCLR"] = ("temporal", Xs_s)
    return arms


# ================= 阶段二：训练与对比 =================

def train_directions():
    from sklearn.metrics import roc_auc_score
    from training.compare_ab import load_ab_data, train_lgbm_oof, rank_avg
    from training.train_temporal import report_metrics, train_b_oof

    t0 = time.time()
    events, Xa, Xs_b, y, games, feat_keys, use_aug = load_ab_data()
    from training.train_temporal import build_folds_by_game
    folds = build_folds_by_game([e["video"] for e in events])
    print(f"基池: {len(y)} 事件 / {len(set(games.tolist()))} 比赛日 / "
          f"{len(folds)} 折（与 compare_ab 一致）\n")

    oof_a, _ = train_lgbm_oof(Xa, y, games, folds)
    oof_bp, _, _ = train_b_oof(Xs_b, y, games, folds, arch="pool",
                               use_aug=use_aug, verbose=False)
    oof_bb, _, _ = train_b_oof(Xs_b, y, games, folds, arch="bigru",
                               use_aug=use_aug, verbose=False)
    oof_b = rank_avg(oof_bp, oof_bb)
    oof_ab = rank_avg(oof_a, oof_b)

    results = {"AB_baseline": report_metrics(y, oof_ab, "AB_baseline(参照)")}
    oofs = {"A": oof_a, "B": oof_b, "AB": oof_ab}

    arms = build_arms(events, use_aug)
    for name, spec in arms.items():
        kind, X = spec
        if kind == "temporal":
            print(f"\n== 方向 {name}（TemporalNet，dim={X['orig'].shape[-1]}）==")
            oof_x, fold_aucs, _ = train_b_oof(X, y, games, folds, arch="bigru",
                                               use_aug=use_aug)
        else:
            print(f"\n== 方向 {name}（LGBM，{X.shape[1]} 维）==")
            oof_x, fold_aucs = train_lgbm_oof(X, y, games, folds)
        m = report_metrics(y, oof_x, name)
        m["fold_aucs"] = [round(float(a), 4) for a in fold_aucs]
        results[name] = m
        oofs[name] = oof_x

    # ---- 集成搜索：两两对比 + 贪心前向（跳过拉低分数的臂）----
    print("\n===== 集成对比（rank 平均）=====")
    ens_results = {}
    new_keys = [k for k in oofs if k not in ("A", "B", "AB")]
    for k in new_keys:
        combo = rank_avg(oofs["AB"], oofs[k])
        auc = roc_auc_score(y, combo)
        ens_results[f"AB+{k}"] = float(auc)
        print(f"  AB+{k}: AUC={auc:.4f}")
    # 贪心：从 AB 出发，逐个加入能提升 AUC 的臂；提升不动即停
    # （避免无信息臂稀释集成——Flow_S 单臂 0.548 曾把 AB+all 拖到 0.942）
    selected, cur_auc = [], float(roc_auc_score(y, oofs["AB"]))
    pool = list(new_keys)
    while pool:
        best_k, best_auc = None, cur_auc
        for k in pool:
            combo = rank_avg(oofs["AB"], *[oofs[s] for s in selected], oofs[k])
            a = float(roc_auc_score(y, combo))
            if a > best_auc:
                best_k, best_auc = k, a
        if best_k is None:
            break
        selected.append(best_k)
        pool.remove(best_k)
        cur_auc = best_auc
        name = "AB+" + "+".join(selected)
        ens_results[name] = cur_auc
        combo = rank_avg(oofs["AB"], *[oofs[s] for s in selected])
        results[name] = report_metrics(y, combo, name)
        print(f"  {name}: AUC={cur_auc:.4f} ← 贪心采纳")
    best_ens = max(ens_results, key=ens_results.get) if ens_results else None
    if best_ens:
        print(f"\n最优集成: {best_ens} AUC={ens_results[best_ens]:.4f}"
              f"（基线 AB={roc_auc_score(y, oofs['AB']):.4f}）")

    report = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_events": int(len(y)), "n_pos": int(y.sum()),
        "grouping": "by_game_day",
        "arms": results,
        "ensembles": {k: round(float(v), 4) for k, v in ens_results.items()},
        "best_ensemble": best_ens,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    detail = []
    for i, e in enumerate(events):
        detail.append({"event_id": e["event_id"], "video": e["video"],
                       "ts": e["ts"], "label": int(y[i]),
                       **{f"pred_{k.lower()}": float(oofs[k][i]) for k in oofs}})
    DETAIL_OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in detail),
        encoding="utf-8")
    print(f"\n报告: {REPORT_FILE}\nOOF 明细: {DETAIL_OUT}\n"
          f"耗时 {(time.time() - t0) / 60:.1f}min")


# ================= 阶段三：盲测（留出最近比赛日） =================

def blind_eval_directions():
    """最近比赛日盲测：各方向只在非盲数据上训练，盲集直接评估。

    时序臂：非盲数据上 5 折 CV 定最优 epoch 中位数 → 全量非盲重训 → 盲集推理。
    静态臂：非盲训练 LGBM → 盲集 predict_proba。
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from training.compare_ab import load_ab_data, rank_avg
    from training.train_temporal import (build_folds_by_game, report_metrics,
                                         train_b_oof, train_final_model)

    events, Xa, Xs_b, y, games, feat_keys, use_aug = load_ab_data()
    dated = sorted({g for g in games.tolist() if g.isdigit()})
    blind_game = dated[-1]
    va = games == blind_game
    tr = ~va
    n_pos_blind = int(y[va].sum())
    print(f"盲测比赛日: {blind_game} | {int(va.sum())} 事件"
          f"（正 {n_pos_blind} / 负 {int((1 - y[va]).sum())}）")

    folds_tr = build_folds_by_game([e["video"] for e, m in zip(events, tr) if m])
    y_tr, games_tr = y[tr], games[tr]

    def blind_temporal(Xs):
        oof, _, best_eps = train_b_oof(
            {v: Xs[v][tr] for v in Xs}, y_tr, games_tr, folds_tr,
            arch="bigru", use_aug=use_aug, verbose=False)
        eps = max(5, int(np.median(best_eps)))
        net = train_final_model({v: Xs[v][tr] for v in Xs}, y_tr,
                                "bigru", use_aug, eps)
        import torch
        device = next(net.parameters()).device
        with torch.no_grad():
            p = torch.sigmoid(net(torch.from_numpy(Xs["orig"][va]).to(device))
                              ).cpu().numpy()
        return p, roc_auc_score(y_tr, oof)

    def blind_lgbm(X):
        n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
        w = np.ones(int(tr.sum()))
        w[y_tr == 1] = max(1.0, n_neg / max(n_pos, 1))
        clf = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, verbose=-1)
        clf.fit(X[tr], y_tr, sample_weight=w)
        return clf.predict_proba(X[va])[:, 1]

    preds, cv_ref = {}, {}
    # 基线臂
    preds["A"] = blind_lgbm(Xa)
    from training.train_temporal import train_b_oof as _tbo  # noqa
    p_pool, cv_pool = blind_temporal(Xs_b)   # 注：B 基线用 bigru 单臂近似
    preds["B"] = p_pool
    cv_ref["B"] = cv_pool
    preds["AB"] = rank_avg(preds["A"], preds["B"])

    arms = build_arms(events, use_aug)
    for name, spec in arms.items():
        kind, X = spec
        print(f"  盲测 {name}...", flush=True)
        if kind == "temporal":
            preds[name], cv_ref[name] = blind_temporal(X)
        else:
            preds[name] = blind_lgbm(X)

    print(f"\n===== 盲测结果（{blind_game}）=====")
    results = {}
    for name, p in preds.items():
        try:
            auc = roc_auc_score(y[va], p)
        except ValueError:
            auc = float("nan")
        m = report_metrics(y[va], p, f"BLIND {name}")
        m["blind_auc"] = round(float(auc), 4)
        if name in cv_ref:
            m["cv_auc_nonblind"] = round(float(cv_ref[name]), 4)
        results[name] = m
        print()

    # ---- 盲测集成（贪心前向，盲集上直接评估）----
    print("===== 盲测集成（贪心）=====")
    ens_results = {}
    new_keys = [k for k in preds if k not in ("A", "B", "AB")]
    for k in new_keys:
        a = roc_auc_score(y[va], rank_avg(preds["AB"], preds[k]))
        ens_results[f"AB+{k}"] = float(a)
        print(f"  AB+{k}: AUC={a:.4f}")
    selected, cur_auc = [], float(roc_auc_score(y[va], preds["AB"]))
    pool_keys = list(new_keys)
    best_combo_pred = preds["AB"]
    while pool_keys:
        best_k, best_auc, best_pred = None, cur_auc, None
        for k in pool_keys:
            combo = rank_avg(preds["AB"], *[preds[s] for s in selected], preds[k])
            a = float(roc_auc_score(y[va], combo))
            if a > best_auc:
                best_k, best_auc, best_pred = k, a, combo
        if best_k is None:
            break
        selected.append(best_k)
        pool_keys.remove(best_k)
        cur_auc, best_combo_pred = best_auc, best_pred
        name = "AB+" + "+".join(selected)
        ens_results[name] = cur_auc
        results[name] = report_metrics(y[va], best_combo_pred, f"BLIND {name}")
        results[name]["blind_auc"] = round(cur_auc, 4)
        print(f"  {name}: AUC={cur_auc:.4f} ← 贪心采纳")
        print()
    best_ens = max(ens_results, key=ens_results.get) if ens_results else None
    if best_ens:
        print(f"盲测最优集成: {best_ens} AUC={ens_results[best_ens]:.4f}")

    # 逐事件明细（供后续分析/复算集成）
    blind_idx = np.where(va)[0]
    detail = []
    for j, i in enumerate(blind_idx):
        detail.append({"event_id": events[i]["event_id"],
                       "video": events[i]["video"], "ts": events[i]["ts"],
                       "label": int(y[i]),
                       **{f"pred_{k.lower()}": float(preds[k][j]) for k in preds}})
    DETAIL_OUT.parent.joinpath("oof_directions_blind.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in detail),
        encoding="utf-8")

    report = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "blind_game": blind_game,
        "n_blind": int(va.sum()), "n_pos_blind": n_pos_blind,
        "n_train": int(tr.sum()),
        "arms": results,
        "ensembles": {k: round(float(v), 4) for k, v in ens_results.items()},
        "best_ensemble": best_ens,
    }
    BLIND_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"盲测报告: {BLIND_REPORT}")


# ================= 阶段四：冠军集成部署（终模型 + 线上阈值标定） =================

FLOW_DEPLOY = TR / "model_flow_t.pt"
VM_DEPLOY = TR / "model_vm_lgbm.txt"
DEPLOY_REPORT = TR / "deploy_ensemble.json"
META_FILE = TR / "model_temporal_meta.json"


def _wp_threshold(y, p, min_prec):
    """最低阈值使 precision>=min_prec（最大召回工作点）。"""
    for th in np.unique(np.round(p, 3)):
        sel = p >= th
        tp = int((sel & (y == 1)).sum())
        fp = int((sel & (y == 0)).sum())
        if tp > 0 and tp / (tp + fp) >= min_prec:
            return {"threshold": round(float(th), 3),
                    "precision": round(tp / (tp + fp), 4),
                    "recall": round(tp / max(int(y.sum()), 1), 4)}
    return None


def _reject_threshold(y, p, max_kill=0.02):
    """最高阈值（0.01 步进）使误杀率（真进球被自动×）<= max_kill。"""
    pos = y == 1
    best = 0.05
    for th in np.arange(0.05, 0.60, 0.01):
        if (p[pos] <= th).mean() <= max_kill:
            best = round(float(th), 2)
        else:
            break
    return best


def deploy_ensemble():
    """冠军集成（AB+Flow_T+VM）部署产物：终模型 + 线上阈值标定。

    与线上 goal_verifier 完全同构的组合口径（部署形态，单事件可算）：
      B = mean(pool, bigru)（线上是 sigmoid 均值，非评测用的 rank 平均）
      集成 = 各臂 sigmoid 概率加权均值（缺臂时权重重归一化）
    组合择优：等权 4 臂 vs AB 高半权（0.5/0.5/1/1），OOF AUC 高者上线。
    阈值沿用 KEEP/REJECT 机制在 OOF 分布重标：
      keep_thr = precision>=0.95 工作点（最大召回）
      reject_thr = 误杀<=2% 的最高阈值
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from training.compare_ab import load_ab_data, train_lgbm_oof
    from training.train_temporal import (build_folds_by_game, report_metrics,
                                         train_b_oof, train_final_model)
    from training.extract_frames_b import FRAME_OFFS, ZOOM, SIZE

    t0 = time.time()
    events, Xa, Xs_b, y, games, feat_keys, use_aug = load_ab_data()
    folds = build_folds_by_game([e["video"] for e in events])
    print(f"部署池: {len(y)} 事件 / {len(set(games.tolist()))} 比赛日\n", flush=True)

    print("== OOF 各臂（线上组合口径精确重算）==", flush=True)
    oof_a, _ = train_lgbm_oof(Xa, y, games, folds)
    oof_pool, _, _ = train_b_oof(Xs_b, y, games, folds, arch="pool",
                                 use_aug=use_aug, verbose=False)
    oof_bigru, _, _ = train_b_oof(Xs_b, y, games, folds, arch="bigru",
                                  use_aug=use_aug, verbose=False)
    oof_b = (oof_pool + oof_bigru) / 2.0      # 线上 B：双结构 sigmoid 均值
    oof_ab = (oof_a + oof_b) / 2.0            # 线上 AB

    arms = build_arms(events, use_aug)
    if "Flow_T" not in arms or "VM" not in arms:
        raise RuntimeError(f"冠军集成两臂特征不全: {list(arms)} "
                           f"（先跑 --feat 与 VM 特征抽取）")
    Xs_f = arms["Flow_T"][1]
    X_vm = arms["VM"][1]
    oof_f, fold_f, eps_f = train_b_oof(Xs_f, y, games, folds, arch="bigru",
                                       use_aug=use_aug, verbose=False)
    oof_v, _ = train_lgbm_oof(X_vm, y, games, folds)

    print("\n== 部署组合择优（各臂概率均值，与线上同构）==", flush=True)
    cand = {
        "mean4": ((oof_a + oof_b + oof_f + oof_v) / 4.0,
                  {"lgbm": 1.0, "b": 1.0, "flow": 1.0, "vm": 1.0}),
        "ab_f_vm": ((oof_ab + oof_f + oof_v) / 3.0,
                    {"lgbm": 0.5, "b": 0.5, "flow": 1.0, "vm": 1.0}),
    }
    scores = {k: float(roc_auc_score(y, p)) for k, (p, _) in cand.items()}
    for k, s in scores.items():
        print(f"  {k}: OOF AUC={s:.4f}")
    best_name = max(scores, key=scores.get)
    ens, weights = cand[best_name]
    print(f"  → 上线组合: {best_name}")

    keep_wp = _wp_threshold(y, ens, 0.95)
    reject_thr = _reject_threshold(y, ens, 0.02)
    keep_thr = keep_wp["threshold"]
    print(f"  keep_thr={keep_thr} (p95: prec={keep_wp['precision']} "
          f"recall={keep_wp['recall']})")
    print(f"  reject_thr={reject_thr} (误杀<=2%)")

    # ---- Flow_T 终模型（全量重训，epoch 取 OOF 各折最优中位数）----
    eps = max(5, int(np.median(eps_f)))
    print(f"\n== Flow_T 终模型（{eps} epochs 全量重训）==", flush=True)
    net = train_final_model(Xs_f, y, "bigru", use_aug, eps)
    import torch
    ckpt = {
        "state_dict": net.state_dict(),
        "arch": "bigru", "hidden": 128, "dim": int(Xs_f["orig"].shape[-1]),
        "frame_offsets": FRAME_OFFS, "zoom": ZOOM, "size": SIZE,
        "backbone": "torchvision resnet18 IMAGENET1K_V1 (fc→Identity)",
        "input": "筐心 16 帧块 → 灰度 → Farneback(0.5,3,21,3,5,1.2,0) "
                 "→ 幅度×12.0 clip uint8 (15,224,224) → 3ch 复制 → RGB /255 "
                 "→ ImageNet mean/std",
        "aug_variants": ("vflip", "dark", "bright"),
    }
    torch.save(ckpt, FLOW_DEPLOY)
    print(f"已保存: {FLOW_DEPLOY}")

    # ---- VM 终模型（全量加权 LGBM）----
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    w = np.ones(len(y))
    w[y == 1] = max(1.0, n_neg / max(n_pos, 1))
    clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, verbose=-1)
    clf.fit(X_vm, y, sample_weight=w)
    clf.booster_.save_model(str(VM_DEPLOY))
    # Booster.predict 对 binary 目标默认输出概率，与 predict_proba 数值一致
    chk = float(np.abs(clf.predict_proba(X_vm[:8])[:, 1]
                       - clf.booster_.predict(X_vm[:8])).max())
    print(f"已保存: {VM_DEPLOY}（Booster vs sklearn 概率差 {chk:.2e}）")

    # ---- 阈值/权重写入 model_temporal_meta.json 的 ensemble 段（线上读取）----
    meta = {}
    try:
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    meta["ensemble"] = {
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "composition": best_name,
        "weights": weights,
        "keep_thr": keep_thr,
        "reject_thr": reject_thr,
        "oof_auc": round(scores[best_name], 4),
        "n_events": int(len(y)),
        "keep_wp": keep_wp,
        "arms_oof_auc": {
            "A": round(float(roc_auc_score(y, oof_a)), 4),
            "B(mean pool+bigru)": round(float(roc_auc_score(y, oof_b)), 4),
            "Flow_T": round(float(roc_auc_score(y, oof_f)), 4),
            "VM": round(float(roc_auc_score(y, oof_v)), 4),
            "AB(mean)": round(float(roc_auc_score(y, oof_ab)), 4),
            "mean4": round(scores["mean4"], 4),
            "ab_f_vm": round(scores["ab_f_vm"], 4),
        },
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    report = {**meta["ensemble"],
              "fold_aucs_flow": [round(float(a), 4) for a in fold_f],
              "flow_final_epochs": eps,
              "reject_kill_cap": 0.02,
              "elapsed_sec": round(time.time() - t0, 1)}
    DEPLOY_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"已保存: {DEPLOY_REPORT}")
    print(f"标定完成，耗时 {(time.time() - t0) / 60:.1f}min")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--deploy", action="store_true")
    args = ap.parse_args()
    if not (args.feat or args.train or args.blind or args.deploy):
        args.feat = args.train = True
    if args.feat:
        run_feat()
    if args.train:
        train_directions()
    if args.blind:
        blind_eval_directions()
    if args.deploy:
        deploy_ensemble()


if __name__ == "__main__":
    main()
