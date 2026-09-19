# -*- coding: utf-8 -*-
"""A/B 轮同折对比：LGBM（手工特征）vs 时序模型（ResNet18 帧特征）+ 集成。

为什么需要这个脚本：
  A 轮和 B 轮的评估必须在同一套折上才能直接比。统一用
  train_temporal.build_folds_by_game（按比赛日分组，防同场多节/多副本泄漏），
  B 侧用与训练脚本一致的配置（增强变体只进训练折）。

评估对象：
  A_lgbm           A 轮 LightGBM（手工特征）
  B_pool_aug       B 轮 pool 结构 + 增强变体（中精度工作点强）
  B_bigru_aug      B 轮 BiGRU 结构 + 增强变体（高精度尾部稳）
  B_ensemble       两个 B 结构的 rank 平均
  AB_ensemble      A_lgbm 与 B_ensemble 的 rank 平均

用法：env\\python.exe training\\compare_ab.py
输出：training/compare_ab.json + training/oof_ab_compare.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.extract_frames_b import load_dataset_events  # noqa: E402
from training.train_temporal import (  # noqa: E402
    AUG_VARIANTS, FEAT_DIR, build_folds_by_game, load_feature_matrix,
    norm_game, report_metrics, train_b_oof)

TRAINING_DIR = PROJECT_ROOT / "training"
FEATURES_FILE = TRAINING_DIR / "features.jsonl"
REPORT_FILE = TRAINING_DIR / "compare_ab.json"
DETAIL_OUT = TRAINING_DIR / "oof_ab_compare.jsonl"

META_KEYS = {"event_id", "label", "video", "ts"}


def load_ab_data():
    """A 轮手工特征 ∩ B 轮帧特征事件。

    返回 (events, Xa, Xs_b, y, games, feat_keys, use_aug)。
    标签统一取 features.jsonl 的 int label，保证两模型同一套标签。
    """
    a_rows = {}
    with open(FEATURES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                a_rows[r["event_id"]] = r
    if not a_rows:
        print("ERROR: features.jsonl 为空")
        sys.exit(1)
    feat_keys = sorted(k for k in next(iter(a_rows.values())) if k not in META_KEYS)

    events = load_dataset_events()
    kept, Xa, y, vids = [], [], [], []
    miss_feat = miss_frames = 0
    for ev in events:
        eid = ev["event_id"]
        if eid not in a_rows:
            miss_feat += 1
            continue
        if not (FEAT_DIR / f"{eid}.npz").exists():
            miss_frames += 1
            continue
        kept.append(ev)
        Xa.append([float(a_rows[eid].get(k, 0.0)) for k in feat_keys])
        y.append(int(a_rows[eid]["label"]))
        vids.append(ev["video"])
    print(f"交集事件: {len(kept)}（A 缺特征 {miss_feat} / B 缺帧块 {miss_frames}）"
          f"  正 {int(sum(y))} / 负 {int(len(y) - sum(y))}")

    Xs_b = {"orig": load_feature_matrix(kept, "orig")}
    have_aug = all((FEAT_DIR / v / f"{e['event_id']}.npz").exists()
                   for v in AUG_VARIANTS for e in kept)
    if have_aug:
        for v in AUG_VARIANTS:
            Xs_b[v] = load_feature_matrix(kept, v)
    games = np.array([norm_game(v) for v in vids])
    return (kept, np.array(Xa, dtype=np.float32), Xs_b,
            np.array(y, dtype=np.float32), games, feat_keys, have_aug)


def train_lgbm_oof(Xa, y, games, folds):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    oof = np.zeros(len(y), dtype=np.float32)
    fold_aucs = []
    for k, val_games in enumerate(folds):
        va_mask = np.isin(games, list(val_games))
        tr, va = ~va_mask, va_mask
        n_pos, n_neg = int(y[tr].sum()), int((y[tr] == 0).sum())
        w = np.ones(int(tr.sum()))
        w[y[tr] == 1] = max(1.0, n_neg / max(n_pos, 1))
        clf = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, verbose=-1,
        )
        clf.fit(Xa[tr], y[tr], sample_weight=w)
        oof[va] = clf.predict_proba(Xa[va])[:, 1]
        try:
            auc = roc_auc_score(y[va], oof[va])
        except ValueError:
            auc = float("nan")
        fold_aucs.append(auc)
        print(f"  [LGBM] fold{k + 1}: val={int(va.sum())} AUC={auc:.3f}")
    return oof, fold_aucs


def rank_avg(*preds):
    """rank 平均集成：概率转秩后取均值，消除量纲差异。"""
    from scipy.stats import rankdata
    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    t0 = time.time()
    events, Xa, Xs_b, y, games, feat_keys, use_aug = load_ab_data()
    folds = build_folds_by_game([e["video"] for e in events])
    n_games = len(set(games.tolist()))
    print(f"视频 {len({e['video'] for e in events})} 个 / 比赛日 {n_games} 个"
          f" → {len(folds)} 折")
    print(f"B 侧增强变体: {'on' if use_aug else 'off（先跑 train_temporal.py 抽变体）'}\n")

    print("== A 轮：LightGBM（手工特征）==")
    oof_a, auc_a = train_lgbm_oof(Xa, y, games, folds)
    print("\n== B 轮：pool + 增强变体 ==")
    oof_bp, auc_bp, _ = train_b_oof(Xs_b, y, games, folds, arch="pool",
                                    use_aug=use_aug)
    print("\n== B 轮：BiGRU + 增强变体 ==")
    oof_bb, auc_bb, _ = train_b_oof(Xs_b, y, games, folds, arch="bigru",
                                    use_aug=use_aug)

    oof_b = rank_avg(oof_bp, oof_bb)      # B 侧两结构融合
    oof_ab = rank_avg(oof_a, oof_b)       # A + B 全融合

    print("\n===== 同折对比（OOF，按比赛日分组）=====")
    report = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_events": len(y), "n_pos": int(y.sum()), "n_games": n_games,
        "grouping": "by_game_day", "b_aug": use_aug,
        "fold_aucs_lgbm": [round(float(a), 4) for a in auc_a],
        "fold_aucs_b_pool": [round(a, 4) for a in auc_bp],
        "fold_aucs_b_bigru": [round(a, 4) for a in auc_bb],
    }
    m_a = report_metrics(y, oof_a, "A_lgbm")
    print()
    m_bp = report_metrics(y, oof_bp, "B_pool_aug")
    print()
    m_bb = report_metrics(y, oof_bb, "B_bigru_aug")
    print()
    m_b = report_metrics(y, oof_b, "B_ensemble(pool+bigru)")
    print()
    m_ab = report_metrics(y, oof_ab, "AB_ensemble")
    report.update({"A_lgbm": m_a, "B_pool_aug": m_bp, "B_bigru_aug": m_bb,
                   "B_ensemble": m_b, "AB_ensemble": m_ab})

    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    detail = [{"event_id": e["event_id"], "video": e["video"], "ts": e["ts"],
               "label": int(yy), "pred_lgbm": float(pa),
               "pred_pool": float(pp), "pred_bigru": float(pb),
               "pred_b_ens": float(pbe), "pred_ab_ens": float(pab)}
              for e, yy, pa, pp, pb, pbe, pab
              in zip(events, y, oof_a, oof_bp, oof_bb, oof_b, oof_ab)]
    DETAIL_OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in detail),
        encoding="utf-8")
    print(f"\n报告: {REPORT_FILE}")
    print(f"OOF 明细: {DETAIL_OUT}")
    print(f"耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
