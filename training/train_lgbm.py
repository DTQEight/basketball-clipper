# -*- coding: utf-8 -*-
"""L4 阶段 A：LightGBM 进球验证器训练。

输入 training/features.jsonl（extract_features.py 产出），
输出 training/model_lgbm.txt（树模型文件）+ training/model_meta.json。

评估原则：
  - 按视频分组交叉验证（GroupKFold）：同视频事件共享场景/篮筐/光照，
    随机切分同视频事件会同时进训练/验证集 → 指标虚高（数据泄漏）。
  - 关键指标是「高精度工作点的召回」：目标是自动确认进球（免人工），
    所以看 precision>=0.95 / 0.98 时能覆盖多少真进球。

用法：env\\python.exe training\\train_lgbm.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
FEATURES_FILE = TRAINING_DIR / "features.jsonl"
MODEL_FILE = TRAINING_DIR / "model_lgbm.txt"
META_FILE = TRAINING_DIR / "model_meta.json"

# 训练用特征列（排除元信息）
META_KEYS = {"event_id", "label", "video", "ts"}

# 交叉验证折数（视频数 ~20，5 折比较稳）
N_FOLDS = 5
SEED = 42


def load_data():
    rows = []
    with open(FEATURES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("ERROR: features.jsonl 为空，先运行 extract_features.py")
        sys.exit(1)
    feat_keys = sorted(k for k in rows[0] if k not in META_KEYS)
    X = np.array([[float(r.get(k, 0.0)) for k in feat_keys] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    videos = [r["video"] for r in rows]
    return rows, feat_keys, X, y, videos


def eval_at_precision(y_true, prob, target_p):
    """在 precision >= target_p 的最高召回工作点，返回 (threshold, precision, recall)。
    不存在满足条件的工作点时返回 None。"""
    order = np.argsort(-prob)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    n = np.arange(1, len(y_sorted) + 1)
    prec = tp / n
    rec = tp / max(y_true.sum(), 1)
    ok = np.where(prec >= target_p)[0]
    if len(ok) == 0:
        return None
    i = ok[-1]  # 满足精度的最大召回点
    th = (prob[order][i] + (prob[order][i + 1] if i + 1 < len(prob) else 0.0)) / 2
    return float(th), float(prec[i]), float(rec[i])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    rows, feat_keys, X, y, videos = load_data()
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    uniq_videos = sorted(set(videos))
    print(f"事件 {len(rows)}（正 {n_pos} / 负 {n_neg}）  视频 {len(uniq_videos)} 个  "
          f"特征 {len(feat_keys)} 维")

    # ===== 按视频分组的交叉验证（防泄漏的诚实指标）=====
    gkf = GroupKFold(n_splits=min(N_FOLDS, len(uniq_videos)))
    oof = np.zeros(len(y))  # out-of-fold 概率
    fold_aucs = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=videos)):
        # 少数类权重平衡（正负接近 1:1.35，影响不大但保持一致）
        w = np.ones(len(tr))
        w[y[tr] == 1] = max(1.0, n_neg / max(n_pos, 1))
        clf = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=SEED, verbose=-1,
        )
        clf.fit(X[tr], y[tr], sample_weight=w)
        oof[va] = clf.predict_proba(X[va])[:, 1]
        auc = roc_auc_score(y[va], oof[va])
        fold_aucs.append(auc)
        va_videos = sorted(set(np.array(videos)[va]))
        print(f"  fold{fold + 1}: train={len(tr)} val={len(va)} "
              f"({len(va_videos)} 视频) AUC={auc:.3f}")

    auc_all = roc_auc_score(y, oof)
    print(f"\nOOF AUC（全体）= {auc_all:.3f}   各折: "
          + " ".join(f"{a:.3f}" for a in fold_aucs))

    # ===== 高精度工作点（业务目标：自动确认免人工）=====
    print("\n高精度工作点（OOF）：")
    for tp in (0.90, 0.95, 0.98, 0.99):
        r = eval_at_precision(y, oof, tp)
        if r is None:
            print(f"  precision>={tp:.2f}: 不可达")
        else:
            th, p, rec = r
            print(f"  precision>={tp:.2f}: threshold={th:.3f}  "
                  f"实际精度={p:.3f}  召回={rec:.3f} "
                  f"(自动确认 {int(rec * n_pos)}/{n_pos} 个真进球)")

    # 固定阈值参考
    for th in (0.5, 0.7, 0.9):
        pred = oof >= th
        tp_ = int((pred & (y == 1)).sum())
        fp_ = int((pred & (y == 0)).sum())
        prec = tp_ / max(tp_ + fp_, 1)
        rec = tp_ / max(n_pos, 1)
        print(f"  固定阈值 {th:.2f}: 精度={prec:.3f} 召回={rec:.3f} "
              f"(FP={fp_}, 漏检={n_pos - tp_})")

    # ===== 全量训练最终模型 =====
    w = np.ones(len(y))
    w[y == 1] = max(1.0, n_neg / max(n_pos, 1))
    final = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=SEED, verbose=-1,
    )
    final.fit(X, y, sample_weight=w)
    final.booster_.save_model(str(MODEL_FILE))

    # 特征重要性
    imp = sorted(zip(feat_keys, final.feature_importances_),
                 key=lambda kv: -kv[1])
    print("\n特征重要性 Top 15：")
    for k, v in imp[:15]:
        print(f"  {k:<20} {int(v)}")

    # 推荐阈值：OOF 上 precision>=0.95 的工作点（业务甜点：自动确认免人工）；
    # 0.98 档通常只剩个位数事件，作为推荐没有实用价值，仅作展示
    th_rec = None
    for tp in (0.95, 0.90):
        r = eval_at_precision(y, oof, tp)
        if r is not None:
            th_rec = r[0]
            break
    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_events": len(rows), "n_pos": n_pos, "n_neg": n_neg,
        "n_videos": len(uniq_videos),
        "features": feat_keys,
        "oof_auc": round(float(auc_all), 4),
        "fold_aucs": [round(float(a), 4) for a in fold_aucs],
        "recommended_threshold": round(float(th_rec), 3) if th_rec is not None else 0.5,
        "model_file": MODEL_FILE.name,
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\n已保存: {MODEL_FILE}")
    print(f"已保存: {META_FILE}（推荐阈值 {meta['recommended_threshold']}）")


if __name__ == "__main__":
    main()
