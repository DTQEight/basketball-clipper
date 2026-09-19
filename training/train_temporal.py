# -*- coding: utf-8 -*-
"""B 轮训练：ResNet18 帧特征 + BiGRU 时序模型 + 离线增强变体（4GB 显存可跑）。

流程：
  1. 阶段一（预抽特征，GPU 一次性）：
     每个事件 16×224×224 帧块过 torchvision 预训练 ResNet18（512 维）写盘：
       frames_b_feat/{eid}.npz          原始
       frames_b_feat/vflip/{eid}.npz    水平翻转（穿筐物理左右对称，合法增强）
       frames_b_feat/dark/{eid}.npz     亮度 ×0.85 / 对比度 ×0.95
       frames_b_feat/bright/{eid}.npz   亮度 ×1.15 / 对比度 ×1.05
     训练阶段不再碰图像，只读 512 维特征。
  2. 阶段二（训练 + 评估）：帧特征 → Linear 投影 → BiGRU → 二分类。
     按比赛日分组 5 折（同一天的多节视频/多目录副本同折，防泄漏）；
     增强变体只进训练折，验证折永远用原始特征。
     报告 OOF AUC + 高精度工作点；全量重训后存 training/model_temporal.pt。

用法：
  python training/train_temporal.py                # 完整流程（缺特征先抽）+ 存模型
  python training/train_temporal.py --feat-only    # 只抽特征（含增强变体）
  python training/train_temporal.py --arch pool    # 消融：旧无序池化结构
  python training/train_temporal.py --no-aug       # 消融：关掉增强变体
  python training/train_temporal.py --blind-test   # 留出最近一个比赛日做盲测
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.extract_frames_b import (  # noqa: E402
    load_dataset_events, OUT_DIR, FRAME_OFFS, SIZE, ZOOM)

FEAT_DIR = PROJECT_ROOT / "training" / "frames_b_feat"
FEAT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT = PROJECT_ROOT / "training" / "model_temporal.pt"
META_OUT = PROJECT_ROOT / "training" / "model_temporal_meta.json"
OOF_OUT = PROJECT_ROOT / "training" / "oof_temporal.jsonl"
SUSPECTS_OUT = PROJECT_ROOT / "training" / "label_suspects.jsonl"
BLIND_OUT = PROJECT_ROOT / "training" / "blind_temporal.jsonl"

# 训练专用增强变体：水平翻转（穿筐物理左右对称）+ 明暗扰动
AUG_VARIANTS = ("vflip", "dark", "bright")


# ================= 比赛日归组（防泄漏分折） =================

_DATE_FULL = re.compile(r"(20\d{2})[.\-/ ]?(\d{2})[.\-/ ]?(\d{2})")
_DATE_SHORT = re.compile(r"(\d{2})[.\-/](\d{2})[.\-/](\d{2})")


def norm_game(video: str) -> str:
    """视频路径 → 比赛日 ID（同一场比赛的多节视频/多目录副本必须同折）。

    识别优先级：完整日期（2026.08.15 / 20260822）→ 补世纪的 typo 日期
    （'225.11.21' → 20251121）→ basename 去空白兜底。
    按视频路径分折会让同场比赛副本跨折泄漏（曾虚高 AUC ~0.015）。
    """
    b = Path(video).stem
    m = _DATE_FULL.search(b)
    if m:
        return "".join(m.groups())
    m = _DATE_SHORT.search(b)
    if m:
        return "20" + "".join(m.groups())
    return re.sub(r"\s+", "", b.lower())


FOLD_ASSIGN_FILE = PROJECT_ROOT / "training" / "fold_assign.json"


def build_folds_by_game(videos, n_folds=5, seed=42):
    """按比赛日分组分折（折数不超过比赛日数）。

    稳定映射：比赛日→折的归属持久化在 training/fold_assign.json，
    旧比赛日永不变折；新比赛日按视频数从大到小补进当前最空的折。
    （此前整表 shuffle 后轮询分配，每加一个比赛日全表重排，OOF 分布
    漂移导致线上 keep_thr 一轮跳 0.09：0.535→0.625。）
    """
    from collections import Counter
    games = sorted({norm_game(v) for v in videos})
    n = max(1, min(n_folds, len(games)))
    assign = {}
    if FOLD_ASSIGN_FILE.exists():
        try:
            assign = {g: int(f) for g, f in json.loads(
                FOLD_ASSIGN_FILE.read_text(encoding="utf-8")).items()
                if isinstance(f, int) and 0 <= f < n}
        except Exception:
            assign = {}
    new_games = [g for g in games if g not in assign]
    if new_games:
        vid_cnt = Counter()
        for v in videos:
            vid_cnt[norm_game(v)] += 1
        load = Counter(assign.values())
        for g in sorted(new_games, key=lambda g: (-vid_cnt[g], g)):
            f = min(range(n), key=lambda k: (load[k], k))
            assign[g] = f
            load[f] += vid_cnt[g]
        FOLD_ASSIGN_FILE.write_text(
            json.dumps(assign, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8")
    folds = [[] for _ in range(n)]
    for g in games:
        folds[assign[g]].append(g)
    return folds


# ================= 阶段一：ResNet18 帧特征（含增强变体） =================

def _brightness_contrast(x: np.ndarray, b: float, c: float) -> np.ndarray:
    f = x.astype(np.float32) * b
    m = f.mean(axis=(1, 2, 3), keepdims=True)
    f = (f - m) * c + m
    return np.clip(f, 0, 255).astype(np.uint8)


def _apply_variant(x: np.ndarray, variant: str) -> np.ndarray:
    """x: (16,224,224,3) uint8 BGR，返回增强副本。"""
    if variant == "vflip":
        return np.ascontiguousarray(x[:, :, ::-1, :])
    if variant == "dark":
        return _brightness_contrast(x, 0.85, 0.95)
    if variant == "bright":
        return _brightness_contrast(x, 1.15, 1.05)
    raise ValueError(f"unknown variant: {variant}")


def extract_frame_features(events, batch_frames=64, variants=("orig",)):
    """GPU 批量抽 512 维帧特征写盘（断点续跑，已存在的跳过）。"""
    import torchvision.models as tvm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()  # 去分类头，输出 512 维
    net = net.to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    for variant in variants:
        out_dir = FEAT_DIR if variant == "orig" else FEAT_DIR / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [(ev, OUT_DIR / f"{ev['event_id']}.npz") for ev in events
                if not (out_dir / f"{ev['event_id']}.npz").exists()]
        print(f"[{variant}] 待抽特征: {len(todo)}")
        if not todo:
            continue
        t0 = time.time()
        done = 0
        i = 0
        # batch_frames 是帧数预算（4GB 显存实测 64 帧安全），÷16 折算事件数
        ev_per_chunk = max(1, batch_frames // 16)
        while i < len(todo):
            chunk = todo[i:i + ev_per_chunk]
            imgs, owners = [], []
            for ev, path in chunk:
                if not path.exists():
                    continue
                x = np.load(path)["x"]  # (16,224,224,3) uint8 BGR
                if variant != "orig":
                    x = _apply_variant(x, variant)
                for f in x:
                    imgs.append(f)
                    owners.append(ev["event_id"])
            if not imgs:
                i += len(chunk)
                continue
            arr = np.stack(imgs).astype(np.float32) / 255.0
            arr = arr[..., ::-1]  # BGR→RGB
            arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))  # NCHW
            with torch.no_grad():
                t = torch.from_numpy(arr).to(device)
                t = (t - mean) / std
                feat = net(t)  # (N,512)
            feat = feat.cpu().numpy().astype(np.float16)
            by_ev = {}
            for eid, f in zip(owners, feat):
                by_ev.setdefault(eid, []).append(f)
            for eid, fl in by_ev.items():
                np.savez_compressed(out_dir / f"{eid}.npz", x=np.stack(fl))
                done += 1
            i += len(chunk)
            if done % 200 < ev_per_chunk:
                print(f"  [{done}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
        print(f"[{variant}] 完成：{done}")


def load_feature_matrix(events, variant="orig"):
    """按 events 顺序载入某变体的 (N,16,512) float32 矩阵。"""
    subdir = FEAT_DIR if variant == "orig" else FEAT_DIR / variant
    arrs = []
    for ev in events:
        p = subdir / f"{ev['event_id']}.npz"
        if not p.exists():
            raise FileNotFoundError(f"缺特征 {p}，先跑 --feat-only")
        arrs.append(np.load(p)["x"].astype(np.float32))
    return np.stack(arrs)


# ================= 阶段二：时序模型 =================

class TemporalNet(nn.Module):
    """帧特征 → 时序二分类。

    arch:
      bigru  双向 GRU（默认）：编码 16 帧顺序——球自上而下穿网的时序结构
      gru    单向 GRU
      pool   无序池化（mean/max/last；旧结构，消融对照）
    """

    def __init__(self, dim=512, hidden=128, arch="bigru"):
        super().__init__()
        self.arch = arch
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.2),
        )
        if arch == "bigru":
            self.rnn = nn.GRU(hidden, hidden // 2, batch_first=True,
                              bidirectional=True)
        elif arch == "gru":
            self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        elif arch == "pool":
            self.rnn = None
        else:
            raise ValueError(f"unknown arch: {arch}")
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: (B,16,512)
        h = self.proj(x)                       # (B,16,H)
        if self.rnn is not None:
            h, _ = self.rnn(h)                 # (B,16,H)
        pooled = torch.cat([h.mean(1), h.max(1).values, h[:, -1]], dim=-1)
        return self.head(pooled).squeeze(-1)


def train_b_oof(Xs, y, games, folds, arch="bigru", use_aug=True, epochs=60,
                seed=42, verbose=True):
    """按比赛日分组交叉验证训练时序模型。

    Xs: {"orig": (N,16,512)} + 可选增强变体；增强变体只进训练折。
    y: (N,) 0/1；games: (N,) 比赛日 ID。
    返回 (oof, fold_aucs, best_epochs)。
    """
    from sklearn.metrics import roc_auc_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    y = np.asarray(y, dtype=np.float32)
    games_arr = np.asarray(games)
    oof = np.zeros(len(y), dtype=np.float32)
    fold_aucs, best_epochs = [], []

    for k, val_games in enumerate(folds):
        va_mask = np.isin(games_arr, list(val_games))
        train_variants = ["orig"] + (list(AUG_VARIANTS) if use_aug else [])
        tr_X = np.concatenate([Xs[v][~va_mask] for v in train_variants], axis=0)
        tr_y = np.concatenate([y[~va_mask]] * len(train_variants))
        va_X = Xs["orig"][va_mask]

        torch.manual_seed(seed + k)
        net = TemporalNet(dim=Xs["orig"].shape[-1], arch=arch).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        pm = float(tr_y.mean())
        lossf = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(1 - pm) / max(pm, 1e-6)]).to(device))

        tr_X_t = torch.from_numpy(tr_X).to(device)
        tr_y_t = torch.from_numpy(tr_y).to(device)
        va_X_t = torch.from_numpy(va_X).to(device)

        best_auc, best_pred, best_ep = -1.0, None, 0
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(tr_X_t))
            for s in range(0, len(perm), 256):
                idx = perm[s:s + 256]
                opt.zero_grad()
                loss = lossf(net(tr_X_t[idx]), tr_y_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                p = torch.sigmoid(net(va_X_t)).cpu().numpy()
            try:
                auc = roc_auc_score(y[va_mask], p)
            except ValueError:
                auc = 0.5
            if auc > best_auc:
                best_auc, best_pred, best_ep = auc, p.copy(), ep + 1
        if best_pred is None:  # epochs=0 防御
            net.eval()
            with torch.no_grad():
                best_pred = torch.sigmoid(net(va_X_t)).cpu().numpy()
        oof[va_mask] = best_pred
        fold_aucs.append(float(best_auc))
        best_epochs.append(best_ep)
        if verbose:
            print(f"  fold{k + 1}: val={int(va_mask.sum())} AUC={best_auc:.3f} "
                  f"(best_ep={best_ep})")
    return oof, fold_aucs, best_epochs


def train_final_model(Xs, y, arch, use_aug, epochs, seed=42):
    """全量（含增强变体）重训最终模型。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_variants = ["orig"] + (list(AUG_VARIANTS) if use_aug else [])
    tr_X = np.concatenate([Xs[v] for v in train_variants], axis=0)
    tr_y = np.concatenate([np.asarray(y, dtype=np.float32)] * len(train_variants))

    torch.manual_seed(seed)
    net = TemporalNet(dim=Xs["orig"].shape[-1], arch=arch).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pm = float(tr_y.mean())
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(1 - pm) / max(pm, 1e-6)]).to(device))
    tr_X_t = torch.from_numpy(tr_X).to(device)
    tr_y_t = torch.from_numpy(tr_y).to(device)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(tr_X_t))
        for s in range(0, len(perm), 256):
            idx = perm[s:s + 256]
            opt.zero_grad()
            loss = lossf(net(tr_X_t[idx]), tr_y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
    net.eval()
    return net


def report_metrics(y, oof, title):
    """打印 OOF AUC + 高精度工作点，返回指标 dict。"""
    from sklearn.metrics import roc_auc_score

    from training.train_lgbm import eval_at_precision

    n_pos = max(int(np.asarray(y).sum()), 1)
    try:
        auc = roc_auc_score(y, oof)
    except ValueError:
        print(f"\n{title}: 无法计算 AUC（标签单一）")
        return {"oof_auc": None}
    print(f"\n{title}: OOF AUC = {auc:.3f}")
    res = {"oof_auc": round(float(auc), 4), "working_points": {}}
    for tp in (0.90, 0.95, 0.98):
        r = eval_at_precision(y, oof, tp)
        if r is None:
            print(f"  precision>={tp:.2f}: 不可达")
        else:
            th, p, rec = r
            print(f"  precision>={tp:.2f}: th={th:.3f} 实际精度={p:.3f} "
                  f"召回={rec:.3f} (自动确认 {int(rec * n_pos)}/{n_pos})")
            res["working_points"][str(tp)] = {
                "threshold": round(th, 4), "precision": round(p, 4),
                "recall": round(rec, 4)}
    pred05 = np.asarray(oof) >= 0.5
    y = np.asarray(y)
    tp_ = int((pred05 & (y == 1)).sum())
    fp_ = int((pred05 & (y == 0)).sum())
    prec = tp_ / max(tp_ + fp_, 1)
    rec = tp_ / n_pos
    print(f"  固定阈值 0.50: 精度={prec:.3f} 召回={rec:.3f} (FP={fp_})")
    res["fixed_050"] = {"precision": round(prec, 4), "recall": round(rec, 4)}
    return res


def run_blind_test(events, Xs, y, games, arch, use_aug, epochs):
    """留出最近一个比赛日做盲测：只在其余数据上 CV 定 epoch + 训练。"""
    dated = sorted({g for g in games.tolist() if g.isdigit()})
    if not dated:
        print("ERROR: 找不到带日期的比赛日，无法盲测")
        return
    blind_game = dated[-1]
    va_mask = games == blind_game
    blind_events = [e for e, m in zip(events, va_mask) if m]
    print(f"\n盲测比赛日: {blind_game}  事件 {int(va_mask.sum())} 个"
          f"（正 {int(y[va_mask].sum())} / 负 {int((1 - y[va_mask]).sum())}）")

    tr_events = [e for e, m in zip(events, ~va_mask) if m]
    Xs_tr = {v: Xs[v][~va_mask] for v in Xs}
    y_tr = y[~va_mask]
    games_tr = games[~va_mask]

    folds = build_folds_by_game([e["video"] for e in tr_events])
    oof, fold_aucs, best_eps = train_b_oof(Xs_tr, y_tr, games_tr, folds,
                                           arch=arch, use_aug=use_aug,
                                           epochs=epochs)
    report_metrics(y_tr, oof, f"[训练集 CV 参考] {arch}")

    final_epochs = int(np.median(best_eps))
    print(f"\n用 {final_epochs} epochs 在非盲测数据上训练最终模型...")
    net = train_final_model(Xs_tr, y_tr, arch, use_aug, final_epochs)
    device = next(net.parameters()).device
    with torch.no_grad():
        p = torch.sigmoid(net(torch.from_numpy(Xs["orig"][va_mask]).to(device))
                          ).cpu().numpy()
    report_metrics(y[va_mask], p, f"BLIND（{blind_game}）")
    detail = [{"event_id": e["event_id"], "video": e["video"], "ts": e["ts"],
               "label": int(yy), "pred": float(pp)}
              for e, yy, pp in zip(blind_events, y[va_mask], p)]
    BLIND_OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in detail),
        encoding="utf-8")
    print(f"盲测明细: {BLIND_OUT}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-only", action="store_true")
    ap.add_argument("--arch", default="bigru", choices=["bigru", "gru", "pool"])
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--blind-test", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    use_aug = not args.no_aug

    events = load_dataset_events()
    events = [e for e in events if (OUT_DIR / f"{e['event_id']}.npz").exists()]
    print(f"events with frames: {len(events)}")

    variants = ("orig",) + AUG_VARIANTS
    extract_frame_features(events, variants=variants)
    if args.feat_only:
        return

    Xs = {v: load_feature_matrix(events, v) for v in variants}
    y = np.array([1.0 if e["label"] == "pos" else 0.0 for e in events],
                 dtype=np.float32)
    games = np.array([norm_game(e["video"]) for e in events])
    n_games = len(set(games.tolist()))
    print(f"载入: {len(y)} 事件（正 {int(y.sum())} / 负 {int((1 - y).sum())}）"
          f"  比赛日 {n_games} 个")
    print(f"config: arch={args.arch} aug={'on' if use_aug else 'off'} "
          f"epochs={args.epochs}")

    if args.blind_test:
        run_blind_test(events, Xs, y, games, args.arch, use_aug, args.epochs)
        return

    folds = build_folds_by_game([e["video"] for e in events])
    print(f"→ {len(folds)} 折（按比赛日分组）")
    oof, fold_aucs, best_eps = train_b_oof(Xs, y, games, folds,
                                           arch=args.arch, use_aug=use_aug,
                                           epochs=args.epochs)
    metrics = report_metrics(y, oof, f"B_temporal[{args.arch}]")

    # OOF 明细 + 疑似误标清单（双高置信且与人工标签相反，供人工复核）
    detail = [{"event_id": e["event_id"], "video": e["video"], "ts": e["ts"],
               "label": int(yy), "pred": float(pp)}
              for e, yy, pp in zip(events, y, oof)]
    OOF_OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in detail),
        encoding="utf-8")
    suspects = [r for r in detail
                if (r["label"] == 1 and r["pred"] < 0.05)
                or (r["label"] == 0 and r["pred"] > 0.95)]
    if suspects:
        SUSPECTS_OUT.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in suspects),
            encoding="utf-8")
        print(f"疑似误标事件: {len(suspects)} → {SUSPECTS_OUT.name}（建议人工复核）")

    # 全量重训 + 存模型（仅正式配置存盘，消融运行不覆盖）
    if args.arch == "bigru" and use_aug:
        final_epochs = int(np.median(best_eps))
        print(f"\n全量重训（{final_epochs} epochs，各折最优 epoch 中位数）...")
        net = train_final_model(Xs, y, args.arch, use_aug, final_epochs)
        ckpt = {
            "state_dict": net.state_dict(),
            "arch": args.arch, "hidden": 128, "dim": 512,
            "frame_offsets": FRAME_OFFS, "zoom": ZOOM, "size": SIZE,
            "backbone": "torchvision resnet18 IMAGENET1K_V1 (fc→Identity)",
            "input": "BGR uint8 帧块 → RGB /255 → ImageNet mean/std",
            "aug_variants": AUG_VARIANTS,
        }
        torch.save(ckpt, MODEL_OUT)
        meta = {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "arch": args.arch, "use_aug": True,
            "n_events": len(y), "n_pos": int(y.sum()), "n_games": n_games,
            "fold_aucs": [round(a, 4) for a in fold_aucs],
            "final_epochs": final_epochs,
            **metrics,
        }
        META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"已保存: {MODEL_OUT}")
        print(f"已保存: {META_OUT}")
    else:
        print("\n（消融配置，不存模型）")


if __name__ == "__main__":
    main()
