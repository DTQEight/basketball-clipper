# -*- coding: utf-8 -*-
"""方向2：域内自监督预训练（SimCLR 式对比学习）。

动机：B 轮 ResNet18 用 ImageNet 冻结特征，与固定机位篮球场景存在域差
（盲测 AUC 0.89→0.78 的主因之一）。用项目自有的 35,712 张筐心裁剪
（frames_b，2232 事件 × 16 帧，不利用标签）继续预训练编码器，
学到本域视觉先验后再提特征训练时序模型。

预训练：ResNet18(ImageNet 初始化) + 投影头，双视图 NT-Xent，
增强 = 随机裁剪缩放 + 水平翻转 + 亮度对比度抖动（GPU 上实现，批量快）。
输出：simclr_resnet18.pt（backbone state_dict，fc→Identity 前的全部层）。

--feat 模式：用预训练编码器对 frames_b（含增强变体）提 512 维特征
→ frames_b_feat_simclr/，与 train_temporal 同口径，供后续训练对比。

--flow 模式（2026.09.19）：数据源换成 flow_b（光流幅度图，15 帧/事件），
   给 Flow 臂换骨干用。同一套 SimCLR 配方，只是看的图不一样——
   光流图是灰度运动边缘图，ImageNet 特征与它失配比彩色帧更严重。
   产物：flow_simclr_resnet18.pt / flow_feat_simclr/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 两个数据源共用同一套 SimCLR 配方，只有「看的图」和产物路径不同
SOURCES = {
    "b": {
        "dir": PROJECT_ROOT / "training" / "frames_b",
        "key": "x",              # npz 里的键
        "frames": 16,            # 每事件帧数
        "memmap": PROJECT_ROOT / "training" / "crops_all.npy",
        "ckpt": PROJECT_ROOT / "training" / "simclr_resnet18.pt",
        "feat": PROJECT_ROOT / "training" / "frames_b_feat_simclr",
        "desc": "筐心彩色裁剪（BGR uint8）",
    },
    "flow": {
        "dir": PROJECT_ROOT / "training" / "flow_b",
        "key": "mag",            # (15,224,224) 单通道幅度图
        "frames": 15,
        "memmap": PROJECT_ROOT / "training" / "flow_crops_all.npy",
        "ckpt": PROJECT_ROOT / "training" / "flow_simclr_resnet18.pt",
        "feat": PROJECT_ROOT / "training" / "flow_feat_simclr",
        "desc": "光流幅度图（Farneback，单通道复制成 3 通道）",
    },
}
CFG = SOURCES["b"]              # 由 main() 按 --flow 选定，函数内统一读它

TEMP = 0.5
BATCH_FILES = 8      # 每个事件文件 16 帧 → 8 文件 = 128 crops/batch
PROJ_DIM = 128


def load_frames(path: Path) -> np.ndarray:
    """读一个事件 npz → (T,224,224,3) uint8。

    光流幅度图是单通道，复制成 3 通道以复用同一套 ResNet18 入口
    （与服务端 goal_verifier._flow_maps 的口径一致）。
    """
    x = np.load(path)[CFG["key"]]
    return np.repeat(x[..., None], 3, axis=-1) if x.ndim == 3 else x



class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        base = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, PROJ_DIM))

    def forward(self, x):
        h = self.backbone(x)
        z = F.normalize(self.proj(h), dim=-1)
        return h, z


def make_views(imgs: torch.Tensor) -> torch.Tensor:
    """imgs: (B,3,224,224) float[0,1] on GPU → 增强视图（同形状）。

    全向量化（替代旧逐样本循环，提速 ~50×）：
    随机仿射裁剪缩放（等效面积 55%~100% 的随机框）+ 随机翻转 + 亮度/对比度抖动。
    """
    B, _, H, W = imgs.shape
    device = imgs.device
    # 随机缩放（等效裁剪面积 0.55~1.0 → 线性尺度 0.74~1.0）+ 随机平移
    s = torch.empty(B, device=device).uniform_(0.74, 1.0)
    # 平移限幅：保证采样窗基本不越界（|t| <= 1-s，留少量余量给反射填充）
    tx = (torch.rand(B, device=device) - 0.5) * 2 * (1 - s) * 0.95
    ty = (torch.rand(B, device=device) - 0.5) * 2 * (1 - s) * 0.95
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, imgs.size(), align_corners=False)
    out = F.grid_sample(imgs, grid, mode="bilinear",
                        padding_mode="reflection", align_corners=False)
    # 随机水平翻转（向量化）
    flip = (torch.rand(B, 1, 1, 1, device=device) < 0.5)
    out = torch.where(flip, out.flip(-1), out)
    # 亮度
    out = out * torch.empty(B, 1, 1, 1, device=device).uniform_(0.75, 1.25)
    # 对比度（对每样本均值）
    m = out.mean(dim=(2, 3), keepdim=True)
    out = (out - m) * torch.empty(B, 1, 1, 1, device=device).uniform_(0.85, 1.15) + m
    return out.clamp(0, 1)


def nt_xent(z1, z2, temp=TEMP):
    """NT-Xent：z1/z2 (B,D) 已归一化。"""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                    # (2B,D)
    sim = z @ z.T / temp                              # (2B,2B)
    # 去掉自身对角
    mask_self = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask_self, -1e9)
    # 正样本：i 与 i+B
    pos = torch.cat([
        (z1 * z2).sum(-1) / temp, (z2 * z1).sum(-1) / temp])  # (2B,)
    logsum = torch.logsumexp(sim, dim=-1)
    return -(pos - logsum).mean()


def ensure_crops_memmap(files):
    """一次性把全部帧块拼成单个未压缩 npy（内存映射读）。

    每步 8 个 npz 的 zlib 解压实测 524ms，是预训练最大瓶颈；
    换成单文件后每步只做十几 MB 随机读（几毫秒级）。
    """
    per = CFG["frames"]
    n = len(files) * per
    mm_path = CFG["memmap"]
    if mm_path.exists():
        try:
            mm = np.load(mm_path, mmap_mode="r")
            if mm.shape[0] == n:
                print(f"crops 内存映射已存在: {n} crops", flush=True)
                return mm
        except Exception:
            pass
    print(f"构建 crops 内存映射（{n} crops，~{n * 150528 / 1e9:.1f}GB，一次性）...",
          flush=True)
    t0 = time.time()
    mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.uint8,
                                   shape=(n, 224, 224, 3))
    for i, f in enumerate(files):
        mm[i * per:(i + 1) * per] = load_frames(f)
        if (i + 1) % 400 == 0:
            print(f"  [{i + 1}/{len(files)}] {(time.time() - t0) / 60:.1f}min",
                  flush=True)
    mm.flush()
    del mm
    print(f"内存映射就绪，耗时 {(time.time() - t0) / 60:.1f}min", flush=True)
    return np.load(mm_path, mmap_mode="r")


def pretrain(epochs=30, lr=3e-4, seed=42, batch_files=BATCH_FILES):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    files = sorted(CFG["dir"].glob("*.npz"))
    per = CFG["frames"]
    crops = ensure_crops_memmap(files)
    n_files = len(files)
    print(f"源={CFG['desc']}  事件文件 {n_files} 个（{n_files * per} crops）"
          f" batch={batch_files} 文件", flush=True)

    model = SimCLRModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t0 = time.time()
    for ep in range(epochs):
        rng = np.random.RandomState(seed + ep)
        order = np.arange(n_files)
        rng.shuffle(order)
        losses = []
        for s in range(0, n_files - batch_files + 1, batch_files):
            idx = order[s:s + batch_files]
            # 每事件隔帧抽一半（16→8 / 15→8）：对比学习不需全部帧，单步计算减半
            imgs = np.concatenate(
                [np.asarray(crops[f * per:(f + 1) * per])[::2] for f in idx],
                axis=0)
            x = torch.from_numpy(
                np.ascontiguousarray(imgs[..., ::-1].transpose(0, 3, 1, 2))
            ).float().to(device) / 255.0
            v1, v2 = make_views(x), make_views(x)
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = nt_xent(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        print(f"epoch {ep + 1}/{epochs}: loss={np.mean(losses):.3f} "
              f"({(time.time() - t0) / 60:.1f}min)", flush=True)

    torch.save({"backbone": model.backbone.state_dict(),
                "epochs": epochs, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pool": f"{len(files)} events x {per} crops (unlabeled)",
                "source": CFG["desc"]},
               CFG["ckpt"])
    print(f"已保存编码器: {CFG['ckpt']}")


def extract_features(batch_frames=64):
    """用 SimCLR 编码器提 512 维特征（口径同 train_temporal.extract_frame_features）。"""
    from training.train_temporal import _apply_variant, AUG_VARIANTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    per = CFG["frames"]
    ck = torch.load(CFG["ckpt"], map_location=device, weights_only=False)
    import torchvision.models as tvm
    net = tvm.resnet18(weights=None)
    net.fc = nn.Identity()
    net.load_state_dict(ck["backbone"])
    net = net.to(device).eval()
    print(f"SimCLR 编码器已加载（{ck.get('trained_at')}，{ck.get('pool')}）")

    from training.extract_frames_b import load_dataset_events
    events = load_dataset_events()
    feat_root = CFG["feat"]

    for variant in ("orig",) + AUG_VARIANTS:
        out_dir = feat_root if variant == "orig" else feat_root / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [(ev, CFG["dir"] / f"{ev['event_id']}.npz") for ev in events
                if (CFG["dir"] / f"{ev['event_id']}.npz").exists()
                and not (out_dir / f"{ev['event_id']}.npz").exists()]
        print(f"[{variant}] 待抽: {len(todo)}", flush=True)
        t0 = time.time()
        done = 0
        i = 0
        ev_per_chunk = max(1, batch_frames // per)
        while i < len(todo):
            chunk = todo[i:i + ev_per_chunk]
            imgs, owners = [], []
            for ev, path in chunk:
                x = load_frames(path)
                if variant != "orig":
                    x = _apply_variant(x, variant)
                for f in x:
                    imgs.append(f)
                    owners.append(ev["event_id"])
            if imgs:
                # 口径必须与 pretrain() 一致：BGR→RGB + /255，**不做** ImageNet
                # mean/std（编码器是在裸 /255 上自监督训出来的）。
                # 旧实现漏了 [..., ::-1]，喂进去的是 BGR 而编码器学的是 RGB，
                # 通道序错位 → 特征与编码器不匹配。
                # （光流图三通道相同，该反转是空操作，保留只为两源共用一条代码路径）
                arr = np.stack(imgs).astype(np.float32)[..., ::-1] / 255.0
                arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
                with torch.no_grad():
                    feat = net(torch.from_numpy(arr).to(device))
                by_ev = {}
                for eid, f in zip(owners, feat.cpu().numpy().astype(np.float16)):
                    by_ev.setdefault(eid, []).append(f)
                for eid, fl in by_ev.items():
                    np.savez_compressed(out_dir / f"{eid}.npz", x=np.stack(fl))
                    done += 1
            i += len(chunk)
            if done and done % 400 < ev_per_chunk:
                print(f"  [{variant} {done}/{len(todo)}] "
                      f"{(time.time() - t0) / 60:.1f}min", flush=True)
        print(f"[{variant}] 完成 {done}，{(time.time() - t0) / 60:.1f}min")


def main():
    global CFG
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", action="store_true", help="跳过预训练，只提特征")
    ap.add_argument("--flow", action="store_true",
                    help="数据源换成 flow_b（光流幅度图），给 Flow 臂换骨干")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-files", type=int, default=BATCH_FILES,
                    help="每步事件数（每事件隔帧抽一半，故图像数 = 4×该值）")
    args = ap.parse_args()
    if args.flow:
        CFG = SOURCES["flow"]
    print(f"数据源: {CFG['desc']}  ({CFG['dir'].name}，{CFG['frames']} 帧/事件)")
    if not args.feat:
        pretrain(epochs=args.epochs, batch_files=args.batch_files)
    extract_features()


if __name__ == "__main__":
    main()
