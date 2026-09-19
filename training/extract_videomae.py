# -*- coding: utf-8 -*-
"""方向1：VideoMAE v2 视频基础模型提特征（冻结主干，替换 B 轮 ResNet18 帧特征）。

输入复用 frames_b/{eid}.npz（16×224×224 BGR 筐心帧块，与 B 轮完全同口径），
16 帧整段进 VideoMAE（时空自注意力，ResNet18 逐帧特征做不到的运动建模）。
特征 = last_hidden_state 全 token 均值池化（768 维）→ vm_feat/{eid}.npz。

增强变体（vflip/dark/bright）同 train_temporal 口径：只进训练折。
先跑 --probe 验证显存与速度，再全量。

模型：MCG-NJU/videomaev2-base（首跑自动下载 ~350MB）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "cache" / "hf"))

FRAMES_B = PROJECT_ROOT / "training" / "frames_b"
VM_OUT = PROJECT_ROOT / "training" / "vm_feat"
# v1 SSL 原版权重与 HF 键名不兼容（attention bias 未映射）；
# Kinetics 微调版是 HF 原生格式，加载干净，且语义上更贴近动作判别
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
VARIANTS = ("orig", "vflip", "dark", "bright")


def _apply_variant(x: np.ndarray, variant: str) -> np.ndarray:
    """与 train_temporal._apply_variant 完全一致。"""
    from training.train_temporal import _apply_variant as _av
    return _av(x, variant)


def load_model():
    """手动键名映射加载：MCG-NJU 仓库 checkpoint 是原版格式
    （videomae. 前缀 + 独立 q_bias/v_bias），HF 自动映射会漏掉
    attention 的 query/value bias（随机初始化 → 特征报废）。
    """
    import re
    import torch
    from transformers import VideoMAEModel, VideoMAEConfig
    from huggingface_hub import hf_hub_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = VideoMAEConfig.from_pretrained(MODEL_ID)
    model = VideoMAEModel(cfg)
    ckpt_path = hf_hub_download(MODEL_ID, "pytorch_model.bin")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    mapped = {}
    for k, v in sd.items():
        nk = k[len("videomae."):] if k.startswith("videomae.") else k
        m = re.match(r"encoder\.layer\.(\d+)\.attention\.attention\.([qv])_bias", nk)
        if m:
            tgt = "query" if m.group(2) == "q" else "value"
            nk = f"encoder.layer.{m.group(1)}.attention.attention.{tgt}.bias"
        mapped[nk] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    # 原版 VideoMAE 无 key bias：显式置零（不能留随机初始化）
    with torch.no_grad():
        for layer in model.encoder.layer:
            kb = layer.attention.attention.key.bias
            if kb is not None:
                kb.zero_()
    bad = [k for k in missing if not k.endswith("key.bias")]
    if bad:
        raise RuntimeError(f"权重映射后仍缺关键参数: {bad[:5]}")
    print(f"[VM] 权重加载成功（忽略 {len(unexpected)} 项 decoder/分类头，"
          f"key.bias 置零）")
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()   # 与 fp16 输入对齐，4GB 显存下省一半
    for p in model.parameters():
        p.requires_grad_(False)
    return model, device


def to_input(x: np.ndarray, device, dtype):
    """(16,224,224,3) uint8 BGR → (1,16,3,224,224) 归一化张量。

    VideoMAE v2 归一化：(x/255 - 0.5) / 0.5。
    """
    import torch
    t = torch.from_numpy(np.ascontiguousarray(x.astype(np.float32)[..., ::-1]))
    t = t.permute(3, 0, 1, 2).unsqueeze(0) / 255.0                 # (1,3,16,224,224)
    t = (t - 0.5) / 0.5
    return t.transpose(1, 2).to(device=device, dtype=dtype)        # (1,16,3,224,224)


def extract(events, variant="orig", batch=8):
    import torch
    VM_OUT.mkdir(parents=True, exist_ok=True)
    sub = VM_OUT if variant == "orig" else VM_OUT / variant
    sub.mkdir(parents=True, exist_ok=True)

    todo = [e for e in events
            if (FRAMES_B / f"{e['event_id']}.npz").exists()
            and not (sub / f"{e['event_id']}.npz").exists()]
    print(f"[{variant}] 待抽: {len(todo)}", flush=True)
    if not todo:
        return

    model, device = load_model()
    dtype = torch.float16 if device == "cuda" else torch.float32
    t0 = time.time()
    done = 0
    i = 0
    while i < len(todo):
        chunk = todo[i:i + batch]
        tensors = []
        for ev in chunk:
            x = np.load(FRAMES_B / f"{ev['event_id']}.npz")["x"]
            if variant != "orig":
                x = _apply_variant(x, variant)
            tensors.append(to_input(x, device, dtype))
        with torch.no_grad():
            stacked = torch.cat(tensors, dim=0)        # (B,16,3,224,224)
            out = model(pixel_values=stacked)
            feats = out.last_hidden_state.float().mean(dim=1).cpu().numpy()
        for ev, f in zip(chunk, feats):
            np.savez_compressed(sub / f"{ev['event_id']}.npz",
                                x=f.astype(np.float16))
            done += 1
        i += len(chunk)
        if done % 200 < batch:
            dt = time.time() - t0
            eta = dt / max(done, 1) * (len(todo) - done)
            print(f"  [{variant} {done}/{len(todo)}] {dt / 60:.1f}min "
                  f"({done / dt * 60:.0f} ev/min) ETA {eta / 60:.1f}min",
                  flush=True)
    print(f"[{variant}] 完成 {done}，耗时 {(time.time() - t0) / 60:.1f}min")
    if device == "cuda":
        import torch
        print(f"峰值显存: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只抽 3 个事件验证可行性")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()

    from training.extract_frames_b import load_dataset_events
    events = load_dataset_events()
    if args.probe:
        events = events[:3]
    for v in args.variants:
        extract(events, variant=v)


if __name__ == "__main__":
    main()
