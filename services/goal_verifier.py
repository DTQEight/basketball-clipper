# -*- coding: utf-8 -*-
"""四臂集成进球验证器：给检测出的候选打「自动通过」标记。

背景
    GoalDetector 找出的候选里含相当比例的误报，逐个人工确认成本高。
    本模块把离线标定的冠军集成搬到线上，给每个候选打一个 0~1 的分数，
    高分候选直接标「自动通过」，用户可跳过人工确认。

集成口径（权重与阈值由 --deploy / recalib_ensemble.py 标定，写进
training/model_temporal_meta.json 的 ensemble 段）
    A     training/model_lgbm.txt       手工特征 LGBM（±1.5s 密集 YOLO 复检提特征）
    B     training/model_b_simclr.pt    筐心彩色帧 → SimCLR 域内自监督骨干 →
                                        bigru/pool 双 TemporalNet 取 sigmoid 均值
                                        （缺失时退回 model_temporal*.pt 的 ImageNet 口径）
    Flow  training/model_flow_simclr.pt 光流幅度序列 → 光流域 SimCLR 骨干 → bigru
                                        （缺失时退回 model_flow_t.pt 的 ImageNet 口径）
    VM    training/model_vm_lgbm.txt    VideoMAE 768 维特征 → LGBM
    score = 各臂 sigmoid 概率按 ENS_WEIGHTS 加权均值（缺臂时按剩余权重重归一化，
            单臂可用时退化为该臂分）
    权重与阈值由 --deploy 写进 training/model_temporal_meta.json 的 ensemble 段，
    运行时读取；代码里的默认值仅在该段缺失时兜底。

    B/Flow/VM 三臂共享同一批 16 帧筐心裁剪块，与训练侧 FRAME_OFFS / zoom 3.2 /
    letterbox 严格同口径 —— 裁剪错位会直接毁掉打分。

重要约束
    本模块**只做标记，不删除任何候选**。低分候选照常进入人工确认流程，
    因此不会造成额外漏球。任何情况下不得改成分数过滤。
    模型缺失或加载失败时静默降级（缺哪臂按剩余权重重归一化），主流程不受影响。
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

_log = logging.getLogger("goal_verifier")

_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = _ROOT / "training"

try:
    from services.state import CACHE_ROOT
except ImportError:  # 直接以脚本方式 import 时兜底
    CACHE_ROOT = str(_ROOT / "cache")

LGBM_MODEL = TRAINING_DIR / "model_lgbm.txt"
LGBM_META = TRAINING_DIR / "model_meta.json"
TEMPORAL_BIGRU = TRAINING_DIR / "model_temporal.pt"
TEMPORAL_POOL = TRAINING_DIR / "model_temporal_pool.pt"
# B 臂换骨干：SimCLR 域内自监督编码器 + bigru/pool 双头（自包含 checkpoint）
B_SIMCLR_FILE = TRAINING_DIR / "model_b_simclr.pt"
# Flow 臂换骨干：光流域 SimCLR 自监督编码器 + bigru 头（自包含 checkpoint）
FLOW_SIMCLR_FILE = TRAINING_DIR / "model_flow_simclr.pt"
FLOW_MODEL = TRAINING_DIR / "model_flow_t.pt"
VM_MODEL = TRAINING_DIR / "model_vm_lgbm.txt"
ENSEMBLE_META = TRAINING_DIR / "model_temporal_meta.json"
# 球检测权重目录：A 臂特征由 extract_features.py 用当前球检测权重逐帧推理得到，
# 换权重会改变 A 臂输入分布 → 必须计入口径指纹（见 model_fingerprint）
BALL_WEIGHTS_DIR = _ROOT / "weights"

# 兜底阈值/权重（仅在 ENSEMBLE_META 缺 ensemble 段时生效）
AUTO_THR = 0.70
# 自动 ×（低带）阈值：低于它模型判为误报，人工可直接跳过。
# 三带分诊：≥AUTO_THR 自动 √ / <REJECT_THR 自动 × / 中间带人工只看这一带。
# 9 场 419 片段实测：0.15 时低带 160 个含 0 个真球（最低真球分 0.152），
# 高带 131 个含 1 个误报，中间带 128 个（占 30%）。
REJECT_THR = 0.15
ENS_WEIGHTS = {"lgbm": 0.5, "b": 0.5, "flow": 1.0, "vm": 1.0}

# A 臂单批候选数：A 臂逐帧 YOLO 很慢（约 6.5s/候选），分批只为刷 UI 进度
A_BATCH = 8


# ===== A 臂：手工特征 LGBM =====

_lgbm = None
_lgbm_tried = False
_lgbm_lock = threading.Lock()
_feat_names: list = []
_extract_fn = None


def _load_lgbm():
    global _lgbm, _lgbm_tried, _feat_names
    if _lgbm_tried:
        return _lgbm
    with _lgbm_lock:
        if _lgbm_tried:
            return _lgbm
        _lgbm_tried = True
        _log.info("goal_verifier: [加载] A 臂（LGBM %s）…", LGBM_MODEL.name)
        try:
            import lightgbm as lgb
            meta = json.loads(LGBM_META.read_text(encoding="utf-8"))
            names = [str(k) for k in meta.get("features", [])]
            booster = lgb.Booster(model_file=str(LGBM_MODEL))
            if len(names) != booster.num_feature():
                raise ValueError(f"meta 特征数 {len(names)} != 模型特征数 "
                                 f"{booster.num_feature()}")
            _feat_names, _lgbm = names, booster
            _log.info("goal_verifier: A 臂 LGBM 已加载（%d 特征）", len(names))
        except Exception as e:
            _log.warning("goal_verifier: A 臂加载失败（该臂禁用）: %s", e)
            _lgbm, _feat_names = None, []
            # 只有"产物缺失"才闩锁。文件在、但加载抛异常（CUDA OOM / 文件正被训练
            # 脚本改写 / 瞬时 IO）时解除闩锁，下一场视频还有机会加载回来。
            # 闩锁的代价：本进程内永久禁臂，而降级后的重归一化分数在指纹上仍表现为
            # "四臂全量口径"（_available_arms 只看文件是否存在）→ 错误无法自愈
            if LGBM_MODEL.exists() and LGBM_META.exists():
                _lgbm_tried = False
    return _lgbm


def _get_extract():
    """懒加载 training/extract_features.py 的 extract()（脚本非包，用 importlib）。"""
    global _extract_fn
    if _extract_fn is not None:
        return _extract_fn
    try:
        mod = _load_module(str(TRAINING_DIR / "extract_features.py"),
                           "bball_extract_features")
        _extract_fn = mod.extract
    except Exception as e:
        _log.warning("goal_verifier: extract_features 加载失败（A 臂禁用）: %s", e)
        _extract_fn = False
    return _extract_fn or None


# ===== 通用：按文件路径加载 training/ 下的脚本（training 非包） =====

def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===== B / Flow 臂：ResNet18 帧特征 + TemporalNet =====

_temporal = None
_temporal_tried = False
_temporal_lock = threading.Lock()
_train_temporal_mod = None
_frames_b_mod = None


def _load_temporal():
    """懒加载 B/Flow 臂推理栈。

    两臂都优先用各自域内的 SimCLR 自监督编码器，缺失时退回 ImageNet：
      B      training/model_b_simclr.pt（筐心彩色帧域，自包含骨干 + 双头）
             → 退回 ImageNet ResNet18 + model_temporal*.pt
      Flow   training/model_flow_simclr.pt（光流域，自包含骨干 + bigru 头）
             → 退回 ImageNet ResNet18 + model_flow_t.pt

    预处理口径必须严格区分（两臂各自与自己的编码器训练口径一致）：
      B(SimCLR)    BGR → RGB，/255，**不做** mean/std（编码器在裸 /255 上训的）
      Flow(SimCLR) 光流幅度 /255，**不做** mean/std（三通道相同，无通道序问题）
      ImageNet 口径才做 mean/std
    同时读取 ensemble 段的权重与阈值。失败返回 None。
    """
    global _temporal, _temporal_tried, AUTO_THR, ENS_WEIGHTS
    if _temporal_tried:
        return _temporal
    with _temporal_lock:
        if _temporal_tried:
            return _temporal
        _temporal_tried = True
        try:
            import torch
            import torchvision.models as tvm

            global _train_temporal_mod, _frames_b_mod
            if _train_temporal_mod is None:
                _train_temporal_mod = _load_module(
                    str(TRAINING_DIR / "train_temporal.py"), "bball_train_temporal")
            if _frames_b_mod is None:
                _frames_b_mod = _load_module(
                    str(TRAINING_DIR / "extract_frames_b.py"), "bball_frames_b")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            def _backbone(sd=None):
                """resnet18(fc→Identity)：sd=None 用 ImageNet，否则用给定权重。"""
                net = (tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
                       if sd is None else tvm.resnet18(weights=None))
                net.fc = torch.nn.Identity()
                if sd is not None:
                    net.load_state_dict(sd)
                return net.to(device).eval()

            def _load_net(path):
                ck = torch.load(path, map_location=device, weights_only=False)
                net = _train_temporal_mod.TemporalNet(
                    dim=ck.get("dim", 512), arch=ck["arch"]).to(device)
                net.load_state_dict(ck["state_dict"])
                return net.eval()

            # ---- B 臂：SimCLR 骨干优先，缺失退回 ImageNet ----
            # 心跳：torch.load + 骨干 + TemporalNet 在慢盘/首次加载可能要几十秒。
            # 日志停在这条之后 = 卡在 B/Flow 加载（不是复核推理）
            _log.info("goal_verifier: [加载] 进入 B/Flow 臂（device=%s）…", device)
            b_resnet = b_nets = None
            b_pre = "rgb255"
            try:
                ck = torch.load(B_SIMCLR_FILE, map_location=device,
                                weights_only=False)
                b_resnet = _backbone(ck["backbone"])
                b_nets = []
                for n in ck["nets"]:
                    net = _train_temporal_mod.TemporalNet(
                        dim=ck.get("dim", 512), arch=n["arch"]).to(device)
                    net.load_state_dict(n["state_dict"])
                    b_nets.append(net.eval())
                _log.info("goal_verifier: B 臂骨干 = SimCLR 自监督（%s，OOF %s）",
                          "+".join(n["arch"] for n in ck["nets"]),
                          ck.get("oof_auc"))
            except Exception as e:
                _log.warning("goal_verifier: SimCLR B 臂加载失败，退回 ImageNet: %s", e)
                b_resnet = b_nets = None
            if b_nets is None:
                # M1：兜底整体包 try（**必须含 `_backbone()`**）—— 否则这里任一异常
                # 会冒泡到外层 except，把**完好的 Flow 臂一起禁用**（两臂不对称）。
                # `_backbone()` 用 torchvision 的 ImageNet 权重，缓存缺失且离线时
                # 会抛，正是高发点；旧补丁只包了 `_load_net`，漏了它。
                try:
                    b_resnet = _backbone()
                    b_pre = "imagenet"
                    b_nets = [_load_net(p) for p in (TEMPORAL_BIGRU, TEMPORAL_POOL)]
                except Exception as e:
                    _log.warning("goal_verifier: B 臂 ImageNet 兜底也失败"
                                 "（该臂禁用）: %s", e)
                    b_resnet = b_nets = None

            # ---- Flow 臂：光流域 SimCLR 骨干优先，缺失退回 ImageNet ----
            _log.info("goal_verifier: [加载] B 臂就绪（骨干=%s），进入 Flow 臂…", b_pre)
            flow_resnet = flow_nets = None
            flow_pre = "rgb255"
            try:
                ck = torch.load(FLOW_SIMCLR_FILE, map_location=device,
                                weights_only=False)
                flow_resnet = _backbone(ck["backbone"])
                flow_nets = []
                for n in ck["nets"]:
                    net = _train_temporal_mod.TemporalNet(
                        dim=ck.get("dim", 512), arch=n["arch"]).to(device)
                    net.load_state_dict(n["state_dict"])
                    flow_nets.append(net.eval())
                _log.info("goal_verifier: Flow 臂骨干 = 光流 SimCLR（%s，OOF %s）",
                          "+".join(n["arch"] for n in ck["nets"]),
                          ck.get("oof_auc"))
            except Exception as e:
                _log.warning("goal_verifier: 光流 SimCLR Flow 臂加载失败，"
                             "退回 ImageNet: %s", e)
                flow_resnet = flow_nets = None
            if flow_nets is None:
                # 与 B 臂同理（M1）：`_backbone()` 也必须在 try 内
                try:
                    flow_resnet = _backbone()
                    flow_pre = "imagenet"
                    flow_nets = [_load_net(FLOW_MODEL)]
                except Exception as e:
                    _log.warning("goal_verifier: Flow 臂加载失败（该臂禁用）: %s", e)
                    flow_resnet = flow_nets = None

            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

            _read_ensemble()

            _temporal = {
                "b_resnet": b_resnet, "b_nets": b_nets, "b_pre": b_pre,
                "flow_resnet": flow_resnet, "flow_nets": flow_nets,
                "flow_pre": flow_pre,
                "device": device, "mean": mean, "std": std,
                "frame_offs": list(_frames_b_mod.FRAME_OFFS),
                "crop_hoop": _frames_b_mod.crop_hoop,
            }
            _log.info("goal_verifier: B/Flow 臂已加载 (device=%s, B骨干=%s, "
                      "Flow骨干=%s, flow=%s)",
                      device, b_pre, flow_pre, "on" if flow_nets else "off")
        except Exception as e:
            _log.warning("goal_verifier: B/Flow 臂加载失败（两臂禁用）: %s", e)
            _temporal = None
            # 同 _load_lgbm：产物在时解除闩锁，允许后续重试（torch/CUDA 的瞬时失败
            # 不该让这两臂在本进程内永久失效）
            if any(p.exists() for p in (B_SIMCLR_FILE, FLOW_SIMCLR_FILE,
                                        TEMPORAL_BIGRU, TEMPORAL_POOL, FLOW_MODEL)):
                _temporal_tried = False
    return _temporal


# ===== VM 臂：VideoMAE 冻结主干 + LGBM 头 =====
#
# 2026.09.19 试过把 LGBM 换成时序头（并改用逐时序位置特征保留运动结构）：
# OOF 上确实更好（单臂 0.8208→0.8838、集成 0.9640→0.9741），但 **8 场真实视频上没兑现**
# ——AUC 持平（0.9858 vs 0.9856），同等精度（各 1 个误报）下召回反而更低
# （LGBM 83.5% vs 时序头 73.7%；时序头分数更饱和、工作点对阈值更敏感）。
# 故不部署，实验脚本与产物见 training/train_vm_head.py 与 training/exp_vm_head/。

_vm = None
_vm_tried = False
_vm_lock = threading.Lock()


def _load_vm():
    """懒加载 VideoMAE 臂。权重缓存在 cache/hf（extract_videomae.py 下载）。

    服务进程不允许运行时下载（HF_HUB_OFFLINE=1），缺缓存/缺依赖则该臂禁用。
    输入口径与训练严格一致：(16,224,224,3) uint8 BGR → RGB → (x/255-0.5)/0.5，
    特征为 last_hidden_state 全 token 均值池化（768 维）。
    """
    global _vm, _vm_tried
    if _vm_tried:
        return _vm
    with _vm_lock:
        if _vm_tried:
            return _vm
        _vm_tried = True
        _log.info("goal_verifier: [加载] VM 臂（VideoMAE，离线缓存）…")
        try:
            os.environ.setdefault("HF_HOME", str(Path(CACHE_ROOT) / "hf"))
            os.environ["HF_HUB_OFFLINE"] = "1"
            import lightgbm as lgb

            mod = _load_module(str(TRAINING_DIR / "extract_videomae.py"),
                               "bball_extract_videomae")
            model, device = mod.load_model()
            booster = lgb.Booster(model_file=str(VM_MODEL))
            _vm = {
                "model": model, "booster": booster, "device": device,
                "dtype": next(model.parameters()).dtype,
                "to_input": mod.to_input,
            }
            _log.info("goal_verifier: VM 臂已加载 (device=%s, dtype=%s)",
                      device, _vm["dtype"])
        except Exception as e:
            _log.info("goal_verifier: VM 臂不可用（该臂禁用）: %s", e)
            _vm = None
            # 同 _load_lgbm：产物在时解除闩锁，允许后续重试
            if VM_MODEL.exists():
                _vm_tried = False
    return _vm


# ===== 集成标定（阈值/权重）=====

_ens_mtime = None


def _read_ensemble():
    """读 model_temporal_meta.json 的 ensemble 段（阈值 + 权重），文件变更自动失效。

    刻意不加载任何模型：历史回读时要按当前阈值重推 auto 标记，不该为此付一次
    ResNet18/VideoMAE 的加载代价。
    """
    global AUTO_THR, REJECT_THR, ENS_WEIGHTS, _ens_mtime
    try:
        mtime = ENSEMBLE_META.stat().st_mtime
        if _ens_mtime != mtime:
            ens = json.loads(ENSEMBLE_META.read_text(
                encoding="utf-8")).get("ensemble", {})
            _ens_mtime = mtime
            AUTO_THR = float(ens.get("keep_thr", AUTO_THR))
            REJECT_THR = float(ens.get("reject_thr", REJECT_THR))
            w = ens.get("weights")
            if isinstance(w, dict) and w:
                ENS_WEIGHTS = {k: float(v) for k, v in w.items()}
            _log.info("goal_verifier: 集成标定 keep_thr=%s reject_thr=%s 权重=%s"
                      "（组合 %s）",
                      AUTO_THR, REJECT_THR, ENS_WEIGHTS, ens.get("composition"))
        return AUTO_THR
    except Exception as e:
        _log.warning("goal_verifier: ensemble 段读取失败，用代码默认 "
                     "keep_thr=%s reject_thr=%s 权重=%s: %s",
                     AUTO_THR, REJECT_THR, ENS_WEIGHTS, e)
        return AUTO_THR


# ===== 公共查询接口 =====

def auto_threshold() -> float:
    """当前自动通过阈值（读 ensemble.keep_thr，缺省用代码默认值）。"""
    return _read_ensemble()


def reject_threshold() -> float:
    """当前自动排除阈值（读 ensemble.reject_thr，缺省用代码默认值）。"""
    _read_ensemble()
    return REJECT_THR


# ===== AI 复核总开关（UI「AI 识别」）=====
# 关闭后不再跑四臂打分：检测更快、不产生任何 auto / auto_reject 标记，
# 所有候选都留给人工判定。仅进程内生效，重启恢复默认开启（与其余 UI 开关一致）。
_ENABLED = True


def set_enabled(on: bool) -> None:
    """设置是否启用 AI 复核（四臂集成）。"""
    global _ENABLED
    _ENABLED = bool(on)
    _log.info("goal_verifier: AI 复核已%s", "开启" if _ENABLED else "关闭")


def is_enabled() -> bool:
    """当前是否启用 AI 复核。"""
    return _ENABLED


def _available_arms() -> list:
    """当前可参与打分的臂（只看产物齐备性，不加载模型）。

    R9：指纹必须涵盖**实际参与的臂组合**。缺臂时 combine 会按剩余权重重归一化，
    算出的分数与四臂全量不是同一口径；若两者共用同一指纹，臂恢复后
    invalidate_stale 会判"口径未变"而不重算，降级分数就冒充了全量口径
    （阈值按全组合 OOF 标定，缺臂沿用即错配 → 低带误 × 漏球、高带误 √ 固化）。
    """
    arms = []
    if LGBM_MODEL.exists():
        arms.append("a")
    # B 臂：SimCLR 优先，缺失时退 ImageNet 双头兜底
    if B_SIMCLR_FILE.exists() or (TEMPORAL_BIGRU.exists() and TEMPORAL_POOL.exists()):
        arms.append("b")
    if FLOW_SIMCLR_FILE.exists() or FLOW_MODEL.exists():
        arms.append("flow")
    if VM_MODEL.exists():
        arms.append("vm")
    return arms


def model_fingerprint() -> str:
    """当前判定口径的指纹：各臂模型文件 + 球检测权重 + 集成权重/阈值。

    分数只在某一套「模型 + 权重 + 阈值」下有意义。换骨干、调权重、改阈值之后，
    缓存里的旧 score 会和新阈值组合出错误的 √/×，所以必须能识别出"这批分数
    是另一套口径算的"。

    A 臂与球检测权重也必须进指纹（2026.09.21 补）：
      · A 臂特征由 extract_features.py 用**当前球检测权重**逐帧推理得到，
        换 weights/*.pt 会改变 A 臂输入分布 → A 臂分变化
      · 重训 model_lgbm.txt 同样直接改变 A 臂分
    两者都与阈值组合决定 √/×，漏掉就会出现「新旧口径分数混在一起」的静默错判。
    球检测权重按目录内全部 *.pt 的 mtime 计入（不加载模型，避开 get_ball_model 的开销）。
    """
    h = hashlib.md5()
    # 必须先刷新 ENS_WEIGHTS/AUTO_THR：否则首次调用会把「代码默认权重」哈希进去，
    # 而下一次调用哈希的是文件里的权重 → 指纹自己就变了（分数被反复误判为过期）
    thr = _read_ensemble()
    for p in (B_SIMCLR_FILE, FLOW_SIMCLR_FILE, TEMPORAL_BIGRU, TEMPORAL_POOL,
              FLOW_MODEL, VM_MODEL, LGBM_MODEL):
        try:
            h.update(f"{p.name}:{p.stat().st_mtime_ns}".encode())
        except OSError:
            h.update(f"{p.name}:-".encode())
    try:
        for p in sorted(BALL_WEIGHTS_DIR.glob("*.pt")):
            h.update(f"w:{p.name}:{p.stat().st_mtime_ns}".encode())
    except OSError:
        h.update(b"w:-")
    h.update(json.dumps(sorted(ENS_WEIGHTS.items())).encode())
    h.update(str(thr).encode())
    # R9：把「当前可参与的臂组合」计入指纹（缺臂 → 分数是重归一化的另一口径）
    h.update(",".join(_available_arms()).encode())
    return h.hexdigest()[:12]


def invalidate_stale(clips) -> int:
    """丢弃口径已过期的分数（就地）。返回被作废的片段数。

    verify_ver 与当前指纹不一致（或旧数据根本没有该字段）→ 清掉 score 及各臂分，
    调用方的"缺分数"分支就会重跑复核。

    M2：同时清掉**模型来源**的 mark/mark_source —— mark 是旧口径分数分带的产物，
    留着会在本片段重打分失败时被 _sync_marks 当成有效判断写进历史（口径污染）。
    人工标记（mark_source == "manual"）永远不动。
    """
    fp = model_fingerprint()
    n = 0
    for c in clips:
        if c.get("verify_ver") == fp:
            continue
        if "score" not in c and c.get("mark_source") != "auto":
            continue                    # 既无分数也无模型标记 → 无事可做
        for k in ("score", "verify_ver", "auto", "auto_reject", "verify_score",
                  "score_lgbm", "score_b", "score_flow", "score_vm"):
            c.pop(k, None)
        if c.get("mark_source") == "auto":
            c.pop("mark", None)
            c.pop("mark_source", None)
        n += 1
    return n


def refresh_auto(clips) -> int:
    """按当前阈值从已有 score 重推三带判定，并对高/低带自动打标（不重跑模型）。

    三带：
      高带 score >= keep_thr   → mark=keep,  mark_source=auto（可直接跳过人工）
      中间带 keep>s>=reject    → 清掉模型标记，**留给人工**（这是唯一要看的带）
      低带 score <  reject_thr → mark=reject, mark_source=auto（模型判为误报）
    自动标记只写 mark_source != 'manual' 的片段：**人工标记永远优先，绝不覆盖**
    （用户已判 × 的球不能因为模型给高分就被翻成 √）。阈值可被手改
    （model_temporal_meta.json），片段缓存里的标记是旧阈值的产物——历史回读不
    重推就会一直显示过时的徽标；阈值调高后原本 auto 的片段会落回中间带，
    此时必须把模型标记清掉，否则会出现"没人看却已被判定"的片段。
    """
    keep_thr = auto_threshold()
    rej_thr = reject_threshold()
    n = 0
    for c in clips:
        if "score" not in c:
            # N6：无分（本轮重打分失败 / 已被 invalidate_stale 清掉）时，残留的
            # 模型标记必须清掉——它是**旧口径**分带的产物，留着会被 _sync_marks
            # 当成有效判断写进历史（口径污染）。人工标记不动。
            if c.get("mark_source") == "auto":
                c.pop("mark", None)
                c.pop("mark_source", None)
                n += 1
            continue
        s = float(c["score"])
        c["verify_score"] = round(s, 3)
        c["auto"] = bool(s >= keep_thr)
        c["auto_reject"] = bool(s < rej_thr)
        if c.get("mark_source") == "manual":
            continue
        if c["auto"]:
            c["mark"], c["mark_source"] = "keep", "auto"
            n += 1
        elif c["auto_reject"]:
            c["mark"], c["mark_source"] = "reject", "auto"
        else:
            c.pop("mark", None)
            c.pop("mark_source", None)
    return n


def unavailable_reason() -> str:
    """各臂可用性摘要，供日志/UI 提示。"""
    t = _load_temporal()
    return (f"A={'on' if _load_lgbm() is not None else 'off'} "
            f"B/Flow={'on' if t else 'off'} "
            f"VM={'on' if _load_vm() else 'off'} thr={auto_threshold():.3f}")


# ===== 打分 =====

def _flow_maps(frames):
    """16 帧筐心块 → Farneback 光流幅度序列 (15,224,224,3) uint8。

    与训练侧 extract_motion.run_flow 严格一致：逐帧灰度 →
    calcOpticalFlowFarneback(0.5,3,21,3,5,1.2,0) → 幅度×12 clip uint8
    → 3 通道复制（喂 ResNet18 的口径）。
    """
    import cv2
    import numpy as np
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = frames.shape[1], frames.shape[2]
    mags = np.zeros((len(gray) - 1, h, w), dtype=np.uint8)
    for j in range(len(gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray[j], gray[j + 1], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags[j] = np.clip(mag * 12.0, 0, 255).astype(np.uint8)
    return np.repeat(mags[..., None], 3, axis=-1)


def _score_visual(video_path, clips, hoop, on_progress=None):
    """B/Flow/VM 三臂打分（共享同一批 16 帧筐心块解码）。

    就地写 clip['score_b'] / clip['score_flow'] / clip['score_vm']（float 0~1）。
    片段按 ts 排序顺序解码（单 reader，不逐片段重开视频）。
    某臂缺模型或推理失败时静默跳过（集成端按剩余权重重归一化）。
    """
    tt = _load_temporal()
    # todo：只跳过"三臂分数齐全"的片段。旧判据只看 score_b —— 若上一次运行里 B
    # 成功而 Flow/VM 抛异常，该片段会带着 score_b 被**永久**排除出 todo，缺的臂
    # 再也补不回来（combine 按剩余权重重归一化 → 与四臂口径不可比，而分数已被
    # 缓存与历史固化成"全量口径"）
    _VIS_ARMS = ("score_b", "score_flow", "score_vm")
    todo = [c for c in clips if not all(k in c for k in _VIS_ARMS)]
    if tt is None or not todo or not hoop:
        if on_progress:
            on_progress(1.0, '视觉三臂不可用')
        return
    import numpy as np
    import torch
    from video_io import VideoReader

    todo.sort(key=lambda c: float(c["ts"]))
    b_resnet, b_nets, b_pre = tt["b_resnet"], tt["b_nets"], tt["b_pre"]
    flow_resnet, flow_nets = tt["flow_resnet"], tt.get("flow_nets")
    flow_pre = tt.get("flow_pre", "imagenet")
    device = tt["device"]
    mean, std = tt["mean"], tt["std"]
    frame_offs, crop_hoop = tt["frame_offs"], tt["crop_hoop"]
    if on_progress:
        on_progress(0.0, '视觉三臂：加载模型')
    vm = _load_vm()
    n_frames, min_valid = len(frame_offs), 8
    # 再按"实际可用的臂"筛一次：某臂本进程不可用时它的键永远缺失，不筛的话每次
    # 重跑都会把所有片段当成待办、把其余臂的分数白白重算一遍
    _need = [k for k, _on in (("score_b", b_nets is not None),
                              ("score_flow", flow_nets is not None),
                              ("score_vm", vm is not None)) if _on]
    if _need:
        todo = [c for c in todo if not all(k in c for k in _need)]
    if not todo:
        if on_progress:
            on_progress(1.0, '视觉三臂已完成')
        return
    _log.info("goal_verifier: [B/Flow/VM] 开始打分 %d 个候选（B=%s Flow=%s VM=%s）",
              len(todo), bool(b_nets), bool(flow_nets), vm is not None)

    blocks = []  # [(clip, (16,224,224,3) uint8 BGR)]
    reader = None
    try:
        _log.info("goal_verifier: [B/Flow/VM] 抽帧开始（单 reader 顺序解码）")
        # single_thread：A 臂刚跑完 YOLO CUDA，再走 PyAV 帧线程池解码会偶发
        # 永久卡死（与 read_frame 同一规避，见 video_io.VideoReader 注释）
        reader = VideoReader(video_path, single_thread=True)
        fps, total = reader.fps, reader.total
        for k, c in enumerate(todo):
            ts = float(c["ts"])
            frames = np.zeros((n_frames, 224, 224, 3), dtype=np.uint8)
            # offsets 单调递增 → 顺序解码到最后一帧（同 extract_frames_b.main）
            last = max(0, min(int((ts + frame_offs[-1]) * fps), total - 1))
            want = {}
            for i, off in enumerate(frame_offs):
                fidx = max(0, min(int((ts + off) * fps), total - 1))
                want[fidx] = i
            got = 0
            for fidx, frame in reader.iter_frames(start=min(want), end=last + 1):
                if fidx in want:
                    frames[want[fidx]] = crop_hoop(frame, hoop)
                    got += 1
                    if got >= n_frames:
                        break
            if got >= min_valid:
                blocks.append((c, frames))
            if on_progress and (k + 1) % 8 == 0:
                # 抽帧（解码）占视觉阶段前半：让进度在长时间解码里也能动
                on_progress(0.5 * (k + 1) / len(todo),
                            f'三臂抽帧 {k + 1}/{len(todo)}')
                _log.info("goal_verifier: [B/Flow/VM] 抽帧 %d/%d（已有 %d 个有效块）",
                          k + 1, len(todo), len(blocks))
    except Exception as e:
        _log.warning("goal_verifier: 筐心块抽帧失败 %s: %s", video_path, e)
        return
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
    if on_progress:
        on_progress(0.5, f'三臂抽帧 {len(blocks)}/{len(todo)}')
    _log.info("goal_verifier: [B/Flow/VM] 抽帧完成 %d/%d，进入 GPU 推理",
              len(blocks), len(todo))

    # 批量 GPU 推理（4 片段 = 64 帧/批，同训练特征提取的显存预算）
    try:
        n_chunks = (len(blocks) + 3) // 4
        for i in range(0, len(blocks), 4):
            chunk = blocks[i:i + 4]
            bn = i // 4 + 1
            if on_progress:
                on_progress(0.5 + 0.5 * i / max(len(blocks), 1),
                            f'三臂推理 {i}/{len(blocks)}')
            # ---- B 臂：帧特征 → bigru/pool 双头均值 ----
            # 预处理按骨干口径分流：SimCLR 用裸 /255（编码器这么训的），
            # 退回 ImageNet 时才是 mean/std
            # 与 Flow 臂对称：本臂不可用（None）或推理失败时只跳过它，不连累其它臂
            if b_nets:
                _log.info("goal_verifier: [B/Flow/VM] 批 %d/%d：B 臂推理…", bn, n_chunks)
                try:
                    arr = np.concatenate([b[1] for b in chunk]).astype(np.float32) / 255.0
                    arr = arr[..., ::-1]  # BGR→RGB
                    arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
                    with torch.no_grad():
                        t = torch.from_numpy(arr).to(device)
                        if b_pre == "imagenet":
                            t = (t - mean) / std
                        feat = b_resnet(t)  # (N*16, 512)
                        for j, (c, _) in enumerate(chunk):
                            f = feat[j * n_frames:(j + 1) * n_frames].unsqueeze(0)
                            ps = [float(torch.sigmoid(n(f)).item()) for n in b_nets]
                            c["score_b"] = round(sum(ps) / len(ps), 3)
                except Exception as e:
                    _log.warning("goal_verifier: B 臂推理失败（跳过该臂）: %s", e)
                    for c, _ in chunk:
                        c.pop("score_b", None)
            # ---- Flow 臂：光流幅度序列 → ResNet18 → bigru ----
            # 与 B 臂同理按骨干口径分流：光流 SimCLR 用裸 /255，退回
            # ImageNet 时才是 mean/std（两套口径不能混）
            if flow_nets:
                _log.info("goal_verifier: [B/Flow/VM] 批 %d/%d：Flow 臂推理…", bn, n_chunks)
                try:
                    fmap_blocks = [_flow_maps(b[1]) for b in chunk]
                    farr = np.concatenate(fmap_blocks).astype(np.float32) / 255.0
                    farr = np.ascontiguousarray(farr.transpose(0, 3, 1, 2))
                    with torch.no_grad():
                        ft = torch.from_numpy(farr).to(device)
                        if flow_pre == "imagenet":
                            ft = (ft - mean) / std
                        ffeat = flow_resnet(ft)  # (N*15, 512)
                        for j, (c, _) in enumerate(chunk):
                            nf = fmap_blocks[j].shape[0]
                            ff = ffeat[j * nf:(j + 1) * nf].unsqueeze(0)
                            ps = [float(torch.sigmoid(n(ff)).item())
                                  for n in flow_nets]
                            c["score_flow"] = round(sum(ps) / len(ps), 3)
                except Exception as e:
                    _log.warning("goal_verifier: Flow 臂推理失败（跳过该臂）: %s", e)
                    for c, _ in chunk:
                        c.pop("score_flow", None)
            # ---- VM 臂：VideoMAE 768 维 → LGBM Booster ----
            if vm is not None:
                _log.info("goal_verifier: [B/Flow/VM] 批 %d/%d：VM 臂推理…", bn, n_chunks)
                try:
                    with torch.no_grad():
                        t_in = torch.cat(
                            [vm["to_input"](b, vm["device"], vm["dtype"])
                             for _, b in chunk], dim=0)
                        out = vm["model"](pixel_values=t_in)
                        vfeat = out.last_hidden_state.float().mean(dim=1)
                    preds = vm["booster"].predict(vfeat.cpu().numpy())
                    for (c, _), p in zip(chunk, preds):
                        c["score_vm"] = round(min(max(float(p), 0.0), 1.0), 3)
                except Exception as e:
                    _log.warning("goal_verifier: VM 臂推理失败（跳过该臂）: %s", e)
                    for c, _ in chunk:
                        c.pop("score_vm", None)
    except Exception as e:
        _log.warning("goal_verifier: 三臂推理失败 %s: %s", video_path, e)
        # 逐批推进，异常可能出现在中途 → todo 里只有一部分片段拿到了臂分。
        # 留着会让同一批候选混用两套重归一化口径（部分 3 臂 / 部分 1 臂，分数
        # 不可比）；统一撤掉本次尝试过的视觉臂分，交给下次重算
        for _c in todo:
            for _k in _VIS_ARMS:
                _c.pop(_k, None)
    else:
        _log.info("goal_verifier: [B/Flow/VM] GPU 推理完成（%d 批 / %d 片段）",
                  (len(blocks) + 3) // 4, len(blocks))


def _score_lgbm(video_path, clips, hoop, on_progress=None):
    """A 臂打分：±1.5s 密集 YOLO 复检提手工特征 → LGBM。就地写 clip['score_lgbm']。

    分批调用 extract()（每批 A_BATCH 个候选）：单批耗时长（逐帧 YOLO），
    分批是为了让 on_progress 有机会刷新 UI 进度——否则整个复核阶段界面全黑箱。
    """
    model = _load_lgbm()
    extract = _get_extract()
    todo = [c for c in clips if "score_lgbm" not in c]
    if model is None or extract is None or not todo or not hoop:
        if on_progress:
            on_progress(1.0, 'A臂不可用')
        return
    try:
        from video_io import get_video_info
        info = get_video_info(video_path)
    except Exception as e:
        _log.warning("goal_verifier: 读取视频信息失败 %s: %s", video_path, e)
        if on_progress:
            on_progress(1.0, 'A臂读取视频失败')
        return
    try:
        from app import get_ball_model, get_ball_class_ids, get_device
        m, weights = get_ball_model()
        ball_classes, device = get_ball_class_ids(m, weights), get_device()
    except Exception as e:
        _log.warning("goal_verifier: A 臂 YOLO 不可用（跳过该臂）: %s", e)
        return

    n_done, n_err = 0, 0
    nb = (len(todo) + A_BATCH - 1) // A_BATCH
    tb = time.time()
    _log.info("goal_verifier: [A 臂] 开始提特征：%d 个候选 / %d 批", len(todo), nb)
    for s in range(0, len(todo), A_BATCH):
        chunk = todo[s:s + A_BATCH]
        _log.info("goal_verifier: [A 臂] 批 %d/%d（%d 个候选）YOLO 逐帧提特征…",
                  s // A_BATCH + 1, nb, len(chunk))
        events, clip_by_eid = [], {}
        for c in chunk:
            ts = round(float(c["ts"]), 3)
            eid = f"live_{int(ts * 1000):010d}"
            events.append({
                "event_id": eid, "video": video_path, "ts": ts,
                "hoop": list(hoop), "label": 0,
                "video_width": info["width"], "video_height": info["height"],
            })
            clip_by_eid[eid] = c
        try:
            feats, errors = extract(m, ball_classes, device, events)
        except Exception as e:
            # N13：单批提特征失败只跳过**这一批**。旧实现 break 会放弃其后全部
            # 候选的 A 臂打分——部分片段缺 A 臂而其余有，跨候选分数不可比
            # （combine 对缺臂片段按剩余权重重归一化，等于换了个口径）。
            _log.warning("goal_verifier: A 臂提特征失败（跳过本批 %d 个候选）: %s",
                         len(chunk), e)
            n_err += len(chunk)
            continue
        n_err += len(errors)
        for row in feats:
            c = clip_by_eid.get(row.get("event_id"))
            if c is None:
                continue
            try:
                # 特征顺序以 meta.features 为准（Booster 内部特征名是 Column_N）
                vals = [float(row.get(k, 0.0)) for k in _feat_names]
                c["score_lgbm"] = round(
                    min(max(float(model.predict([vals])[0]), 0.0), 1.0), 3)
            except Exception:
                continue
        n_done += len(chunk)
        if on_progress:
            on_progress(min(n_done, len(todo)) / len(todo),
                        f'A臂特征 {min(n_done, len(todo))}/{len(todo)}')
    if n_err:
        _log.info("goal_verifier: A 臂有 %d 个候选提特征失败（保留人工判断）", n_err)
    _log.info("goal_verifier: [A 臂] 提特征完成：%d 个候选（失败 %d，耗时 %.1fs）",
              n_done, n_err, time.time() - tb)


def combine(clip) -> float | None:
    """按 ENS_WEIGHTS 组合一个 clip 的各臂分数（缺臂按剩余权重重归一化）。"""
    arms = {"lgbm": clip.get("score_lgbm"), "b": clip.get("score_b"),
            "flow": clip.get("score_flow"), "vm": clip.get("score_vm")}
    avail = [(ENS_WEIGHTS.get(k, 0.0), v) for k, v in arms.items()
             if v is not None and ENS_WEIGHTS.get(k, 0.0) > 0]
    if not avail:
        return None
    wsum = sum(w for w, _ in avail)
    return sum(w * v for w, v in avail) / wsum


def score_clips(video_path, clips, hoop, progress=None):
    """给 clips 跑四臂打分，就地写各臂分与集成分。返回打分成功的片段数。

    progress: 可选回调 progress(frac: float, stage: str)，供 UI 显示长耗时的复核进度。
    A 臂独占前 70%（逐帧 YOLO 是绝对瓶颈），视觉三臂占后 30%。
    """
    if not clips or not hoop:
        return 0

    def _p_a(frac, stage):
        if progress:
            progress(0.05 + 0.65 * frac, stage)

    def _p_v(frac, stage):
        if progress:
            progress(0.70 + 0.30 * frac, stage)

    # 心跳：日志停在哪一条，就说明卡在哪一臂（服务卡死时唯一现场）
    _log.info("goal_verifier: [复核开始] %s —— %d 个候选", video_path, len(clips))
    t0 = time.time()
    _score_lgbm(video_path, clips, hoop, on_progress=_p_a)
    _log.info("goal_verifier: [复核] A 臂结束（%.1fs），进入视觉三臂",
              time.time() - t0)
    _score_visual(video_path, clips, hoop, on_progress=_p_v)
    _log.info("goal_verifier: [复核] 视觉三臂结束（%.1fs），开始融合",
              time.time() - t0)
    # 打分完成后才取指纹：ENS_WEIGHTS 是 _load_temporal 里读进来的
    fp = model_fingerprint()
    n = 0
    for c in clips:
        s = combine(c)
        if s is None:
            continue
        c["score"] = round(s, 3)
        c["verify_ver"] = fp
        n += 1
    _log.info("goal_verifier: [复核完成] %d/%d 个候选有集成分（耗时 %.1fs，指纹 %s）",
              n, len(clips), time.time() - t0, fp)
    return n


def mark_auto(clips, video_path, hoop, progress=None):
    """给 clips 打上 auto 标记（就地修改）。返回自动通过的数量。

    clip 需含 "ts"；会新增 verify_score / auto（及各臂分供排查）。
    progress: 可选进度回调（见 score_clips）。
    **只标记，不删除候选**；任何异常都不抛出——打分是增强功能，失败必须静默降级。
    """
    if not clips or not _ENABLED:
        # 关闭「AI 识别」时兜底：调用方已判过开关，这里再判一次，
        # 避免将来新增调用点漏判而白跑一遍四臂推理
        return 0
    try:
        n_scored = score_clips(video_path, clips, hoop, progress=progress)
        if not n_scored:
            return 0
        return refresh_auto(clips)
    except Exception as e:
        _log.warning("goal_verifier: 打标异常（静默跳过）: %s", e)
        return 0
