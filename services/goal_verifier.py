# -*- coding: utf-8 -*-
"""L4 进球验证器：冠军集成判决（两端自动 √/×）。

检测完成后台调用：
  1) A 轮 LGBM：对每个候选片段做 ±1.5s 密集 YOLO 复检提取手工特征，
     用 training/model_lgbm.txt 打分（写 clip['score_lgbm']）。
  1.5) 三个视觉臂（共享同一批 16 帧篮筐中心裁剪块，与训练严格同口径）：
     B   ResNet18 帧特征 → pool+bigru 双 TemporalNet 均值（score_b）
     Flow Farneback 光流幅度序列 → ResNet18 → TemporalNet（score_flow，
          training/model_flow_t.pt）
     VM  VideoMAE 768 维特征 → LGBM（score_vm，training/model_vm_lgbm.txt）
     冠军集成分 clip['score'] = 各臂 sigmoid 概率按 ENS_WEIGHTS 加权均值
     （mean4，2232 事件 OOF AUC 0.9632，p95 工作点召回 78.2%；组合与阈值
     由 train_directions.py --deploy 标定，存 model_temporal_meta.json
     的 ensemble 段；缺臂时按剩余权重重归一化，单臂可用退化为该臂分）：
       score >= AB_KEEP_THR → 自动标记 √（mark='keep'）
       score <= REJECT_THR  → 自动标记 ×（mark='reject'）
       中间区间留灰区，等人工标记
  （VLM 灰区仲裁已于 2026-08-31 移除：盲测中灰区决策反而降准确率
  88.2% vs 91.2%，离线评测脚本保留在 training/vlm_eval*.py。）

人工标记过的片段（mark_source='manual'）不会被自动标记覆盖。
模型/元数据缺失或加载失败时各阶段静默禁用（B 轮挂了退化为纯 LGBM 分），
检测主流程不受影响。

全自动模式（FULL_AUTO，UI 开关）：检测后零人工快速粗剪——集成分 >= keep_thr
自动 √，其余全部自动 ×（灰区默认×，盲测准确率 91.2%/精确率 100%），跳过
每场自适应校准。
"""
from __future__ import annotations

import importlib.util
import json
import logging
import threading
from pathlib import Path

_log = logging.getLogger("goal_verifier")

TRAINING_DIR = Path(__file__).resolve().parent.parent / "training"
MODEL_FILE = TRAINING_DIR / "model_lgbm.txt"
META_FILE = TRAINING_DIR / "model_meta.json"

try:
    from services.state import CACHE_ROOT
except ImportError:  # 直接以脚本方式 import 时兜底
    from state import CACHE_ROOT

# 验证层总开关：B 轮上线（2026-08-25）恢复两级验证（LGBM + 时序模型）
VERIFY_ENABLED = True

# ===== 每场自适应校准（解决跨场馆/机位/光线的分数分布偏移）=====
# 机制：初始判决只自动 ×（保守端），AB 分最高的前 N 个片段标为「待校准」
# （calib='need'），用户确认这 N 个后，用 乐观分位法 估计本场偏移并平移
# 全场分数再触发自动 √ —— 排序能力模型是够的（盲测 AUC 0.81~0.97），
# 缺的只是把分数尺度摆正。
CALIB_N = 5              # 每场送校准的高分片段数
CALIB_MIN_GAIN = 0.05    # 估计的偏移量低于此值视为「无需平移」（训练分布内）

# ===== 全自动模式（UI 开关，2026-08-30 上线）=====
# 适合「快速出粗剪、全程零人工」的场景。金联两节真盲测（34 标注候选）：
#   准确率 91.2% / 进球精确率 100% / 召回 78.6%（灰区默认× 口径）
# 与默认模式的区别：
#   集成分 >= keep_thr → 自动 √（同默认）
#   集成分 <  keep_thr → 自动 ×（灰区直接判×，不等校准）
#   跳过每场自适应校准（不标 calib='need'）
FULL_AUTO = False


def set_full_auto(enabled: bool):
    """UI 开关回调：切换全自动模式（立即生效，影响后续触发的验证）。"""
    global FULL_AUTO
    FULL_AUTO = bool(enabled)
    _log.info("goal_verifier: 全自动模式 %s", "开启" if FULL_AUTO else "关闭")

# 自动标记阈值：score 为冠军集成分（LGBM + B 时序 + Flow 光流 + VideoMAE，
# 各臂 sigmoid 概率加权均值；组合与阈值由 training/train_directions.py
# --deploy 在 OOF 分布上标定，写进 model_temporal_meta.json 的 ensemble 段）。
# 代码里的默认值仅在 meta 缺 ensemble 段时兜底（旧 AB 双路口径）。
# 均可被 training/model_temporal_meta.json 的 ensemble.keep_thr/reject_thr 覆盖。
AB_KEEP_THR = 0.70
REJECT_THR = 0.10
# 各臂权重（缺臂时按剩余权重重归一化，单臂可用时退化为该臂分数）
ENS_WEIGHTS = {"lgbm": 1.0, "b": 1.0, "flow": 1.0, "vm": 1.0}

# B 轮时序模型权重（pool + bigru 双结构 sigmoid 均值）
TEMPORAL_BIGRU_FILE = TRAINING_DIR / "model_temporal.pt"
TEMPORAL_POOL_FILE = TRAINING_DIR / "model_temporal_pool.pt"
TEMPORAL_META_FILE = TRAINING_DIR / "model_temporal_meta.json"
# 冠军集成新增两臂（2026-08-29，training/train_directions.py --deploy 产出）
FLOW_MODEL_FILE = TRAINING_DIR / "model_flow_t.pt"
VM_MODEL_FILE = TRAINING_DIR / "model_vm_lgbm.txt"

# 每视频验证状态（UI 轮询用）：{video_path: {"done": int, "total": int, "running": bool}}
verify_status: dict = {}
_status_lock = threading.Lock()

_model = None          # lightgbm.Booster
_model_tried = False   # 只尝试加载一次，失败不再重试
_model_lock = threading.Lock()
_extract_fn = None     # training/extract_features.py 的 extract()
_feat_names: list = []  # 训练特征名顺序（Booster 内是 Column_N，必须用 meta 的顺序取值）


def _load_model():
    global _model, _model_tried, _feat_names
    if _model_tried:
        return _model
    with _model_lock:
        if _model_tried:
            return _model
        _model_tried = True
        try:
            import json
            from pathlib import Path
            import lightgbm as lgb
            meta = json.loads(Path(META_FILE).read_text(encoding="utf-8"))
            _feat_names = [str(k) for k in meta.get("features", [])]
            _model = lgb.Booster(model_file=MODEL_FILE)
            if len(_feat_names) != _model.num_feature():
                raise ValueError(
                    f"meta 特征数 {len(_feat_names)} != 模型特征数 {_model.num_feature()}")
            _log.info(f"goal_verifier: LGBM 模型已加载 {MODEL_FILE} "
                      f"({len(_feat_names)} 特征)")
        except Exception as e:
            _log.warning(f"goal_verifier: 模型加载失败，自动验证禁用: {e}")
            _model = None
            _feat_names = []
    return _model


def _get_extract():
    """懒加载 training/extract_features.py 的 extract()（脚本非包，用 importlib）。"""
    global _extract_fn
    if _extract_fn is not None:
        return _extract_fn
    try:
        mod_path = str(TRAINING_DIR / "extract_features.py")
        spec = importlib.util.spec_from_file_location("bball_extract_features", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _extract_fn = mod.extract
    except Exception as e:
        _log.warning(f"goal_verifier: extract_features 加载失败: {e}")
        _extract_fn = False
    return _extract_fn or None


# ===== B 轮时序模型（pool+bigru 双结构 sigmoid 均值）=====
_temporal = None          # {"resnet", "nets", "device", "mean", "std", "frame_offs", "crop_hoop"}
_temporal_tried = False
_temporal_lock = threading.Lock()
_train_temporal_mod = None  # training/train_temporal.py（TemporalNet 类定义）
_frames_b_mod = None        # training/extract_frames_b.py（FRAME_OFFS/crop_hoop）


def _load_module(path: str, name: str):
    """importlib 按文件路径加载 training/ 下的脚本（training 非包）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_temporal():
    """懒加载 B 轮推理栈：ResNet18 帧特征 + pool/bigru 双 TemporalNet
    + Flow 光流臂 TemporalNet（共享同一个 ImageNet ResNet18）。

    与训练脚本（train_temporal.py / train_directions.py）完全一致的预处理：
    BGR uint8 帧块 → RGB /255 → ImageNet mean/std。
    Flow 臂权重缺失时置 None（该臂静默禁用，不影响 B 轮打分）。
    失败返回 None（调用方退化为纯 LGBM 分）。
    """
    global _temporal, _temporal_tried, AB_KEEP_THR, REJECT_THR, ENS_WEIGHTS
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
            resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            resnet.fc = torch.nn.Identity()
            resnet = resnet.to(device).eval()

            nets = []
            for p in (TEMPORAL_BIGRU_FILE, TEMPORAL_POOL_FILE):
                ck = torch.load(p, map_location=device, weights_only=False)
                net = _train_temporal_mod.TemporalNet(arch=ck["arch"]).to(device)
                net.load_state_dict(ck["state_dict"])
                net.eval()
                nets.append(net)

            # Flow 臂（可选）：同一 ResNet18 提特征 + 专属 TemporalNet(bigru)
            flow_net = None
            try:
                if FLOW_MODEL_FILE.exists():
                    ck = torch.load(FLOW_MODEL_FILE, map_location=device,
                                    weights_only=False)
                    flow_net = _train_temporal_mod.TemporalNet(
                        dim=ck.get("dim", 512), arch=ck["arch"]).to(device)
                    flow_net.load_state_dict(ck["state_dict"])
                    flow_net.eval()
            except Exception as e:
                _log.warning(f"goal_verifier: Flow 臂加载失败（该臂禁用）: {e}")
                flow_net = None

            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

            # 阈值/权重可被 model_temporal_meta.json 的 ensemble 段覆盖
            # （train_directions.py --deploy 标定后写入）
            try:
                tmeta = json.loads(TEMPORAL_META_FILE.read_text(encoding="utf-8"))
                ens = tmeta.get("ensemble", {})
                AB_KEEP_THR = float(ens.get("keep_thr", AB_KEEP_THR))
                REJECT_THR = float(ens.get("reject_thr", REJECT_THR))
                w = ens.get("weights")
                if isinstance(w, dict) and w:
                    ENS_WEIGHTS = {k: float(v) for k, v in w.items()}
            except Exception:
                pass

            _temporal = {
                "resnet": resnet, "nets": nets, "flow_net": flow_net,
                "device": device, "mean": mean, "std": std,
                "frame_offs": list(_frames_b_mod.FRAME_OFFS),
                "crop_hoop": _frames_b_mod.crop_hoop,
            }
            _log.info(f"goal_verifier: B 轮时序模型已加载 "
                      f"(pool+bigru, flow={'on' if flow_net else 'off'}, "
                      f"device={device}, keep_thr={AB_KEEP_THR}, "
                      f"weights={ENS_WEIGHTS})")
        except Exception as e:
            _log.warning(f"goal_verifier: B 轮模型加载失败，退化为纯 LGBM: {e}")
            _temporal = None
    return _temporal


# ===== VideoMAE 臂（可选，+0.3GB 显存 + transformers 依赖）=====
_vm = None            # {"model", "booster", "device", "dtype", "to_input"}
_vm_tried = False
_vm_lock = threading.Lock()


def _load_vm():
    """懒加载 VideoMAE 臂：冻结主干（fp16 on cuda）+ LGBM 头。

    权重缓存在 cache/hf（extract_videomae.py 下载）；离线加载（HF_HUB_OFFLINE），
    缺依赖/缺缓存时任一失败 → 返回 None，该臂静默禁用。
    输入口径与训练严格一致：(16,224,224,3) uint8 BGR → RGB → (x/255-0.5)/0.5。
    """
    global _vm, _vm_tried
    if _vm_tried:
        return _vm
    with _vm_lock:
        if _vm_tried:
            return _vm
        _vm_tried = True
        try:
            import os
            # 服务进程不允许运行时下载：只用本地缓存（缺缓存 = 臂禁用）
            os.environ.setdefault("HF_HOME", str(Path(CACHE_ROOT) / "hf"))
            os.environ["HF_HUB_OFFLINE"] = "1"
            import lightgbm as lgb

            mod = _load_module(str(TRAINING_DIR / "extract_videomae.py"),
                               "bball_extract_videomae")
            model, device = mod.load_model()
            booster = lgb.Booster(model_file=str(VM_MODEL_FILE))
            _vm = {
                "model": model, "booster": booster, "device": device,
                "dtype": next(model.parameters()).dtype,
                "to_input": mod.to_input,
            }
            _log.info(f"goal_verifier: VideoMAE 臂已加载 "
                      f"(device={device}, dtype={_vm['dtype']})")
        except Exception as e:
            _log.info(f"goal_verifier: VideoMAE 臂不可用（该臂禁用）: {e}")
            _vm = None
    return _vm


_meta_ens_cache = None  # (mtime, keep_thr, reject_thr)


def get_thresholds():
    """当前集成阈值 (keep_thr, reject_thr)。

    优先读 model_temporal_meta.json 的 ensemble 段（--deploy 标定产物，
    文件变更自动失效缓存）；缺 ensemble 段时用代码默认值（旧 AB 口径）。
    供 UI 徽标着色等需要「验证尚未跑就显示阈值」的场景。
    """
    global _meta_ens_cache
    try:
        mtime = TEMPORAL_META_FILE.stat().st_mtime
        if _meta_ens_cache is None or _meta_ens_cache[0] != mtime:
            ens = json.loads(
                TEMPORAL_META_FILE.read_text(encoding="utf-8")).get("ensemble", {})
            _meta_ens_cache = (mtime, float(ens.get("keep_thr", AB_KEEP_THR)),
                               float(ens.get("reject_thr", REJECT_THR)))
        return _meta_ens_cache[1], _meta_ens_cache[2]
    except Exception:
        return AB_KEEP_THR, REJECT_THR


def _hoop_at(hoop_track, ts, default_hoop):
    """按事件时间取篮筐坐标：轨迹里最后一个 ts <= 事件 ts 的位置。

    hoop_track: [{"frame", "ts", "hoop"}]（检测中篮筐移位轨迹，按时间升序）。
    事件发生在移位前 → 用默认标定；移位后 → 用新坐标。
    裁剪错位会直接毁掉 B 轮打分，所以这里必须按时间取，不能全程用一个框。
    """
    if not hoop_track:
        return default_hoop
    best = None
    for t in hoop_track:
        if float(t.get("ts", 0.0)) <= float(ts):
            best = t
        else:
            break  # 轨迹按时间升序，后面只会更大
    return tuple(best["hoop"]) if best else default_hoop


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


def _score_temporal(video_path, clips, hoop, hoop_track=None):
    """三个视觉臂打分（共享同一批 16 帧筐心块解码）。

    就地写 clip['score_b'] / clip['score_flow'] / clip['score_vm']（float 0~1）：
      B    ResNet18 帧特征 → pool+bigru 双 TemporalNet 均值
      Flow _flow_maps → ResNet18 → model_flow_t.pt（bigru）
      VM   VideoMAE 768 维 → model_vm_lgbm.txt（Booster 概率）
    与训练侧抽帧/预处理严格一致（FRAME_OFFS 16 帧 / zoom 3.2 / letterbox 224）。
    片段按 ts 排序顺序解码（单 reader，不逐片段重开视频）。
    hoop_track 非空时按各事件时间取对应的篮筐坐标（检测中移位场景）。
    Flow/VM 臂缺模型或推理失败时该臂静默跳过（集成端按剩余权重重归一化），
    模型/视频不可用时返回 0（保留 LGBM 分）。
    """
    tt = _load_temporal()
    todo = [c for c in clips if "score_b" not in c]
    if tt is None or not todo or not hoop:
        return 0
    import numpy as np
    import torch
    from video_io import VideoReader

    todo.sort(key=lambda c: float(c["ts"]))
    resnet, nets, device = tt["resnet"], tt["nets"], tt["device"]
    mean, std = tt["mean"], tt["std"]
    frame_offs, crop_hoop = tt["frame_offs"], tt["crop_hoop"]
    flow_net = tt.get("flow_net")
    vm = _load_vm()
    n_frames, min_valid = len(frame_offs), 8

    blocks = []  # [(clip, (16,224,224,3) uint8 BGR)]
    reader = None
    try:
        reader = VideoReader(video_path)
        fps, total = reader.fps, reader.total
        for c in todo:
            ts = float(c["ts"])
            frames = np.zeros((n_frames, 224, 224, 3), dtype=np.uint8)
            # offsets 单调递增 → 顺序解码到最后一帧（同 extract_frames_b.main）
            last = max(0, min(int((ts + frame_offs[-1]) * fps), total - 1))
            want = {}
            for i, off in enumerate(frame_offs):
                fidx = max(0, min(int((ts + off) * fps), total - 1))
                want[fidx] = i
            got = 0
            eff_hoop = _hoop_at(hoop_track, ts, hoop)
            for fidx, frame in reader.iter_frames(start=min(want), end=last + 1):
                if fidx in want:
                    frames[want[fidx]] = crop_hoop(frame, eff_hoop)
                    got += 1
                    if got >= n_frames:
                        break
            if got >= min_valid:
                blocks.append((c, frames))
    except Exception as e:
        _log.warning(f"goal_verifier: B 轮抽帧失败 {video_path}: {e}")
        return 0
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass

    # 批量 GPU 推理（4 片段 = 64 帧/批，同训练特征提取的显存预算）
    try:
        for i in range(0, len(blocks), 4):
            chunk = blocks[i:i + 4]
            # ---- B 臂：ResNet18 帧特征 → pool+bigru 均值 ----
            arr = np.concatenate([b[1] for b in chunk]).astype(np.float32) / 255.0
            arr = arr[..., ::-1]  # BGR→RGB
            arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
            with torch.no_grad():
                t = (torch.from_numpy(arr).to(device) - mean) / std
                feat = resnet(t)  # (N*16, 512)
                for j, (c, _) in enumerate(chunk):
                    f = feat[j * n_frames:(j + 1) * n_frames].unsqueeze(0)
                    ps = [float(torch.sigmoid(n(f)).item()) for n in nets]
                    c["score_b"] = round(sum(ps) / len(ps), 3)
            # ---- Flow 臂：光流幅度序列 → 同一 ResNet18 → bigru ----
            if flow_net is not None:
                try:
                    fmap_blocks = [_flow_maps(b[1]) for b in chunk]
                    farr = np.concatenate(fmap_blocks).astype(np.float32) / 255.0
                    farr = np.ascontiguousarray(farr.transpose(0, 3, 1, 2))
                    with torch.no_grad():
                        ft = (torch.from_numpy(farr).to(device) - mean) / std
                        ffeat = resnet(ft)  # (N*15, 512)
                        for j, (c, _) in enumerate(chunk):
                            nf = fmap_blocks[j].shape[0]
                            ff = ffeat[j * nf:(j + 1) * nf].unsqueeze(0)
                            c["score_flow"] = round(
                                float(torch.sigmoid(flow_net(ff)).item()), 3)
                except Exception as e:
                    _log.warning(f"goal_verifier: Flow 臂推理失败（跳过该臂）: {e}")
                    for c, _ in chunk:
                        c.pop("score_flow", None)
            # ---- VM 臂：VideoMAE 768 维 → LGBM Booster ----
            if vm is not None:
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
                    _log.warning(f"goal_verifier: VM 臂推理失败（跳过该臂）: {e}")
                    for c, _ in chunk:
                        c.pop("score_vm", None)
    except Exception as e:
        _log.warning(f"goal_verifier: B 轮推理失败 {video_path}: {e}")
    return sum(1 for c in todo if "score_b" in c)


def needs_verify(clips) -> bool:
    """是否需要触发验证：有片段缺集成分数。"""
    if not VERIFY_ENABLED:
        return False
    if not clips:
        return False
    return any("score" not in c for c in clips)


def verify_clips(video_path, clips, hoop, auto_mark=True, hoop_track=None):
    """对 clips（list[dict]，含 ts）逐个打分：LGBM + B 轮时序模型 → AB 集成。

    就地写 clip['score_lgbm'] / clip['score_b'] / clip['score']（AB 集成，
    float 0~1；B 轮不可用时 score 退化为纯 LGBM）。
    auto_mark 时按阈值写 clip['mark']/'mark_source'（不覆盖人工标记）。
    hoop_track: 检测中篮筐移位轨迹，各阶段按事件时间取对应坐标。
    已有 score 的片段跳过提特征（重复触发不重复花 GPU 时间）。
    返回 (n_scored, n_auto_keep, n_auto_reject)；模型不可用返回 (0, 0, 0)。
    """
    if not VERIFY_ENABLED:
        return 0, 0, 0
    model = _load_model()
    extract = _get_extract()
    if model is None or extract is None or not clips or not hoop:
        # 模型不可用：清掉 start_verify_thread 预置的 running 标记，
        # 否则 UI 进度占位永远等不到「标完」（卡片永不显示）
        with _status_lock:
            verify_status[video_path] = {
                "done": 0, "total": len(clips) if clips else 0,
                "running": False,
                # 全部未打分：UI 据此提示「标记不完整」，避免误当已标完
                "incomplete": len(clips) if clips else 0,
            }
        return 0, 0, 0

    # 只对缺分数的片段提特征
    todo = [c for c in clips if "score" not in c]
    events = []
    clip_by_eid = {}
    if todo:
        from video_io import get_video_info
        try:
            info = get_video_info(video_path)
        except Exception as e:
            _log.warning(f"goal_verifier: 读取视频信息失败 {video_path}: {e}")
            return 0, 0, 0
        for c in todo:
            ts = round(float(c["ts"]), 3)
            eid = f"live_{int(ts * 1000):010d}"
            events.append({
                "event_id": eid, "video": video_path, "ts": ts,
                "hoop": list(_hoop_at(hoop_track, ts, hoop)), "label": 0,
                "video_width": info["width"], "video_height": info["height"],
            })
            clip_by_eid[eid] = c

    with _status_lock:
        verify_status[video_path] = {
            "done": len(clips) - len(todo), "total": len(clips),
            "running": bool(events),
        }
    n_keep = n_reject = 0
    try:
        # ---- 阶段 1：A 轮 LGBM 手工特征打分（写 score_lgbm）----
        if events:
            try:
                from app import get_ball_model, get_ball_class_ids, get_device
                m, weights = get_ball_model()
                ball_classes = get_ball_class_ids(m, weights)
                device = get_device()
                feats, errors = extract(m, ball_classes, device, events)
            except Exception as e:
                import traceback
                _log.warning(f"goal_verifier: 特征提取失败 {video_path}: {e}\n{traceback.format_exc()}")
                feats, errors = [], []

            if errors:
                _log.info(f"goal_verifier: {video_path} 有 {len(errors)} 个事件提取失败（保留人工判断）")

            for f_row in feats:
                c = clip_by_eid.get(f_row.get("event_id"))
                if c is None:
                    continue
                try:
                    # 特征顺序以 meta.features 为准（Booster 内部特征名是 Column_N）
                    row = [float(f_row.get(k, 0.0)) for k in _feat_names]
                    score = float(model.predict([row])[0])
                except Exception:
                    continue
                c["score_lgbm"] = round(min(max(score, 0.0), 1.0), 3)
                with _status_lock:
                    st = verify_status.get(video_path)
                    if st:
                        st["done"] = st.get("done", 0) + 1

        # ---- 阶段 1.5：视觉三臂打分（B/Flow/VM）+ 冠军集成 + 自动标记 ----
        if todo:
            _score_temporal(video_path, todo, hoop, hoop_track=hoop_track)
        for c in todo:
            arms = {"lgbm": c.get("score_lgbm"), "b": c.get("score_b"),
                    "flow": c.get("score_flow"), "vm": c.get("score_vm")}
            avail = [(ENS_WEIGHTS.get(k, 0.0), v) for k, v in arms.items()
                     if v is not None and ENS_WEIGHTS.get(k, 0.0) > 0]
            if avail:
                # 缺臂时按剩余权重重归一化（单臂可用退化为该臂分）
                wsum = sum(w for w, _ in avail)
                c["score"] = round(sum(w * v for w, v in avail) / wsum, 3)

        # ---- 阶段 1.6：自动标记 ----
        # 全自动模式：两端直接判决（灰区默认×），跳过校准
        # 默认模式：每场自适应校准两阶段制——
        #   未校准：只自动 ×（保守端，48/48 盲测零错误），高分前 N 个标 calib='need'
        #   等用户确认 → calibrate_clips() 平移分数 → 已校准的自动 √ 走原逻辑
        #   已校准（calib_shift 已在 clips[0]）：先平移分数再按阈值判决
        if auto_mark and FULL_AUTO:
            for c in todo:
                if "score" not in c or c.get("mark_source") == "manual":
                    continue
                if float(c["score"]) >= AB_KEEP_THR:
                    c["mark"] = "keep"
                    c["mark_source"] = "auto"
                    n_keep += 1
                else:
                    c["mark"] = "reject"
                    c["mark_source"] = "auto"
                    n_reject += 1
        elif auto_mark:
            calib_shift = _get_calib_shift(clips)
            scored = [c for c in todo
                      if "score" in c and c.get("mark_source") != "manual"]
            for c in scored:
                s = float(c["score"])
                if calib_shift is not None:
                    s = min(1.0, s + calib_shift)
                    c["score_calibrated"] = round(s, 3)
                if calib_shift is not None and s >= AB_KEEP_THR:
                    c["mark"] = "keep"
                    c["mark_source"] = "auto"
                    n_keep += 1
                elif s <= REJECT_THR:
                    c["mark"] = "reject"
                    c["mark_source"] = "auto"
                    n_reject += 1
            # 未校准 && 该视频还没有 calib 标记 → 挑最高分 N 个送校准
            if calib_shift is None and not any("calib" in c for c in clips):
                cand = sorted(scored, key=lambda c: -float(c["score"]))[:CALIB_N]
                for c in cand:
                    if REJECT_THR < float(c["score"]) < AB_KEEP_THR:
                        c["calib"] = "need"
    finally:
        with _status_lock:
            st = verify_status.get(video_path)
            if st:
                st["running"] = False
    n_scored = sum(1 for c in clips if "score" in c)
    return n_scored, n_keep, n_reject


# ===== 每场自适应校准 =====

def _get_calib_shift(clips):
    """读本场已确认的校准偏移（存在 clips[0]['calib_shift']，跨次调用持久）。

    None = 未校准。校准信息挂在第一个 clip 上是因为 clips 列表由调用方
    （detection.py）持有并跨 UI 刷新存活，视频重跑检测会换新 list，
    校准自然失效 —— 符合「每场重新校准」语义。
    """
    for c in clips:
        if "calib_shift" in c:
            return float(c["calib_shift"])
    return None


def calibrate_clips(clips, confirmed_n_kept):
    """用户确认完前 N 个高分片段后调用：估计本场偏移并平移全场分数。

    confirmed_n_kept: (已确认进球数 n, 确认总数 N)——前 N 个高分片段里
    用户判 n 个为真进球。

    估计方法（乐观分位法）：假设模型排序大体正确，则本场第 (N-n) 名
    分数附近是「真进球的尾巴」。训练分布里真进球大约能到 0.9+，
    平移量 = 0.90 - 本场第 (N-n) 高分，使确认过的真进球大多越过 √ 阈值。
    保守规则：只向上平移（分数下移是分布偏移的常态；上移=模型过 pessimistic
    罕见且危险），且封顶 0.25 防极端值；n=0（前 N 全不是进球）不平移。
    返回估计的偏移量（未校准/无需平移返回 None）。
    """
    scored = [c for c in clips if "score" in c]
    if not scored:
        return None
    # 清掉「待校准」标记（本次校准动作消化它们）
    n_need = sum(1 for c in clips if c.get("calib") == "need")
    for c in clips:
        c.pop("calib", None)

    n_kept, n_total = confirmed_n_kept
    if not n_total or n_kept <= 0 or n_need == 0:
        # 前几名全否 → 本场分布与模型严重不符，不敢平移；无待校准则无事可做
        if n_total and n_kept == 0:
            _log.info("goal_verifier: 校准放弃（前 %d 个高分片段全部被否）", n_total)
        return None

    ss = sorted((float(c["score"]) for c in scored), reverse=True)
    # 第 (n_total - n_kept) 名 = 乐观假设下真进球的分数下界
    idx = min(n_total - n_kept, len(ss) - 1)
    tail = ss[idx]
    shift = min(0.90 - tail, 0.25)  # 封顶 0.25
    if shift < CALIB_MIN_GAIN:
        _log.info("goal_verifier: 校准判定无需平移 (shift=%.3f)", shift)
        return None
    shift = round(shift, 3)
    scored[0]["calib_shift"] = shift
    _log.info("goal_verifier: 本场校准偏移 +%.3f（尾巴分 %.3f → 目标 0.90）",
              shift, tail)
    return shift


def apply_calibration(video_path, clips, hoop):
    """校准后重判：平移分数 → 重算自动 √/×（不碰人工标记）。

    返回 (n_keep, n_reject)。
    """
    shift = _get_calib_shift(clips)
    if shift is None:
        return 0, 0
    n_keep = n_reject = 0
    for c in clips:
        if "score" not in c or c.get("mark_source") == "manual":
            continue
        s = min(1.0, float(c["score"]) + shift)
        c["score_calibrated"] = round(s, 3)
        if s >= AB_KEEP_THR:
            if c.get("mark") != "keep":
                c["mark"] = "keep"
                c["mark_source"] = "auto"
            n_keep += 1
        elif s <= REJECT_THR:
            # 平移只抬分不会降分，这里只可能是本来就低的
            if c.get("mark") != "reject":
                c["mark"] = "reject"
                c["mark_source"] = "auto"
            n_reject += 1
        else:
            # 灰区：清掉旧的自动标记（平移后可能从 keep 掉回灰区）
            if c.get("mark_source") == "auto":
                c.pop("mark", None)
                c.pop("mark_source", None)
    _log.info("goal_verifier: 校准重判完成 +%s → 自动√ %d / 自动× %d",
              shift, n_keep, n_reject)
    return n_keep, n_reject


def start_verify_thread(video_path, clips, hoop, hoop_track=None):
    """后台线程跑 verify_clips（检测完成后调用，不阻塞主流程）。

    hoop_track: 检测中篮筐移位轨迹（None = 全程一个标定位置）。
    同一视频重复触发时只保留一个（running 中直接返回，避免双跑重复花 GPU）。
    """
    if not clips:
        return
    with _status_lock:
        st = verify_status.get(video_path)
        if st and st.get("running"):
            return
        # 同步预置 running=True：UI 依赖它在验证期间显示进度占位，
        # 全部标完（running=False）才一次性渲染片段卡片
        verify_status[video_path] = {
            "done": 0, "total": len(clips), "running": True,
        }

    def _worker():
        try:
            verify_clips(video_path, clips, hoop, hoop_track=hoop_track)
            # 完成摘要（旧实现静默返回，打分失败/部分失败无从归因）
            n_unscored = sum(1 for c in clips if "score" not in c)
            n_keep = sum(1 for c in clips if c.get("mark") == "keep")
            n_reject = sum(1 for c in clips if c.get("mark") == "reject")
            _log.info(f"goal_verifier: 验证完成 {Path(video_path).name} "
                      f"打分 {len(clips) - n_unscored}/{len(clips)} | √ {n_keep} × {n_reject}")
            if n_unscored:
                _log.warning(f"goal_verifier: {Path(video_path).name} "
                             f"有 {n_unscored} 个片段未完成打分（自动标记不完整）")
            with _status_lock:
                st = verify_status.get(video_path)
                if st:
                    st["incomplete"] = n_unscored
            # 分数/标记回写缓存与历史：否则回读路径拿到无分副本，
            # 每次都重跑整轮验证（~5 分钟/视频）
            try:
                from services import state
                state.update_clip_cache_marks(video_path, clips)
                state.update_history_marks(video_path, clips)
            except Exception as e:
                _log.warning(f"goal_verifier: 标记回写失败（不影响本次结果）: {e}")
        except Exception:
            import traceback
            _log.warning(f"goal_verifier: 后台验证异常 {video_path}\n{traceback.format_exc()}")
            # 异常兜底清标记：否则 UI 进度占位永远等不到「标完」
            with _status_lock:
                st = verify_status.get(video_path)
                if st:
                    st["running"] = False

    threading.Thread(target=_worker, daemon=True,
                     name=f"goal-verify-{threading.current_thread().name}").start()
