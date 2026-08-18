"""篮球检测核心逻辑库（YOLO 球检测 / 帧读取 / 标注绘制）。

被 demo_nicegui.py 等前端导入使用。
"""
import logging
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# ============ 缓存目录（跨平台）============
# 复用 services/state.py 的单一实现：各自维护一份环境变量回退逻辑会漂移
from services.state import CACHE_ROOT as _CACHE_ROOT
os.environ["MPLCONFIGDIR"] = os.path.join(_CACHE_ROOT, "matplotlib")
os.environ["ULTRALYTICS_CONFIG_DIR"] = os.path.join(_CACHE_ROOT, "ultralytics")
os.environ["TORCH_HOME"] = os.path.join(_CACHE_ROOT, "torch")
for _d in [os.environ[k] for k in ["MPLCONFIGDIR",
          "ULTRALYTICS_CONFIG_DIR", "TORCH_HOME"]]:
    os.makedirs(_d, exist_ok=True)

_log = logging.getLogger("app")

# 公共模块（当前仅暴露给其他模块的函数在此）


def get_device() -> str:
    """检测可用推理设备，优先 CUDA，回退 CPU。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


# ============ 全局状态 ============
_model_cache = {}
_model_lock = threading.Lock()  # 懒加载锁：双线程同时未命中会重复加载模型


def get_model(weights: str):
    from ultralytics import YOLO
    with _model_lock:
        if weights not in _model_cache:
            m = YOLO(weights)
            # 加载后立即搬到推理设备（与 get_device() 对齐），避免首次推理在 CPU 上跑
            try:
                dev = get_device()
                if dev != "cpu":
                    m.to(dev)
            except Exception as e:
                # 静默 pass 会让 get_device() 结论与模型真实驻地不一致，至少留痕
                _log.warning(f"[WARN] 模型搬运到 {get_device()} 失败，将由推理端自愈: {e}")
            _model_cache[weights] = m
        return _model_cache[weights]


# 球类精确名集合：自定义权重通常为 'basketball'，COCO 预训练为 'sports ball'。
# 必须精确匹配：旧实现用子串 'ball' 匹配，会把 COCO 的 'baseball bat'(39)/
# 'baseball glove'(38) 也当球 → "YOLO 硬否决"退化为"筐边有球棒也确认"。
_BALL_CLASS_NAMES = {"basketball", "sports ball", "ball", "soccer ball",
                     "football", "volleyball", "tennis ball"}


def get_ball_class_ids(model, weights_path: str = "") -> list:
    """按 model.names 反查球类别索引列表。

    classes=[0] 只对 names={0:'basketball'} 的自定义微调权重成立；
    回退 COCO 权重（yolov8n.pt）时类 0 是 person（sports ball 是 32），
    硬编码 [0] 会把"YOLO 硬否决"变成"篮筐附近有人就确认"，误检暴增。
    解析失败返回 []（而非回退 [0]）：names 异常说明权重不可信，
    调用方应拒绝检测而不是把 person 当球确认错误结果。
    """
    try:
        names = model.names or {}
        ids = sorted(i for i, n in names.items()
                     if str(n).strip().lower() in _BALL_CLASS_NAMES)
        if ids:
            return ids
    except Exception:
        pass
    return []


def get_ball_model() -> tuple:
    """懒加载球检测 YOLO 模型。

    优先用 weights/ 下的篮球专用微调权重；否则用 yolov8n.pt（COCO 预训练，自动下载）。
    返回 (model, weights_path)。
    """
    wdir = ROOT / "weights"
    weights = None
    if wdir.exists():
        # 优先选名字含 basketball/ball/finetuned 的权重（篮球专用）
        pts = sorted(wdir.glob("*.pt"),
                     key=lambda p: 0 if any(k in p.name.lower()
                                            for k in ("basketball", "finetuned", "ball"))
                                   else 1)
        if pts:
            weights = str(pts[0])
    if weights is None:
        weights = "yolov8n.pt"
    return get_model(weights), weights
