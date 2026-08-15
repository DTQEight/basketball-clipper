"""篮球检测核心逻辑库（YOLO 球检测 / 帧读取 / 标注绘制）。

被 demo_nicegui.py 等前端导入使用。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# ============ 缓存目录（跨平台）============
if os.environ.get("BBALL_CACHE_ROOT"):
    _CACHE_ROOT = os.environ["BBALL_CACHE_ROOT"]
else:
    # 与 services/state.py 保持一致：basketball-project/cache（项目父目录）
    _CACHE_ROOT = str(ROOT.parent / "cache")
os.environ["MPLCONFIGDIR"] = os.path.join(_CACHE_ROOT, "matplotlib")
os.environ["ULTRALYTICS_CONFIG_DIR"] = os.path.join(_CACHE_ROOT, "ultralytics")
os.environ["TORCH_HOME"] = os.path.join(_CACHE_ROOT, "torch")
for _d in [os.environ[k] for k in ["MPLCONFIGDIR",
          "ULTRALYTICS_CONFIG_DIR", "TORCH_HOME"]]:
    os.makedirs(_d, exist_ok=True)

# 公共模块（当前仅暴露给其他模块的函数在此）


def get_device():
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


def get_model(weights):
    from ultralytics import YOLO
    if weights not in _model_cache:
        m = YOLO(weights)
        # 加载后立即搬到推理设备（与 get_device() 对齐），避免首次推理在 CPU 上跑
        try:
            dev = get_device()
            if dev != "cpu":
                m.to(dev)
        except Exception:
            pass
        _model_cache[weights] = m
    return _model_cache[weights]


def get_ball_model():
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
