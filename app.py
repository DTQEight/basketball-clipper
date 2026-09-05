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
# 负缓存（B5）：某权重加载/初始化失败后，同一进程内后续调用直接重抛，
# 不再反复执行 YOLO(weights) 的慢加载/联网下载——错误不会自行消失，
# 每次重试都在浪费数秒~数分钟并刷屏日志
_model_fail_cache = {}
_model_lock = threading.Lock()  # 懒加载锁：双线程同时未命中会重复加载模型


def get_model(weights: str):
    """加载（或取缓存）YOLO 模型；失败时抛带上下文信息的 RuntimeError。

    B4：加载/设备迁移失败必须显式报错——旧实现迁移失败只 log warning，
    模型实际留在 cpu，而 get_device() 声称 cuda，随后 predict 每帧失败、
    空跑数分钟才被 YOLO 熔断拦下，用户无从定位根因。调用方（自检横幅/
    检测入口）会把异常转成用户可见的明确错误。
    """
    from ultralytics import YOLO
    with _model_lock:
        if weights in _model_cache:
            return _model_cache[weights]
        if weights in _model_fail_cache:
            raise _model_fail_cache[weights]   # 复用首次异常（含完整上下文）
        try:
            m = YOLO(weights)
            dev = get_device()
            if dev != "cpu":
                # 搬运失败抛错而不是吞掉：模型驻地与 get_device() 结论分裂时，
                # 尽早报错比等检测空跑数十分钟后熔断更容易定位
                m.to(dev)
            _model_cache[weights] = m
            return m
        except Exception as e:
            err = RuntimeError(f"YOLO 模型加载失败（{weights}）: {e}")
            _model_fail_cache[weights] = err
            raise err


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
        # 只选名字含 basketball/finetuned/ball 的篮球专用权重。
        # 旧实现无关键字时盲选第一个 .pt：类别表若不含球类，
        # get_ball_class_ids 返回 [] 直接拒绝检测；回退 COCO 权重反而可用
        pts = [p for p in sorted(wdir.glob("*.pt"))
               if any(k in p.name.lower()
                      for k in ("basketball", "finetuned", "ball"))]
        if pts:
            weights = str(pts[0])
    if weights is None:
        weights = "yolov8n.pt"
    return get_model(weights), weights
