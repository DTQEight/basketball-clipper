"""篮球检测核心逻辑库（YOLO 球检测 / 帧读取 / 标注绘制）。

被 demo_nicegui.py 等前端导入使用。
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# ============ 缓存目录（跨平台）============
if os.environ.get("BBALL_CACHE_ROOT"):
    _CACHE_ROOT = os.environ["BBALL_CACHE_ROOT"]
elif os.name == "nt":
    _CACHE_ROOT = r"E:\basketball-project\cache"
else:
    _CACHE_ROOT = os.path.join(os.path.expanduser("~"), "basketball-project", "cache")
os.environ["MPLCONFIGDIR"] = os.path.join(_CACHE_ROOT, "matplotlib")
os.environ["ULTRALYTICS_CONFIG_DIR"] = os.path.join(_CACHE_ROOT, "ultralytics")
os.environ["TORCH_HOME"] = os.path.join(_CACHE_ROOT, "torch")
for _d in [os.environ[k] for k in ["MPLCONFIGDIR",
          "ULTRALYTICS_CONFIG_DIR", "TORCH_HOME"]]:
    os.makedirs(_d, exist_ok=True)

# 公共模块
from video_io import read_frame


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
_video_state = {"path": None, "total": 0, "fps": 30.0, "codec": "unknown"}
_ball_conf = 0.30  # YOLO 球检测置信度阈值（与 UI 滑块默认值对齐）
_calib = {
    "clicks": [],   # 待确认的点击点 [(x,y), ...]
    "hoop": None,   # (x1,y1,x2,y2)
    "baseline_frame_idx": None,  # 基准帧号（无球的篮筐画面）
}


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


def list_weights():
    wdir = ROOT / "weights"
    wdir.mkdir(exist_ok=True)
    pts = sorted(wdir.glob("*.pt"))
    return [str(p) for p in pts] if pts else [str(wdir / "yolov8n.pt")]


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


def detect_ball_yolo(frame, conf=None, imgsz=1280, augment=False):
    """用 YOLO 检测球，返回 (cx, cy, x1, y1, x2, y2, conf) 或 None。

    兼容 COCO 预训练（sports ball=32）和篮球微调权重（basketball 类）。
    取置信度最高的球框。
    imgsz: 推理分辨率，篮球微调模型训练时用 1280，默认 1280 提升小目标检出。
    augment: TTA（Test Time Augmentation），慢约 3 倍但提升小目标检出率。
    """
    if conf is None:
        conf = _ball_conf
    model, _ = get_ball_model()
    res = model.predict(frame, conf=conf, imgsz=imgsz, device=get_device(),
                        augment=augment, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    names = res.names
    clses = res.boxes.cls.cpu().numpy().astype(int)
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    best = None
    for i, c in enumerate(clses):
        n = names.get(c, "").lower()
        if "ball" in n or "basketball" in n:
            if best is None or confs[i] > confs[best]:
                best = i
    if best is None:
        return None
    x1, y1, x2, y2 = xyxy[best]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (float(cx), float(cy), float(x1), float(y1), float(x2), float(y2), float(confs[best]))


def get_frame(idx):
    """读取指定帧（BGR）。"""
    if _video_state["path"] is None:
        return None
    return read_frame(_video_state["path"], idx,
                      total=_video_state["total"], fps=_video_state["fps"])


def draw_calib(frame, ball_det=None, net_box=None):
    """在帧上画篮筐框（绿）、篮网区域（蓝）和待确认点击点（黄）。

    ball_det: 可选，YOLO 检测到的球 (cx, cy, x1, y1, x2, y2, conf)，画红框。
    net_box: 可选，篮网区域 (x1,y1,x2,y2)，画蓝框。
    """
    out = frame.copy()
    if _calib["hoop"]:
        x1, y1, x2, y2 = _calib["hoop"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(out, "HOOP", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if net_box is not None:
        nx1, ny1, nx2, ny2 = net_box
        cv2.rectangle(out, (int(nx1), int(ny1)), (int(nx2), int(ny2)), (255, 128, 0), 2)
        cv2.putText(out, "NET", (int(nx1), max(int(ny1) - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
    if ball_det is not None:
        _, _, bx1, by1, bx2, by2, bconf = ball_det
        cv2.rectangle(out, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 3)
        cv2.putText(out, f"BALL {bconf:.2f}", (int(bx1), max(int(by1) - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    for i, (x, y) in enumerate(_calib["clicks"]):
        cv2.circle(out, (x, y), 10, (255, 255, 0), -1)
        cv2.putText(out, str(i + 1), (x - 6, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return out
