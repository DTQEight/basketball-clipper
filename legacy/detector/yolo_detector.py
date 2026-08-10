"""YOLOv8 检测模块：替代原项目的 TF1 + frozen graph。

替代原 basketball-shot-detection 的 tensorflow_init() 与 sess.run() 检测部分，
返回统一的检测结果格式，供下游 shot_judge 使用。
"""
import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, weights, device="cuda:0", conf=0.4):
        """加载 YOLOv8 模型。

        weights: 权重文件路径（.pt）
        device:  推理设备，GTX 1650 用 "cuda:0"
        conf:    置信度阈值
        """
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf

    def detect(self, frame):
        """检测一帧。

        返回: [(cls, x1, y1, x2, y2, conf), ...]
        cls 为类别索引（0=basketball, 1=hoop，依 config 顺序）
        """
        results = self.model.predict(
            frame, conf=self.conf, device=self.device, verbose=False
        )
        dets = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for cls, (x1, y1, x2, y2), conf in zip(clses, xyxy, confs):
                dets.append((int(cls), float(x1), float(y1),
                             float(x2), float(y2), float(conf)))
        return dets

    def detect_best(self, frame, target_cls):
        """返回指定类别置信度最高的单个框 [x1,y1,x2,y2,conf]，无则 None。"""
        dets = self.detect(frame)
        best = None
        for cls, x1, y1, x2, y2, conf in dets:
            if cls != target_cls:
                continue
            if best is None or conf > best[4]:
                best = [x1, y1, x2, y2, conf]
        return best
