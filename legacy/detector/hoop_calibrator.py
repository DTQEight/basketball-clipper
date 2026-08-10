"""固定机位篮筐位置标定。

固定机位下篮筐位置基本不变，无需每帧检测。前 N 帧采样取中位数后锁定，
后续直接复用坐标，既稳定又能省算力。
"""
import numpy as np


class HoopCalibrator:
    def __init__(self, manual_box=None, calibrate_frames=30):
        """manual_box: [x1,y1,x2,y2]，给了就直接用，不再自动标定。
        calibrate_frames: 自动标定时采样多少帧取中位数。
        """
        self.manual_box = manual_box
        self.calibrate_frames = calibrate_frames
        self.samples = []
        self.fixed = None

    def update(self, hoop_boxes):
        """喂入一帧的篮筐检测结果（[x1,y1,x2,y2,...] 列表），返回当前篮筐坐标。

        返回 [x1,y1,x2,y2] 或 None（尚未标定完成）。
        """
        if self.manual_box is not None:
            self.fixed = list(self.manual_box)
            return self.fixed

        if self.fixed is not None:
            return self.fixed

        for b in hoop_boxes:
            self.samples.append(b[:4])

        if len(self.samples) >= self.calibrate_frames:
            arr = np.array(self.samples)
            self.fixed = np.median(arr, axis=0).tolist()
        return self.fixed

    @property
    def ready(self):
        return self.fixed is not None
