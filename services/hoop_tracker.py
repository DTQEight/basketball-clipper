# -*- coding: utf-8 -*-
"""篮筐跟踪器：标定后周期性用模板匹配确认篮筐位置。

场景：录制中有人撞到支架 → 画面里篮筐整体移位 → 原标定作废，
后续整段检测全错。本模块在检测循环里每 ~5s 用模板匹配校验一次
篮筐位置（模板 = 标定帧篮筐框外扩 50% 的灰度图，含篮板结构）：

  - 位置未变：无事发生（分数高时顺手刷新模板，适应光线渐变）
  - 位置移位：返回移动事件，调用方更新检测器坐标并记录轨迹
  - 跟踪丢失（遮挡/变焦）：保持原位置继续，计数供诊断

匹配在 1/scale 的低分辨率下进行（模板高度归一到 96px），
1080p 单次校验 ~几 ms，每 5s 一次对检测吞吐无可测影响。
"""
from __future__ import annotations

import cv2
import numpy as np

_PAD = 0.5           # 模板在篮筐框基础上的四边外扩比例（带上网/篮板结构更好认）
_TPL_H = 96.0        # 模板归一化高度（px；帧同步缩放，匹配成本与分辨率解耦）
_LOCAL_THR = 0.55    # 局部窗口内视为匹配成功的最低分
_FULL_THR = 0.60     # 全图搜索的最低分（误匹配风险更高，门槛更高）
_REFRESH_THR = 0.80  # 分数高于此值时用当前帧刷新模板（光线渐变自适应）


class HoopTracker:
    """篮筐位置跟踪器（模板匹配，非线程安全：检测循环内单线程使用）。"""

    def __init__(self, template_frame, hoop):
        """template_frame: 标定帧（BGR 整帧）；hoop: (x1, y1, x2, y2) 标定框。"""
        x1, y1, x2, y2 = (float(v) for v in hoop)
        self._w = max(8.0, x2 - x1)
        self._h = max(8.0, y2 - y1)
        self._pad = _PAD * max(self._w, self._h)
        self._box = (x1, y1, x1 + self._w, y1 + self._h)   # 最近已知篮筐框
        self._scale = _TPL_H / max(1.0, self._h + 2 * self._pad)
        self.lost_streak = 0   # 连续跟踪丢失次数（诊断用）
        self.n_moves = 0       # 累计移位次数
        self._tpl = None       # (模板图, 模板框在原坐标下的 (ox, oy))
        self._refresh_template(template_frame)

    # ---------- 内部 ----------

    def _tpl_box(self, box=None):
        """模板裁剪框（原坐标）：篮筐框四边外扩 pad，裁剪到画面内。"""
        b = box or self._box
        x1, y1, x2, y2 = b
        cx1 = max(0, int(round(x1 - self._pad)))
        cy1 = max(0, int(round(y1 - self._pad)))
        cx2 = int(round(x2 + self._pad))
        cy2 = int(round(y2 + self._pad))
        return cx1, cy1, cx2, cy2

    def _refresh_template(self, frame):
        """用 frame 在当前篮筐位置重新裁模板。"""
        if frame is None or not len(frame):
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        cx1, cy1, cx2, cy2 = self._tpl_box()
        cx2, cy2 = min(cx2, w), min(cy2, h)
        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            return
        crop = gray[cy1:cy2, cx1:cx2]
        s = self._scale
        tpl = cv2.resize(crop, (max(8, int(round((cx2 - cx1) * s))),
                                max(8, int(round((cy2 - cy1) * s)))),
                         interpolation=cv2.INTER_AREA)
        self._tpl = (tpl, (cx1, cy1), (cx2 - cx1, cy2 - cy1))
        # 模板刷新后篮筐偏移量同步（裁剪被画面边缘截断时偏移会变）
        self._hoop_off = (self._box[0] - cx1, self._box[1] - cy1)

    def _match(self, gray, region):
        """在 region（原坐标 x1,y1,x2,y2，None=全图）里找模板。

        返回 (score, 模板框原坐标位置) 或 (0, None)。
        """
        tpl, (ox, oy), (tw, th) = self._tpl
        if region is None:
            sub, sx, sy = gray, 0, 0
        else:
            rx1, ry1, rx2, ry2 = region
            rx2, ry2 = min(rx2, gray.shape[1]), min(ry2, gray.shape[0])
            if rx2 - rx1 < tw + 1 or ry2 - ry1 < th + 1:
                return 0.0, None
            sub, sx, sy = gray[ry1:ry2, rx1:rx2], rx1, ry1
        # 搜索图同步降采样（与模板同尺度）
        s = self._scale
        sub_s = cv2.resize(sub, (max(2, int(round(sub.shape[1] * s))),
                                 max(2, int(round(sub.shape[0] * s)))),
                           interpolation=cv2.INTER_AREA)
        th_s, tw_s = tpl.shape[:2]
        if sub_s.shape[0] < th_s or sub_s.shape[1] < tw_s:
            return 0.0, None
        res = cv2.matchTemplate(sub_s, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        # 缩放坐标 → 原坐标：sub_s 的 (0,0) 对应 sub 的 (sx, sy)
        mx = sx + loc[0] / max(s, 1e-6)
        my = sy + loc[1] / max(s, 1e-6)
        return float(score), (mx, my, mx + tw, my + th)

    # ---------- 对外 ----------

    def update(self, frame):
        """校验当前帧的篮筐位置。

        返回 None（未移位/跟踪丢失，保持原位置）或移位事件：
          {"hoop": [x1, y1, x2, y2], "score": 匹配分, "shift_px": 位移像素}
        跟踪丢失时 self.lost_streak 递增（连续丢失会在调用方日志里报警）。
        """
        if frame is None or not len(frame) or self._tpl is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        fh, fw = gray.shape[:2]

        # 1) 局部搜索：上次位置周围 ±40% 画面（撞支架的位移量级以内）
        cx = (self._box[0] + self._box[2]) / 2
        cy = (self._box[1] + self._box[3]) / 2
        r = max(0.4 * max(fw, fh), 160)
        _, (ox, oy), (tw, th) = self._tpl
        region = (max(0, int(cx - r)), max(0, int(cy - r)),
                  min(fw, int(cx + r)), min(fh, int(cy + r)))
        score, m = self._match(gray, region)
        if m is None or score < _LOCAL_THR:
            # 2) 全图搜索兜底（局部失败 = 大幅移位或模板老化）
            score, m = self._match(gray, None)
            if m is None or score < _FULL_THR:
                self.lost_streak += 1
                return None
            # 全图匹配可能跳到远处相似物（如对面篮筐）：限制位移上限
            ncx, ncy = (m[0] + m[2]) / 2, (m[1] + m[3]) / 2
            if ((ncx - cx) ** 2 + (ncy - cy) ** 2) ** 0.5 > 0.5 * (fw ** 2 + fh ** 2) ** 0.5:
                self.lost_streak += 1
                return None

        self.lost_streak = 0
        # 模板框位置 → 篮筐框位置（偏移量在 _refresh_template 里维护）
        off_x, off_y = self._hoop_off
        nx1, ny1 = m[0] + off_x, m[1] + off_y
        new_box = (nx1, ny1, nx1 + self._w, ny1 + self._h)
        shift = ((nx1 - self._box[0]) ** 2 + (ny1 - self._box[1]) ** 2) ** 0.5
        move_thr = max(15.0, 0.02 * fw)

        if shift < move_thr:
            # 未移位：分数高时刷新模板（光线渐变自适应）
            if score >= _REFRESH_THR:
                self._refresh_template(frame)
            return None

        # 移位：更新位置 + 用当前帧重裁模板（撞后视角/光线可能变化）
        self._box = new_box
        self.n_moves += 1
        self._refresh_template(frame)
        return {"hoop": [int(round(v)) for v in new_box],
                "score": round(score, 3), "shift_px": round(shift, 1)}

    @property
    def box(self):
        """最近已知篮筐框 (x1, y1, x2, y2)。"""
        return self._box
