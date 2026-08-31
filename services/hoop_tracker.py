# -*- coding: utf-8 -*-
"""篮筐跟踪器：标定后周期性用模板匹配确认篮筐位置。

场景：录制中有人撞到支架 → 画面里篮筐整体移位 → 原标定作废。
本模块在检测循环里每 ~5s 用模板匹配校验一次篮筐位置（模板 = 标定帧
篮筐框外扩 50% 的灰度图，含篮板结构）。

加固设计（2026-08-31，针对实战两类失效）：

1) **单次匹配结果不采信**：移位需连续 _CONFIRM_N 次一致测量才提交；
   等待确认期间 self.pending 置位，调用方加密校验频率（~1s 一次），
   通常 1-2 秒内出结果。旧版单次匹配即提交，球员密集场景出现过坐标
   连续数十秒往复乱跳（每次提交都复位进球状态机），整段检测被清零。
2) **模板刷新需区域"干净"**：用当前帧模板区域与现有模板做零偏移相关，
   低于 _CLEAN_THR（有球员/球进入）或连续稳定次数不足时不刷新。旧版
   无条件用当前帧刷新，撞筐瞬间筐下往往有人，模板混入球员纹理后
   匹配锁人 → 连锁乱跳。
3) **跟踪丢失自动恢复**：连续 _RECOVER_AFTER 次匹配失败后，用放宽阈值
   的多尺度全图搜索重新定位，结果进入同一确认流程（防跳到相似结构）。
   旧版丢失后没有恢复路径，抱着旧坐标直到视频结束。

匹配在 1/scale 的低分辨率下进行（模板高度归一到 96px），
1080p 单次校验 ~几 ms，每 5s 一次对检测 throughput 无可测影响。
"""
from __future__ import annotations

import cv2

_PAD = 0.5           # 模板在篮筐框基础上的四边外扩比例（带上网/篮板结构更好认）
_TPL_H = 96.0        # 模板归一化高度（px；帧同步缩放，匹配成本与分辨率解耦）
_LOCAL_THR = 0.55    # 局部窗口内视为匹配成功的最低分
_FULL_THR = 0.60     # 全图搜索的最低分（误匹配风险更高，门槛更高）
_RECOVER_THR = 0.50  # 连续丢失后恢复搜索的放宽阈值（结果仍需确认才采信）
_REFRESH_THR = 0.80  # 分数高于此值才考虑刷新模板（光线渐变自适应）
_CLEAN_THR = 0.85    # 模板刷新零偏移相关下限：低于它说明区域有遮挡物，禁止刷新
_CONFIRM_N = 2       # 移位需连续多少次一致测量才提交（防单帧误跳）
_AGREE_FRAC = 0.4    # 两次测量一致性半径 = max(16px, 0.4*篮筐框长边)
_STABLE_FOR_REFRESH = 2   # 连续稳定匹配多少次后才允许刷新模板（提交后不立刻刷）
_RECOVER_AFTER = 3   # 连续丢失多少次后启动恢复搜索
_SCALES = (1.0, 0.85, 1.15)  # 多尺度搜索（仅全图兜底/恢复，应对轻微变焦）


class HoopTracker:
    """篮筐位置跟踪器（模板匹配，非线程安全：检测循环内单线程使用）。"""

    def __init__(self, template_frame, hoop):
        """template_frame: 标定帧（BGR 整帧）；hoop: (x1, y1, x2, y2) 标定框。"""
        x1, y1, x2, y2 = (float(v) for v in hoop)
        self._w = max(8.0, x2 - x1)
        self._h = max(8.0, y2 - y1)
        self._pad = _PAD * max(self._w, self._h)
        self._box = (x1, y1, x1 + self._w, y1 + self._h)   # 最近已确认篮筐框
        self._scale = _TPL_H / max(1.0, self._h + 2 * self._pad)
        self._agree_r = max(16.0, _AGREE_FRAC * max(self._w, self._h))
        self.lost_streak = 0     # 连续跟踪丢失次数（诊断用）
        self.n_moves = 0         # 累计已确认移位次数
        # 移位待确认：单次大幅移位只当候选，连续一致测量后才提交
        self._pending_box = None
        self._pending_count = 0
        self._stable_streak = 0  # 连续稳定匹配次数（模板刷新的前置门槛）
        self._max_lost = 0
        self.stats = {"checks": 0, "local": 0, "full": 0, "recover": 0,
                      "lost": 0, "candidates": 0, "committed": 0,
                      "refreshes": 0}
        self._hoop_off = (0.0, 0.0)
        self._tpl = None         # (模板图, (ox, oy), (裁剪原宽, 裁剪原高))
        self._refresh_template(template_frame)

    # ---------- 对外 ----------

    @property
    def pending(self) -> bool:
        """有移位候选待确认（调用方应加密校验频率，尽快给出结论）。"""
        return self._pending_box is not None

    @property
    def box(self):
        """最近已确认篮筐框 (x1, y1, x2, y2)。"""
        return self._box

    def summary(self) -> str:
        """运行统计一行摘要（写入结束日志，便于事后定位跟踪问题）。"""
        st = self.stats
        return (f"checks={st['checks']} (local={st['local']} full={st['full']} "
                f"recover={st['recover']} lost={st['lost']}) "
                f"candidates={st['candidates']} committed={st['committed']} "
                f"refreshes={st['refreshes']} max_lost_streak={self._max_lost}")

    def update(self, frame):
        """校验当前帧的篮筐位置。

        返回 None（未移位 / 候选待确认 / 跟踪丢失）或移位事件：
          {"hoop": [x1, y1, x2, y2], "score": 匹配分, "shift_px": 位移,
           "confirm": 连续一致确认次数}
        移位需 _CONFIRM_N 次连续一致测量才提交；待确认期间
        self.pending 为 True。跟踪丢失时 self.lost_streak 递增，
        连续丢失后自动进入多尺度恢复搜索。
        """
        if frame is None or not len(frame) or self._tpl is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        fh, fw = gray.shape[:2]
        self.stats["checks"] += 1

        score, m, mode = self._locate(gray, fw, fh)
        if m is None:
            self.lost_streak += 1
            self._max_lost = max(self._max_lost, self.lost_streak)
            self.stats["lost"] += 1
            self._pending_reset()
            return None

        self.stats[mode] += 1
        self.lost_streak = 0
        # 模板框位置 → 篮筐框位置（偏移量在 _refresh_template 里维护）
        off_x, off_y = self._hoop_off
        nx1, ny1 = m[0] + off_x, m[1] + off_y
        new_box = (nx1, ny1, nx1 + self._w, ny1 + self._h)
        shift = ((nx1 - self._box[0]) ** 2 + (ny1 - self._box[1]) ** 2) ** 0.5
        move_thr = max(15.0, 0.02 * fw)

        if shift < move_thr:
            # 稳定：无移位；连续稳定且区域干净时刷新模板（光线渐变自适应）
            self._pending_reset()
            self._stable_streak += 1
            if (score >= _REFRESH_THR
                    and self._stable_streak >= _STABLE_FOR_REFRESH
                    and self._region_clean(gray)):
                self._refresh_template(frame)
            return None

        # 移位候选：连续一致测量后才提交（防单帧误跳）
        self._stable_streak = 0
        self.stats["candidates"] += 1
        if self._pending_box is not None:
            d = ((nx1 - self._pending_box[0]) ** 2
                 + (ny1 - self._pending_box[1]) ** 2) ** 0.5
            self._pending_count = self._pending_count + 1 if d <= self._agree_r else 1
        else:
            self._pending_count = 1
        self._pending_box = new_box

        if self._pending_count < _CONFIRM_N:
            return None   # 待确认：调用方看到 self.pending → 加密校验频率

        # 连续一致确认通过：提交移位
        confirm_n = self._pending_count
        self._box = new_box
        self.n_moves += 1
        self.stats["committed"] += 1
        self._pending_reset()
        # 在新位置重裁模板（旧模板对应旧场景）；刷新节奏之后仍受
        # _STABLE_FOR_REFRESH 门槛约束，避免提交瞬间被遮挡物污染
        self._refresh_template(frame)
        self._stable_streak = 0
        return {"hoop": [int(round(v)) for v in new_box],
                "score": round(score, 3),
                "shift_px": round(shift, 1),
                "confirm": confirm_n}

    # ---------- 内部 ----------

    def _pending_reset(self):
        self._pending_box = None
        self._pending_count = 0

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
        self.stats["refreshes"] += 1

    def _region_clean(self, gray) -> bool:
        """当前帧模板区域与现有模板的零偏移相关 ≥ _CLEAN_THR 视为"干净"。

        低于阈值 = 有球员/球等遮挡物进入区域，此时刷新模板会把遮挡物
        混进模板（污染），之后匹配会锁着遮挡物 → 禁止刷新。
        """
        if self._tpl is None:
            return False
        tpl = self._tpl[0]
        cx1, cy1, cx2, cy2 = self._tpl_box()
        cx2, cy2 = min(cx2, gray.shape[1]), min(cy2, gray.shape[0])
        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            return False
        crop = gray[cy1:cy2, cx1:cx2]
        crop_s = cv2.resize(crop, (tpl.shape[1], tpl.shape[0]),
                            interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(crop_s, tpl, cv2.TM_CCOEFF_NORMED)
        return bool(r[0][0] >= _CLEAN_THR)

    def _match(self, gray, region, scales=(1.0,)):
        """在 region（原坐标 x1,y1,x2,y2，None=全图）里找模板，多尺度取最高分。

        返回 (score, 模板框原坐标位置) 或 (0, None)。
        """
        tpl_b, _, (tw, th) = self._tpl
        best_score, best_m = 0.0, None
        if region is None:
            sub, sx, sy = gray, 0, 0
        else:
            rx1, ry1, rx2, ry2 = region
            rx2, ry2 = min(rx2, gray.shape[1]), min(ry2, gray.shape[0])
            if rx2 - rx1 < 16 or ry2 - ry1 < 16:
                return 0.0, None
            sub, sx, sy = gray[ry1:ry2, rx1:rx2], rx1, ry1
        for sc in scales:
            if sc == 1.0:
                tpl = tpl_b
            else:
                tpl = cv2.resize(tpl_b,
                                 (max(8, int(round(tpl_b.shape[1] * sc))),
                                  max(8, int(round(tpl_b.shape[0] * sc)))),
                                 interpolation=cv2.INTER_AREA)
            s = self._scale * sc
            sub_s = cv2.resize(sub, (max(2, int(round(sub.shape[1] * s))),
                                     max(2, int(round(sub.shape[0] * s)))),
                               interpolation=cv2.INTER_AREA)
            if sub_s.shape[0] < tpl.shape[0] or sub_s.shape[1] < tpl.shape[1]:
                continue
            res = cv2.matchTemplate(sub_s, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score > best_score:
                # 缩放坐标 → 原坐标：模板框在原坐标下的宽高 = tw、th（与 sc 无关）
                mx = sx + loc[0] / max(s, 1e-6)
                my = sy + loc[1] / max(s, 1e-6)
                best_score, best_m = float(score), (mx, my, mx + tw, my + th)
        return best_score, best_m

    def _locate(self, gray, fw, fh):
        """定位模板。返回 (score, 模板框位置, 模式) 或 (0, None, None)。

        层级：局部搜索 → 全图多尺度（带距离上限，防跳对面篮筐）→
        连续丢失后的恢复搜索（放宽阈值、不限距离，结果仍走确认流程）。
        """
        cx = (self._box[0] + self._box[2]) / 2
        cy = (self._box[1] + self._box[3]) / 2
        r = max(0.4 * max(fw, fh), 160)
        region = (max(0, int(cx - r)), max(0, int(cy - r)),
                  min(fw, int(cx + r)), min(fh, int(cy + r)))
        score, m = self._match(gray, region)
        if m is not None and score >= _LOCAL_THR:
            return score, m, "local"
        score, m = self._match(gray, None, scales=_SCALES)
        if m is not None:
            if score >= _FULL_THR:
                # 全图匹配可能跳到远处相似物：限制位移上限
                ncx, ncy = (m[0] + m[2]) / 2, (m[1] + m[3]) / 2
                if ((ncx - cx) ** 2 + (ncy - cy) ** 2) ** 0.5 <= 0.5 * (fw ** 2 + fh ** 2) ** 0.5:
                    return score, m, "full"
            # 恢复路径：连续丢失后，或正在验证移位候选时（候选可能离原位
            # 很远、过不了上面的距离上限）；放宽阈值、不限距离，结果仍走
            # 确认流程，不会单次采信
            if (score >= _RECOVER_THR
                    and (self.pending or self.lost_streak + 1 >= _RECOVER_AFTER)):
                return score, m, "recover"
        return 0.0, None, None
