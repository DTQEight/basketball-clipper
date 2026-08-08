"""进球检测器：基准帧差法 + 连通域分析 + 音频峰值融合（固定机位专用）。

算法参考论文：Camera-based Basketball Scoring Detection Using CNN
实践方案参考：CSDN 神投手（lyandgh）+ basketball-highlights 多信号融合

工作原理：
  1. 标定时取一帧无球的画面作为基准帧
  2. 每帧与基准帧做灰度差分 → 二值化 → 连通域分析
  3. 在篮筐周边区域找最大连通域 = 运动的篮球
  4. 跟踪斑块：斑块先经过篮筐上沿 → 后经过篮筐下沿 = 视觉进球
  5. 融合阶段：视觉进球 ± 时间窗口 内有音频峰值 = 高置信度进球

优势（针对固定机位）：
  - 基准帧稳定，差分信号强（不依赖微小晃动）
  - 检测运动物体本身，不依赖 YOLO（不怕漏检）
  - 规则明确：上沿→下沿 = 进球，不靠猜
  - 多信号融合：视觉+音频互证，减少误报漏报
"""

import cv2
import numpy as np


class GoalDetector:
    """进球检测器：基准帧差法 + 连通域 + 篮筐穿越检测 + 音频融合。"""

    def __init__(self, hoop_box, baseline_frame=None,
                 hoop_above_margin=30, hoop_x_margin=25,
                 approach_dist=100, disappear_frames=3, confirm_below=True,
                 min_gap_sec=3.0,
                 diff_threshold=25, min_blob_area=50, max_blob_area=5000,
                 search_margin=60,
                 audio_peaks=None, fusion_window=2.0,
                 fusion_mode="or",
                 loose_mode=False,
                 yolo_confirm=False):
        """
        hoop_box: (x1, y1, x2, y2) 篮筐框
        baseline_frame: 基准帧（无球的篮筐画面）BGR，None 则用第一帧
        diff_threshold: 帧差二值化阈值（25 较保守，可降到 15 提灵敏度）
        min_blob_area: 最小连通域面积（过滤噪声）
        max_blob_area: 最大连通域面积（过滤大物体如人）
        search_margin: 篮筐周边搜索范围（像素）
        audio_peaks: 音频峰值时间戳列表（秒），None 表示不用音频
        fusion_window: 音频视觉融合时间窗口（秒，±窗口）
        fusion_mode: 融合模式
            "or"  - 视觉或音频触发都算进球（默认，高召回）
            "and" - 视觉和音频都必须触发（高精度）
            "fused_only" - 仅输出双信号确认的进球
            "visual_only" - 仅用视觉信号（兼容旧行为）
        loose_mode: 宽松模式（高召回，配合 VLM 二次验证使用）
            True  - 运动斑块进入篮筐框内就标记为候选进球（不需穿越上下沿）
                    用于 CV 扫候选 + VLM 验证精度 的工作流
            False - 默认严格模式（斑块需从上沿穿越到下沿）
        yolo_confirm: YOLO 双确认（需配合 loose_mode=True 使用）
            True  - loose 触发时，检查同一帧 YOLO 球位置是否在篮筐框内
                    两者都满足才算进球（高精度，减少球员手/头误报）
            False - 不做 YOLO 确认（纯 diff loose）
        """
        self.hoop_x1, self.hoop_y1, self.hoop_x2, self.hoop_y2 = [int(v) for v in hoop_box]
        self.hoop_cx = (self.hoop_x1 + self.hoop_x2) / 2
        self.hoop_cy = (self.hoop_y1 + self.hoop_y2) / 2
        self.hoop_w = self.hoop_x2 - self.hoop_x1
        self.hoop_h = self.hoop_y2 - self.hoop_y1
        self.hoop_top = self.hoop_y1
        self.hoop_bot = self.hoop_y2

        self.min_gap_sec = min_gap_sec
        self.diff_threshold = diff_threshold
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.search_margin = search_margin

        # 音频融合参数
        self.audio_peaks = sorted([float(p) for p in audio_peaks]) if audio_peaks else []
        self.fusion_window = float(fusion_window)
        self.fusion_mode = fusion_mode
        self.loose_mode = loose_mode  # 宽松模式：斑块进框即候选，配合 VLM 验证
        self.yolo_confirm = yolo_confirm  # YOLO 双确认：loose 触发时检查 YOLO 球位置

        # 基准帧（灰度）
        self.baseline_gray = None
        if baseline_frame is not None:
            self.set_baseline(baseline_frame)

        # 搜索区域（篮筐周边扩展 margin）
        self.search_x1 = max(0, self.hoop_x1 - search_margin)
        self.search_y1 = max(0, self.hoop_y1 - search_margin)
        self.search_x2 = self.hoop_x2 + search_margin
        self.search_y2 = self.hoop_y2 + search_margin

        # 进球状态机：跟踪斑块穿越篮筐
        self.blob_above_hoop = False   # 斑块是否在篮筐上方
        self.blob_in_hoop = False      # 斑块是否在篮筐框内
        self.blob_in_hoop_frames = 0   # 斑块在框内的连续帧数
        self.blob_history = []         # 斑块 y 坐标历史
        self.last_blob_box = None      # 上一帧斑块位置
        self.last_above_frame = -999   # 上次斑块在篮筐上方的帧号
        self.last_in_hoop_frame = -999 # 上次斑块在篮筐框内的帧号
        self.above_timeout_frames = 45 # 上方状态保持时长（约 1.5 秒 @30fps）

        self.goals = []                # 视觉进球时间戳（兼容旧接口）
        self.visual_goals = []         # 视觉进球时间戳
        self.fused_goals = []          # 融合后最终进球时间戳
        self.last_goal_frame = -1
        self.fps = 30.0
        self.last_diff_ratio = 0.0     # 调试用：最近一次差分比例
        self._audio_used = set()       # 已被匹配的音频峰值索引

        # YOLO 球位置历史缓存（用于双确认的时间窗口检查）
        # 格式: [(frame_idx, cx, cy), ...]，保留最近 yolo_window_frames 帧
        self.ball_pos_history = []
        self.yolo_window_frames = 10   # 时间窗口大小（±10帧 ≈ 0.33秒 @30fps）

        # 诊断计数器
        self.diag = {
            "cross_above": 0,      # 斑块到达篮筐上方次数
            "cross_below": 0,      # 斑块到达篮筐下方次数
            "in_hoop": 0,          # 斑块在篮筐框内次数
            "reject_cooldown": 0,  # 冷却期跳过
            "reject_no_above": 0,  # 没到上方就到下方（侧向进筐被拒）
            "reject_in_x": 0,      # x 不在篮筐范围
            "reject_size": 0,      # 斑块太宽
            "reject_trend": 0,     # 趋势检查失败
            "reject_no_blob": 0,   # 没检测到运动斑块
            "side_goal": 0,        # 侧向进筐判定成功
            "timeout_goal": 0,     # 超时匹配成功
            "yolo_confirmed": 0,   # YOLO 双确认成功
            "yolo_rejected": 0,    # YOLO 双确认失败（diff 触发但 YOLO 没球）
        }

    def set_baseline(self, frame):
        """设置基准帧。"""
        if frame is None:
            return
        if len(frame.shape) == 3:
            self.baseline_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self.baseline_gray = frame.copy()
        # 高斯模糊降噪
        self.baseline_gray = cv2.GaussianBlur(self.baseline_gray, (5, 5), 0)

    def _find_moving_blob(self, frame_gray):
        """在篮筐周边搜索区域找最大运动连通域。

        返回: (cx, cy, x1, y1, x2, y2, area) 或 None
        """
        if self.baseline_gray is None:
            return None

        h, w = frame_gray.shape[:2]
        # 限制搜索区域不超过画面
        sx2 = min(self.search_x2, w)
        sy2 = min(self.search_y2, h)

        # 提取搜索区域 ROI
        curr_roi = frame_gray[self.search_y1:sy2, self.search_x1:sx2]
        base_roi = self.baseline_gray[self.search_y1:sy2, self.search_x1:sx2]

        if curr_roi.size == 0 or base_roi.size == 0:
            return None
        if curr_roi.shape != base_roi.shape:
            return None

        # 帧差
        diff = cv2.absdiff(curr_roi, base_roi)
        _, diff_bin = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        # 形态学操作去噪 + 连接相邻区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, kernel)
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, kernel)

        # 连通域分析
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(diff_bin, connectivity=8)

        # 调试用：差分比例
        self.last_diff_ratio = float(np.sum(diff_bin > 0)) / diff_bin.size * 255

        # 找最大的有效连通域（面积在范围内）
        best_blob = None
        best_area = 0
        for k in range(1, num):
            area = stats[k, cv2.CC_STAT_AREA]
            if not (self.min_blob_area <= area <= self.max_blob_area):
                continue
            bx = stats[k, cv2.CC_STAT_LEFT] + self.search_x1
            by = stats[k, cv2.CC_STAT_TOP] + self.search_y1
            bw = stats[k, cv2.CC_STAT_WIDTH]
            bh = stats[k, cv2.CC_STAT_HEIGHT]
            bcx = centroids[k][0] + self.search_x1
            bcy = centroids[k][1] + self.search_y1

            # 斑块宽度不能超过篮筐宽度太多（是球不是大物体）
            if bw > self.hoop_w * 1.5:
                continue

            if area > best_area:
                best_area = area
                best_blob = (float(bcx), float(bcy),
                             int(bx), int(by), int(bx + bw), int(by + bh),
                             int(area))

        return best_blob

    def feed(self, ball_pos, frame_idx, fps, frame=None):
        """喂入一帧数据。

        ball_pos: YOLO 球位置 (cx, cy, x1, y1, x2, y2, conf) 或 None
                  - diff/diff_loose 模式忽略此参数
                  - diff_yolo 模式需要传入，用于双确认
        frame_idx: 当前帧号
        fps: 帧率
        frame: 当前帧 BGR 图像（必须提供）
        返回: 进球时间戳（秒）或 None
        """
        self.fps = fps

        if frame is None:
            return None

        # 首帧自动设为基准（如果未设置）
        if self.baseline_gray is None:
            self.set_baseline(frame)
            return None

        # 转灰度 + 模糊
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        # 冷却期检查
        if self.last_goal_frame >= 0:
            gap_sec = (frame_idx - self.last_goal_frame) / fps
            if gap_sec < self.min_gap_sec:
                self.diag["reject_cooldown"] += 1
                self.last_blob_box = None
                return None

        # 缓存 YOLO 球位置到历史（用于双确认的时间窗口检查）
        # 进球是一个过程（~0.3秒），即使触发瞬间 YOLO 漏检，前后帧检测到也能确认
        if ball_pos is not None:
            self.ball_pos_history.append((frame_idx, ball_pos[0], ball_pos[1]))
        # 保留最近 yolo_window_frames 帧
        cutoff = frame_idx - self.yolo_window_frames
        while self.ball_pos_history and self.ball_pos_history[0][0] < cutoff:
            self.ball_pos_history.pop(0)

        # 在篮筐周边找运动斑块
        blob = self._find_moving_blob(frame_gray)

        if blob is None:
            # 没检测到运动物体，保持状态但衰减历史
            self.diag["reject_no_blob"] += 1
            if len(self.blob_history) > 5:
                self.blob_history.pop(0)
            self.blob_in_hoop_frames = 0
            self.last_blob_box = None
            return None

        cx, cy, bx1, by1, bx2, by2, area = blob
        self.last_blob_box = (bx1, by1, bx2, by2)

        # 记录斑块 y 历史
        self.blob_history.append((frame_idx, cy))
        if len(self.blob_history) > 30:
            self.blob_history.pop(0)

        # 判断斑块相对篮筐的位置
        in_x = (self.hoop_x1 - int(self.hoop_w * 0.3) <= cx
                <= self.hoop_x2 + int(self.hoop_w * 0.3))
        blob_w = bx2 - bx1
        size_ok = blob_w <= self.hoop_w * 1.2

        # ====== 进球判断：斑块从篮筐上沿穿越到下沿 ======
        # 斑块中心在篮筐上方
        if cy < self.hoop_top:
            self.blob_above_hoop = True
            self.last_above_frame = frame_idx
            self.diag["cross_above"] += 1
            self.blob_in_hoop_frames = 0

        # 斑块在篮筐框内
        elif self.hoop_top <= cy <= self.hoop_bot:
            self.diag["in_hoop"] += 1
            if in_x:
                self.blob_in_hoop_frames += 1
                self.last_in_hoop_frame = frame_idx

                # 宽松模式：斑块进框即标记候选（配合 VLM 二次验证）
                # 不要求穿越上下沿、不检查趋势，最大化召回率
                if self.loose_mode and self.blob_in_hoop_frames >= 1:
                    # YOLO 软确认：检查时间窗口内 YOLO 是否检测到球在篮筐附近
                    # YOLO 有球在篮筐附近 → 确认进球（yolo_confirmed）
                    # YOLO 没球或球不在附近 → 不否决，仍注册候选（yolo_no_evidence）
                    #   理由：底角视角下 YOLO 常跟错物体或漏检，
                    #         硬否决会导致真实进球被漏掉，交给 VLM 二次验证更可靠
                    if self.yolo_confirm:
                        margin_x = self.hoop_w * 1.0
                        margin_y = self.hoop_h * 1.0
                        x_lo = self.hoop_x1 - margin_x
                        x_hi = self.hoop_x2 + margin_x
                        y_lo = self.hoop_y1 - margin_y
                        y_hi = self.hoop_y2 + margin_y
                        yolo_near_hoop = any(
                            x_lo <= bx <= x_hi and y_lo <= by <= y_hi
                            for (_, bx, by) in self.ball_pos_history
                        )
                        if yolo_near_hoop:
                            self.diag["yolo_confirmed"] += 1
                        else:
                            self.diag["yolo_rejected"] += 1
                            # 软确认：不 return，继续注册候选
                    ts = frame_idx / fps
                    if self._register_goal(ts, frame_idx, fps, "loose"):
                        self.blob_in_hoop_frames = 0
                        return ts

        # 斑块在篮筐下方
        elif cy >= self.hoop_bot:
            self.diag["cross_below"] += 1

            # 检查上方状态是否在超时窗口内（允许中间几帧漏检）
            above_in_time = (frame_idx - self.last_above_frame) <= self.above_timeout_frames
            hoop_in_time = (frame_idx - self.last_in_hoop_frame) <= self.above_timeout_frames

            # 路径1：从上方穿越到下方（标准进球路径，含超时匹配）
            if self.blob_above_hoop or above_in_time:
                if not in_x:
                    self.diag["reject_in_x"] += 1
                elif not size_ok:
                    self.diag["reject_size"] += 1
                else:
                    trend_ok = self._check_downward_trend()
                    if not trend_ok:
                        self.diag["reject_trend"] += 1
                    else:
                        if above_in_time and not self.blob_above_hoop:
                            self.diag["timeout_goal"] += 1
                        ts = frame_idx / fps
                        self.visual_goals.append(ts)
                        self.blob_above_hoop = False
                        self.last_above_frame = -999
                        self.blob_in_hoop_frames = 0
                        self.blob_history = []
                        has_audio_match = self._match_audio_peak(ts)
                        if self.fusion_mode in ("or", "visual_only"):
                            self.goals.append(ts)
                            self.last_goal_frame = frame_idx
                            if has_audio_match:
                                self.fused_goals.append(ts)
                            return ts
                        elif self.fusion_mode in ("and", "fused_only"):
                            if has_audio_match:
                                self.goals.append(ts)
                                self.fused_goals.append(ts)
                                self.last_goal_frame = frame_idx
                                return ts
                            else:
                                self.last_goal_frame = frame_idx
                                return None
                        else:
                            self.goals.append(ts)
                            self.last_goal_frame = frame_idx
                            return ts
                self.blob_above_hoop = False

            # 路径2：侧向进筐（斑块在框内出现过 >=1 帧后到下方，不要求先到上方）
            elif self.blob_in_hoop_frames >= 1 or hoop_in_time:
                if not in_x:
                    self.diag["reject_in_x"] += 1
                elif not size_ok:
                    self.diag["reject_size"] += 1
                else:
                    if hoop_in_time and self.blob_in_hoop_frames < 1:
                        self.diag["timeout_goal"] += 1
                    else:
                        self.diag["side_goal"] += 1
                    ts = frame_idx / fps
                    self.visual_goals.append(ts)
                    self.blob_in_hoop_frames = 0
                    self.last_in_hoop_frame = -999
                    self.blob_history = []
                    has_audio_match = self._match_audio_peak(ts)
                    if self.fusion_mode in ("or", "visual_only"):
                        self.goals.append(ts)
                        self.last_goal_frame = frame_idx
                        if has_audio_match:
                            self.fused_goals.append(ts)
                        return ts
                    elif self.fusion_mode in ("and", "fused_only"):
                        if has_audio_match:
                            self.goals.append(ts)
                            self.fused_goals.append(ts)
                            self.last_goal_frame = frame_idx
                            return ts
                        else:
                            self.last_goal_frame = frame_idx
                            return None
                    else:
                        self.goals.append(ts)
                        self.last_goal_frame = frame_idx
                        return ts
            else:
                self.diag["reject_no_above"] += 1

            self.blob_in_hoop_frames = 0

        return None

    def _register_goal(self, ts, frame_idx, fps, source=""):
        """注册一个进球时间戳（供宽松模式复用，避免重复代码）。

        返回: True 表示已注册（受冷却期控制），False 表示被冷却期拒绝
        """
        # 冷却期检查
        if self.last_goal_frame >= 0:
            gap_sec = (frame_idx - self.last_goal_frame) / fps
            if gap_sec < self.min_gap_sec:
                self.diag["reject_cooldown"] += 1
                return False

        self.visual_goals.append(ts)
        self.goals.append(ts)
        self.last_goal_frame = frame_idx
        self.blob_history = []
        if source == "loose":
            self.diag["side_goal"] += 1  # 复用 side_goal 计数器
        return True

    def _match_audio_peak(self, ts):
        """检查时间戳 ts 附近 ±fusion_window 内是否有未使用的音频峰值。

        匹配后标记该音频峰值为已使用（避免重复匹配）。
        返回: True/False
        """
        if not self.audio_peaks:
            return False
        best_j = -1
        best_dist = self.fusion_window
        for j, a in enumerate(self.audio_peaks):
            if j in self._audio_used:
                continue
            d = abs(a - ts)
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j >= 0:
            self._audio_used.add(best_j)
            return True
        return False

    def finalize(self):
        """视频处理完毕后调用，处理仅音频触发的候选（fusion_mode="or" 时）。

        返回: 仅音频触发的进球时间戳列表
        """
        audio_only = []
        if not self.audio_peaks or self.fusion_mode != "or":
            return audio_only
        for j, a in enumerate(self.audio_peaks):
            if j in self._audio_used:
                continue
            # 检查是否与已有进球冲突（避免重复）
            conflict = False
            for g in self.goals:
                if abs(g - a) < self.min_gap_sec:
                    conflict = True
                    break
            if not conflict:
                self.goals.append(float(a))
                self.goals.sort()
                audio_only.append(float(a))
        return audio_only

    def _check_downward_trend(self, n=4):
        """检查最近 n 帧斑块是否整体向下运动。"""
        if len(self.blob_history) < n:
            n = len(self.blob_history)
            if n < 2:
                return True
        recent = self.blob_history[-n:]
        ys = [r[1] for r in recent]
        # y 整体增加（向下）且至少下降 5 像素（放宽，适配斜向运动）
        return (ys[-1] - ys[0]) > 5

    def get_debug_info(self):
        """获取调试信息。"""
        return {
            "diff_ratio": self.last_diff_ratio,
            "blob_above": self.blob_above_hoop,
            "blob_box": self.last_blob_box,
            "search_area": (self.search_x1, self.search_y1, self.search_x2, self.search_y2),
            "visual_goals": len(self.visual_goals),
            "fused_goals": len(self.fused_goals),
            "audio_peaks": len(self.audio_peaks),
            "audio_used": len(self._audio_used),
            "fusion_mode": self.fusion_mode,
        }
