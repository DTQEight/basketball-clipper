"""进球检测器：基准帧差法 + 连通域分析 + 篮筐穿越检测（固定机位专用）。

算法参考论文：Camera-based Basketball Scoring Detection Using CNN
实践方案参考：CSDN 神投手（lyandgh）

工作原理：
  1. 标定时取一帧无球的画面作为基准帧
  2. 每帧与基准帧做灰度差分 → 二值化 → 连通域分析
  3. 在篮筐周边区域找最大连通域 = 运动的篮球
  4. 跟踪斑块：斑块先经过篮筐上沿 → 后经过篮筐下沿 = 视觉进球

优势（针对固定机位）：
  - 基准帧稳定，差分信号强（不依赖微小晃动）
  - 检测运动物体本身，不依赖 YOLO（不怕漏检）
  - 规则明确：上沿→下沿 = 进球，不靠猜
"""

import cv2
import numpy as np


class GoalDetector:
    """进球检测器：基准帧差法 + 连通域 + 篮筐穿越检测。"""

    def __init__(self, hoop_box, baseline_frame=None,
                 hoop_above_margin=30, hoop_x_margin=25,
                 approach_dist=100, disappear_frames=3, confirm_below=True,
                 min_gap_sec=3.0,
                 diff_threshold=15, min_blob_area=30, max_blob_area=5000,
                 search_margin=80,
                 loose_mode=False,
                 yolo_confirm=False,
                 rolling_baseline_sec=60.0,
                 min_circularity=0.35,
                 min_in_hoop_frames=2,
                 auto_threshold=True):
        """
        hoop_box: (x1, y1, x2, y2) 篮筐框
        baseline_frame: 基准帧（无球的篮筐画面）BGR，None 则用第一帧
        diff_threshold: 帧差二值化阈值（15 高灵敏度，可提到 25 降误报）
        min_blob_area: 最小连通域面积（过滤噪声）
        max_blob_area: 最大连通域面积（过滤大物体如人）
        search_margin: 篮筐周边搜索范围（像素）
        loose_mode: 宽松模式（高召回，配合 VLM 二次验证使用）
            True  - 运动斑块进入篮筐框内就标记为候选进球（不需穿越上下沿）
                    用于 CV 扫候选 + VLM 验证精度 的工作流
            False - 默认严格模式（斑块需从上沿穿越到下沿）
        yolo_confirm: YOLO 双确认（需配合 loose_mode=True 使用）
            True  - loose 触发时，检查同一帧 YOLO 球位置是否在篮筐框内
                    两者都满足才算进球（高精度，减少球员手/头误报）
            False - 不做 YOLO 确认（纯 diff loose）
        rolling_baseline_sec: 滚动基准帧间隔（秒）
            >0 - 每隔该秒数自动更新基准帧（解决长视频光线/背景变化导致基准失效）
            0  - 禁用滚动基准帧（固定用初始基准帧）
        min_circularity: 最小圆形度（C = 4πA/P²）
            篮球 ≈ 0.7-1.0（近圆），人体 ≈ 0.2-0.5（长条形）
            0.35 阈值过滤球员躯干/手脚经过篮下导致的误报
            设为 0 可禁用形状过滤
        min_in_hoop_frames: 宽松模式下斑块进框触发所需的最小连续帧数
            默认 2 帧（≈0.07秒@30fps），过滤单帧噪声
            球在框内通常 2-4 帧，不宜设太高否则漏检
        auto_threshold: 自适应阈值（默认开启）
            True  - 前 30 秒预热期收集 diff 的 P95 统计量，结束时自动计算阈值
                    阈值 = median(P95s) + 8，clamp 到 [8, 50]
                    预热期内只采样不判定进球
            False - 使用 diff_threshold 固定值
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

        self.loose_mode = loose_mode  # 宽松模式：斑块进框即候选，配合 VLM 验证
        self.yolo_confirm = yolo_confirm  # YOLO 双确认：loose 触发时检查 YOLO 球位置
        self.rolling_baseline_sec = float(rolling_baseline_sec)  # 滚动基准帧间隔
        self.min_circularity = float(min_circularity)  # 最小圆形度（形状过滤）
        self.min_in_hoop_frames = int(min_in_hoop_frames)  # 宽松模式最小进框帧数

        # 自适应阈值：预热期收集 P95，结束时自动计算 diff_threshold
        self.auto_threshold = bool(auto_threshold)
        self._warmup_p95s = []                # 预热期每帧 P95（0-255）
        self._warmup_target_sec = 30.0        # 预热时长（秒）
        self._warmup_done = not self.auto_threshold  # 关闭自适应时视为已完成
        self._warmup_start_frame = -1         # 预热起始帧（首次 feed 时设置）
        self._user_diff_threshold = int(diff_threshold)  # 保存用户设定值
        self._auto_threshold_value = None     # 自适应计算出的阈值（诊断用）
        self._warmup_p95_median = None        # 预热结束时保存的 P95 中位数（诊断用，避免列表被清空）
        self._warmup_sample_count = 0         # 预热期实际采样的帧数（诊断用）

        # 基准帧（灰度）
        self.baseline_gray = None
        self.last_baseline_frame_idx = -1  # 上次更新基准帧的帧号
        self.baseline_update_count = 0     # 基准帧更新次数（诊断用）
        # 滚动基准帧候选：在更新间隔内持续寻找运动量最小的帧作为下一个基准帧
        # 避免把球/球员拍进基准帧导致 diff 失效
        self._baseline_candidate_frame = None
        self._baseline_candidate_diff = float("inf")
        self._baseline_candidate_idx = -1
        if baseline_frame is not None:
            self.set_baseline(baseline_frame)

        # 搜索区域（篮筐周边扩展 margin）
        self.search_x1 = max(0, self.hoop_x1 - search_margin)
        self.search_y1 = max(0, self.hoop_y1 - search_margin)
        self.search_x2 = self.hoop_x2 + search_margin
        self.search_y2 = self.hoop_y2 + search_margin

        # 形态学 kernel 缓存：_find_moving_blob 每帧都用，__init__ 创建一次复用
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 进球状态机：跟踪斑块穿越篮筐
        self.blob_above_hoop = False   # 斑块是否在篮筐上方
        self.blob_in_hoop = False      # 斑块是否在篮筐框内
        self.blob_in_hoop_frames = 0   # 斑块在框内的连续帧数
        self.blob_persistent_frames = 0  # 斑块持续帧数（有运动物体的连续帧）
        self.blob_history = []         # 斑块 y 坐标历史
        self.last_blob_box = None      # 上一帧斑块位置
        self.last_above_frame = -999   # 上次斑块在篮筐上方的帧号
        self.last_in_hoop_frame = -999 # 上次斑块在篮筐框内的帧号
        self.above_timeout_frames = 45 # 上方状态保持时长（约 1.5 秒 @30fps）

        self.goals = []                # 进球时间戳（兼容旧接口）
        self.visual_goals = []         # 视觉进球时间戳
        self.last_goal_frame = -1
        self.fps = 30.0
        self.last_diff_ratio = 0.0     # 调试用：最近一次差分比例

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
            "reject_shape": 0,     # 形状过滤失败（圆形度过低，疑似人）
            "reject_trend": 0,     # 趋势检查失败
            "reject_no_blob": 0,   # 没检测到运动斑块
            "side_goal": 0,        # 侧向进筐判定成功
            "timeout_goal": 0,     # 超时匹配成功
            "yolo_confirmed": 0,   # YOLO 双确认成功
            "yolo_rejected": 0,    # YOLO 双确认失败（diff 触发但 YOLO 没球）
            "baseline_updates": 0, # 滚动基准帧更新次数
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

    def has_motion_near_hoop(self, frame, threshold=None):
        """快速检查篮筐搜索区域是否有运动像素（用于条件跳过 YOLO）。

        只做裁剪 + absdiff + countNonZero，不做形态学/连通域，~1ms。
        返回: True=有运动（需要 YOLO），False=无运动（可跳过 YOLO）
        """
        if self.baseline_gray is None:
            return True  # 基准帧还没设，不跳过
        # 裁剪 ROI
        sy2 = min(self.search_y2, frame.shape[0])
        sx2 = min(self.search_x2, frame.shape[1])
        frame_roi = frame[self.search_y1:sy2, self.search_x1:sx2]
        if frame_roi.size == 0:
            return True
        if len(frame_roi.shape) == 3:
            frame_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        frame_roi = cv2.GaussianBlur(frame_roi, (5, 5), 0)
        base_roi = self.baseline_gray[self.search_y1:sy2, self.search_x1:sx2]
        diff = cv2.absdiff(frame_roi, base_roi)
        thr = threshold if threshold is not None else (self._auto_threshold_value or self.diff_threshold)
        _, diff_bin = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        # 运动像素超过搜索区域总像素的 1% 才算有运动
        # 球经过时通常 2000+ 像素，噪声/光线微变通常 <200 像素
        _motion_ratio = cv2.countNonZero(diff_bin) / max(diff_bin.size, 1)
        return _motion_ratio > 0.01

    def _find_moving_blob(self, frame_gray):
        """在篮筐周边搜索区域找最大运动连通域。

        过滤条件：
          1. 面积在 [min_blob_area, max_blob_area] 范围内
          2. 宽度 ≤ 篮筐宽度 × 1.5
          3. 圆形度 C = 4πA/P² ≥ min_circularity（过滤长条形人体）
             篮球 ≈ 0.7-1.0，人体 ≈ 0.2-0.5，默认阈值 0.35

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

        # 自适应阈值：预热期收集 P95 统计量（用于自动计算 diff_threshold）
        # 降采样计算 P95：每 4 个像素取 1 个，计算量减少 16 倍，精度足够（统计量）
        if self.auto_threshold and not self._warmup_done:
            sample = diff[::4, ::4]
            self._warmup_p95s.append(float(np.percentile(sample, 95)))

        _, diff_bin = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        # 形态学操作去噪 + 连接相邻区域（kernel 在 __init__ 缓存，避免每帧重建）
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, self._morph_kernel)
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, self._morph_kernel)

        # 用 findContours 替代 connectedComponents，以获取周长计算圆形度
        contours, _ = cv2.findContours(diff_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 调试用：差分比例（0-1，画面变化区域占比）
        # cv2.countNonZero 直接计数，省去 np.sum(bin>0) 的 bool 数组创建开销
        self.last_diff_ratio = cv2.countNonZero(diff_bin) / max(diff_bin.size, 1)

        # 找最大的有效连通域（面积在范围内 + 形状过滤）
        best_blob = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_blob_area <= area <= self.max_blob_area):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            bx = x + self.search_x1
            by = y + self.search_y1

            # 斑块宽度不能超过篮筐宽度太多（是球不是大物体）
            if bw > self.hoop_w * 1.5:
                continue

            # 圆形度过滤：C = 4πA/P²，过滤长条形人体（球员经过篮下）
            if self.min_circularity > 0:
                peri = cv2.arcLength(cnt, True)
                if peri <= 0:
                    continue
                circularity = 4 * np.pi * area / (peri * peri)
                if circularity < self.min_circularity:
                    self.diag["reject_shape"] += 1
                    continue

            # 质心
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                bcx = M["m10"] / M["m00"] + self.search_x1
                bcy = M["m01"] / M["m00"] + self.search_y1
            else:
                bcx = bx + bw / 2.0
                bcy = by + bh / 2.0

            if area > best_area:
                best_area = area
                best_blob = (float(bcx), float(bcy),
                             int(bx), int(by), int(bx + bw), int(by + bh),
                             float(area))

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
            self.last_baseline_frame_idx = frame_idx
            self._warmup_start_frame = frame_idx
            return None

        # 修复：若基准帧是外部传入的（__init__ 中设置），last_baseline_frame_idx 仍为 -1
        # 此时把它对齐到当前 frame_idx，否则滚动基准帧的触发条件 (>=0) 永远不满足
        if self.last_baseline_frame_idx < 0:
            self.last_baseline_frame_idx = frame_idx
        if self._warmup_start_frame < 0:
            self._warmup_start_frame = frame_idx

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

        # ====== 自适应阈值预热：只采样不判定 ======
        if self.auto_threshold and not self._warmup_done:
            warmup_sec = (frame_idx - self._warmup_start_frame) / fps
            if warmup_sec >= self._warmup_target_sec and len(self._warmup_p95s) > 0:
                # 预热结束：阈值 = median(P95) + 8，clamp 到 [8, 50]
                median_p95 = float(np.median(self._warmup_p95s))
                auto_thr = int(median_p95 + 8)
                auto_thr = max(8, min(50, auto_thr))
                self.diff_threshold = auto_thr
                self._auto_threshold_value = auto_thr
                self._warmup_p95_median = median_p95
                self._warmup_sample_count = len(self._warmup_p95s)
                self._warmup_done = True
                self._warmup_p95s = []  # 清空释放内存
            else:
                # 预热期内不判定进球
                return None

        # ====== 滚动基准帧更新（双触发：时间间隔 + 斑块持续） ======
        # 旧逻辑只在 blob is None 时更新，球员常驻篮筐附近时永不更新导致 diff 失效。
        # 新逻辑：① 不管有没有 blob 都检查触发条件；
        #         ② 在更新间隔内持续寻找运动量最小的帧作为候选基准，避免把球拍进基准帧。
        if self.rolling_baseline_sec > 0 and self.last_baseline_frame_idx >= 0:
            gap_sec = (frame_idx - self.last_baseline_frame_idx) / fps
            # 更新候选基准帧（选 diff 最小的帧 = 运动量最小的帧）
            if self.last_diff_ratio < self._baseline_candidate_diff:
                self._baseline_candidate_frame = frame.copy()
                self._baseline_candidate_diff = self.last_diff_ratio
                self._baseline_candidate_idx = frame_idx
            # 触发条件1：时间间隔（默认 60 秒）
            time_trigger = gap_sec >= self.rolling_baseline_sec
            # 触发条件2：斑块持续超过 5 秒（球员常驻篮筐附近，基准帧需更新）
            persistent_trigger = (blob is not None and
                                  self.blob_persistent_frames > self.fps * 5)
            if time_trigger or persistent_trigger:
                cand = self._baseline_candidate_frame
                if cand is not None:
                    self.set_baseline(cand)
                    self.last_baseline_frame_idx = self._baseline_candidate_idx
                else:
                    self.set_baseline(frame)
                    self.last_baseline_frame_idx = frame_idx
                self.baseline_update_count += 1
                self.diag["baseline_updates"] += 1
                # 重置候选
                self._baseline_candidate_frame = None
                self._baseline_candidate_diff = float("inf")
                self._baseline_candidate_idx = -1
                # 重置持续计数，避免"球员常驻"触发后每帧都重复更新基准帧
                self.blob_persistent_frames = 0

        if blob is None:
            # 没检测到运动物体，保持状态但衰减历史
            self.diag["reject_no_blob"] += 1
            if len(self.blob_history) > 5:
                self.blob_history.pop(0)
            self.blob_in_hoop_frames = 0
            self.blob_persistent_frames = 0  # 无斑块，重置持续计数
            self.last_blob_box = None
            return None

        # 有运动斑块，增加持续计数（用于"球员常驻"触发基准帧更新）
        self.blob_persistent_frames += 1
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

                # 宽松模式：斑块进框连续 min_in_hoop_frames 帧即标记候选
                # 默认 2 帧（≈0.07秒@30fps），过滤单帧噪声，不影响真实进球（球在框内 2-4 帧）
                if self.loose_mode and self.blob_in_hoop_frames >= self.min_in_hoop_frames:
                    # YOLO 硬否决：篮筐附近±10帧内没检测到球 → 直接排除（预览片段没球的误检靠这里过滤）
                    yolo_ok, yolo_status = self._check_yolo_near_hoop()
                    if yolo_ok:
                        if yolo_status == "confirmed":
                            self.diag["yolo_confirmed"] += 1
                        ts = frame_idx / fps
                        if self._register_goal(ts, frame_idx, fps, "loose"):
                            self.blob_in_hoop_frames = 0
                            return ts
                    else:
                        self.diag["yolo_rejected"] += 1
                        self.blob_in_hoop_frames = 0
                        return None

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
                        # YOLO 硬否决：篮筐附近没球就不算进球
                        yolo_ok, yolo_status = self._check_yolo_near_hoop()
                        if not yolo_ok:
                            self.diag["yolo_rejected"] += 1
                        else:
                            if yolo_status == "confirmed":
                                self.diag["yolo_confirmed"] += 1
                            if above_in_time and not self.blob_above_hoop:
                                self.diag["timeout_goal"] += 1
                            ts = frame_idx / fps
                            self.blob_above_hoop = False
                            self.last_above_frame = -999
                            self.blob_in_hoop_frames = 0
                            if self._register_goal(ts, frame_idx, fps, "visual"):
                                return ts
                self.blob_above_hoop = False

            # 路径2：侧向进筐（斑块在框内出现过 >=1 帧后到下方，不要求先到上方）
            elif self.blob_in_hoop_frames >= 1 or hoop_in_time:
                if not in_x:
                    self.diag["reject_in_x"] += 1
                elif not size_ok:
                    self.diag["reject_size"] += 1
                else:
                    # YOLO 硬否决：篮筐附近没球就不算进球
                    yolo_ok, yolo_status = self._check_yolo_near_hoop()
                    if not yolo_ok:
                        self.diag["yolo_rejected"] += 1
                    else:
                        if yolo_status == "confirmed":
                            self.diag["yolo_confirmed"] += 1
                        if hoop_in_time and self.blob_in_hoop_frames < 1:
                            self.diag["timeout_goal"] += 1
                        else:
                            self.diag["side_goal"] += 1
                        ts = frame_idx / fps
                        self.blob_in_hoop_frames = 0
                        self.last_in_hoop_frame = -999
                        if self._register_goal(ts, frame_idx, fps, "side"):
                            return ts
            else:
                self.diag["reject_no_above"] += 1

            self.blob_in_hoop_frames = 0

        return None

    def _register_goal(self, ts, frame_idx, fps, source=""):
        """注册一个进球时间戳（受冷却期控制）。

        返回: True 表示已注册，False 表示被冷却期拒绝
        """
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

    def _check_yolo_near_hoop(self):
        """检查 YOLO 历史中，篮筐附近（±1倍筐宽高）是否有球。

        返回: (有球: bool, 用于诊断: 'confirmed'/'rejected'/'skipped')
        """
        if not self.yolo_confirm:
            return True, "skipped"
        margin_x = self.hoop_w * 1.0
        margin_y = self.hoop_h * 1.0
        x_lo = self.hoop_x1 - margin_x
        x_hi = self.hoop_x2 + margin_x
        y_lo = self.hoop_y1 - margin_y
        y_hi = self.hoop_y2 + margin_y
        has_ball = any(
            x_lo <= bx <= x_hi and y_lo <= by <= y_hi
            for (_, bx, by) in self.ball_pos_history
        )
        return has_ball, "confirmed" if has_ball else "rejected"

    def get_debug_info(self):
        """获取调试信息。"""
        if self.auto_threshold and not self._warmup_done and self.fps > 0:
            warmup_elapsed = (self._warmup_p95s and len(self._warmup_p95s) / self.fps) or 0.0
        else:
            warmup_elapsed = self._warmup_target_sec if self._warmup_done else 0.0
        return {
            "diff_ratio": self.last_diff_ratio,
            "blob_above": self.blob_above_hoop,
            "blob_box": self.last_blob_box,
            "search_area": (self.search_x1, self.search_y1, self.search_x2, self.search_y2),
            "visual_goals": len(self.visual_goals),
            "auto_threshold": self.auto_threshold,
            "warmup_done": self._warmup_done,
            "warmup_elapsed_sec": round(float(warmup_elapsed), 1),
            "warmup_target_sec": self._warmup_target_sec,
            "diff_threshold": self.diff_threshold,
            "auto_threshold_value": self._auto_threshold_value,
            "user_diff_threshold": self._user_diff_threshold,
        }
