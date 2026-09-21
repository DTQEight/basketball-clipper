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
from collections import deque
import base64

# 自适应阈值预热时长（秒）：前 N 秒逐帧收集 diff P95 统计量
WARMUP_TARGET_SEC = 30.0


class GoalDetector:
    """进球检测器：基准帧差法 + 连通域 + 篮筐穿越检测。"""

    def __init__(self, hoop_box, baseline_frame=None,
                 min_gap_sec=3.0,
                 diff_threshold=15, min_blob_area=30, max_blob_area=5000,
                 search_margin=80,
                 loose_mode=False,
                 yolo_confirm=False,
                 rolling_baseline_sec=60.0,
                 min_circularity=0.35,
                 min_in_hoop_frames=2,
                 auto_threshold=True,
                 fps=30.0,
                 yolo_probe=None):
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
        fps: 视频帧率（默认 30）。内部所有"按秒定义"的时间窗口
             （上方状态保持 1.5s / YOLO 时间窗 ±0.34s）按 fps 换算成帧数，
             避免硬编码帧数在 60fps 视频上窗口减半、15fps 上翻倍。
        yolo_probe: 两段式兜底的「备用档」补检回调 `fn(frame, frame_idx) -> 球框列表 | None`
            默认档（如裁剪推理）在整窗内没有任何球证据、判定即将否决时，
            用它补检一次；命中则把证据写入 ball_pos_history（本帧按 confirmed 处理）。
            None = 关闭兜底，行为与旧版完全一致。
            设计原因（2026.09.21 实测）：裁剪档在真球帧上会被系统性压低（整窗一帧都不过
            阈值的事件才会漏球，如 08.31-3rd 的 449.16s），而全局换备用档要付
            +14.5% 检测耗时的代价；只对「即将被否决」的轨迹补检，效果同全局换档、
            成本只花在少数边缘轨迹上。
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

        # 两段式兜底（备用档补检）：见 __init__ docstring 的 yolo_probe
        self.yolo_probe = yolo_probe
        self._cur_frame = None        # 当前帧 BGR（供补检使用）
        self._cur_frame_idx = -1
        self._last_probe_frame = -1   # 补检去重：每帧最多一次

        # 自适应阈值：预热期收集 P95，结束时自动计算 diff_threshold
        self.auto_threshold = bool(auto_threshold)
        self._warmup_p95s = []                # 预热期每帧 P95（0-255）
        self._warmup_target_sec = WARMUP_TARGET_SEC  # 预热时长（秒）
        self._warmup_done = not self.auto_threshold  # 关闭自适应时视为已完成
        self._warmup_start_frame = -1         # 预热起始帧（首次 feed 时设置）
        self._auto_threshold_value = None     # 自适应计算出的阈值（诊断用）
        self._warmup_p95_median = None        # 预热结束时保存的 P95 中位数（诊断用，避免列表被清空）
        self._warmup_sample_count = 0         # 预热期实际采样的帧数（诊断用）

        # 搜索区域（篮筐周边扩展 margin）
        # 注意：必须在基准帧初始化之前定义 —— set_baseline/_roi_gray 依赖这些坐标
        self.search_x1 = max(0, self.hoop_x1 - search_margin)
        self.search_y1 = max(0, self.hoop_y1 - search_margin)
        self.search_x2 = self.hoop_x2 + search_margin
        self.search_y2 = self.hoop_y2 + search_margin

        # 基准帧（只存搜索区域 ROI 的灰度模糊图，见 _roi_gray）
        self.baseline_gray = None
        self.last_baseline_frame_idx = -1  # 上次更新基准帧的帧号
        # 滚动基准帧候选：在更新间隔内持续寻找运动量最小的帧作为下一个基准帧
        # 避免把球/球员拍进基准帧导致 diff 失效
        # （只存 ROI 灰度图 ~90KB，不拷整帧 BGR ~6MB）
        self._baseline_candidate_frame = None
        self._baseline_candidate_diff = float("inf")
        self._baseline_candidate_idx = -1
        if baseline_frame is not None:
            self.set_baseline(baseline_frame)

        # 形态学 kernel 缓存：_find_moving_blob 每帧都用，__init__ 创建一次复用
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 进球状态机：跟踪斑块穿越篮筐
        self.blob_above_hoop = False   # 斑块是否在篮筐上方
        self.blob_in_hoop = False      # 斑块是否在篮筐框内
        self.blob_in_hoop_frames = 0   # 斑块在框内的连续帧数
        self.blob_persistent_frames = 0  # 斑块持续帧数（有运动物体的连续帧）
        self.blob_history = deque(maxlen=30)  # 斑块 y 坐标历史（maxlen 自动淘汰旧值）
        self.last_blob_box = None      # 上一帧斑块位置
        self.last_above_frame = -999   # 上次斑块在篮筐上方的帧号
        self.last_in_hoop_frame = -999 # 上次斑块在篮筐框内的帧号
        # 上方状态保持时长按秒定义（1.5 秒），按 fps 换算帧数：
        # 旧实现硬编码 45 帧，60fps 视频窗口缩到 0.75s（漏检）、15fps 拉长到 3s（误配）
        self.fps = float(fps)
        self.above_timeout_frames = max(10, int(1.5 * self.fps))

        self.goals = []                # 进球时间戳
        self.last_goal_frame = -1
        self.last_diff_ratio = 0.0     # 调试用：最近一次差分比例

        # YOLO 球位置历史缓存（用于双确认的时间窗口检查）
        # 格式: [(frame_idx, cx, cy), ...]，保留最近 yolo_window_frames 帧
        # deque：旧实现 list.pop(0) 为 O(n)，popleft 为 O(1)
        # 窗口按秒定义（±0.34 秒 ≈ 10 帧 @30fps），同样按 fps 换算
        self.ball_pos_history = deque()
        self.yolo_window_frames = max(5, int(0.34 * self.fps))

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
            "loose_goal": 0,       # 宽松模式进框判定成功（独立于 side_goal，便于排查漏检来源）
            "timeout_goal": 0,     # 超时匹配成功
            "yolo_confirmed": 0,   # YOLO 双确认成功
            "yolo_rejected": 0,    # YOLO 双确认失败（diff 触发但 YOLO 没球）
            "probe_called": 0,     # 备用档补检次数（两段式兜底）
            "probe_confirmed": 0,  # 备用档补检命中并救回判定的次数
            "baseline_updates": 0, # 滚动基准帧更新次数
        }

    def _roi_gray(self, frame):
        """裁剪篮筐搜索区域 → 灰度 → 高斯模糊。

        差分计算只关心搜索区域（1080p 典型篮筐下约占整帧 5% 面积），
        先裁 ROI 再做 cvtColor/GaussianBlur，避免整帧处理浪费 ~95% 计算量。
        （与基准帧 baseline_gray 的存储格式保持一致：ROI 灰度模糊图）
        """
        sy2 = min(self.search_y2, frame.shape[0])
        sx2 = min(self.search_x2, frame.shape[1])
        roi = frame[self.search_y1:sy2, self.search_x1:sx2]
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(roi, (5, 5), 0)

    def set_baseline(self, frame) -> None:
        """设置基准帧（整帧 BGR 输入，内部只保留搜索区域 ROI 的灰度模糊图）。"""
        if frame is None:
            return
        self.baseline_gray = self._roi_gray(frame)

    def _set_baseline_roi(self, roi):
        """直接用已处理好的 ROI 灰度图设置基准（滚动基准候选复用，避免重复处理）。"""
        if roi is None:
            return
        self.baseline_gray = roi.copy()

    def compute_roi(self, frame):
        """公开包装：预计算当前帧的搜索区 ROI（供检测循环复用，省一次重复处理）。"""
        return self._roi_gray(frame)

    def has_motion_near_hoop(self, frame, threshold: int | None = None,
                             frame_roi=None) -> bool:
        """快速检查筐区有没有「值得 YOLO 看一眼」的运动（用于条件跳过 YOLO）。

        **不变量：本门必须严格宽松于 `_find_moving_blob`** —— 只要斑块检测器
        可能产出候选，本门就必须为真。否则会出现「斑块触发了、YOLO 却从没在
        附近跑过」→ `_check_yolo_near_hoop` 必然判否 → 真进球被硬否决。
        旧实现要求「超阈值像素 > 搜索区总像素的 1%」（1080p 下约 715px），而斑块
        下限只有 30px：落在两者之间的真球全部被静默丢掉（2026.08.15-1st 的
        7:35~10:32 空档即由这一条 + 球框只留 argmax 共同造成）。

        实现：与 `_find_moving_blob` 用**同一套二值化 + 同一套形态学**，只保留
        面积下限，去掉宽度与圆形度过滤——那两道是「选球」用的，不该决定
        「要不要跑 YOLO」。

        frame_roi: 调用方已用 compute_roi 算好的 ROI，传入可省一次重复计算。
        返回: True=有运动（需要 YOLO），False=无运动（可跳过 YOLO）
        """
        if self.baseline_gray is None:
            return True  # 基准帧还没设，不跳过
        frame_roi = frame_roi if frame_roi is not None else self._roi_gray(frame)
        if frame_roi.size == 0 or frame_roi.shape != self.baseline_gray.shape:
            return True
        diff = cv2.absdiff(frame_roi, self.baseline_gray)
        thr = threshold if threshold is not None else (self._auto_threshold_value or self.diff_threshold)
        _, diff_bin = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        # 与 _find_moving_blob 完全相同的形态学（去噪 + 连接）：
        # 只做 countNonZero 会把散点噪声算成"有运动"，做形态学才能对齐"连通域"语义。
        # 代价 ~1ms/帧，换来"斑块能触发 ⇒ YOLO 必已跑过"这个可断言的不变量。
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, self._morph_kernel)
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, self._morph_kernel)
        contours, _ = cv2.findContours(diff_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return any(cv2.contourArea(c) >= self.min_blob_area for c in contours)

    def _find_moving_blob(self, frame_roi_gray):
        """在篮筐周边搜索区域 ROI 内找最大运动连通域。

        frame_roi_gray: 由 _roi_gray 产出的搜索区域灰度模糊图
                        （与 baseline_gray 同格式，直接做差分）

        过滤条件：
          1. 面积在 [min_blob_area, max_blob_area] 范围内
          2. 宽度 ≤ 篮筐宽度 × 1.5
          3. 圆形度 C = 4πA/P² ≥ min_circularity（过滤长条形人体）
             篮球 ≈ 0.7-1.0，人体 ≈ 0.2-0.5，默认阈值 0.35

        返回: (cx, cy, x1, y1, x2, y2, area) 或 None
        """
        if self.baseline_gray is None:
            return None

        curr_roi = frame_roi_gray
        base_roi = self.baseline_gray

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

    def feed(self, ball_pos, frame_idx: int, fps: float, frame=None,
             ball_frame=None, frame_roi=None):
        """喂入一帧数据。

        ball_pos: 该帧 YOLO 检出的**全部**球位置，列表 [(cx, cy, x1, y1, x2, y2, conf), ...]
                  （可为空列表或 None）。按置信度降序排列，调用方通常把第一个当主球。
                  - diff/diff_loose 模式忽略此参数
                  - diff_yolo 模式需要传入，用于双确认
                  必须是列表而不是单个框：只留最高分那一个框时，画面上存在的
                  稳定误报（远端场地上的球/静置物，置信度常达 0.5+）会在真球过筐
                  口时以微弱优势胜出、把筐边的真球挤掉，导致真进球被 YOLO 硬否决
                  （见 2026.08.15-1st 7:35~10:32 空档的逐帧归因）。
        frame_idx: 当前帧号
        fps: 帧率
        frame: 当前帧 BGR 图像（必须提供）
        ball_frame: ball_pos 实际对应的检测帧号（默认与 frame_idx 相同）。
                    跳帧复用上一帧 YOLO 结果时，位置是 1-2 帧前的，
                    传入原始帧号保持时间窗口语义（±N 帧内球在筐边）准确。
        frame_roi: 预计算的搜索区 ROI（compute_roi 产出），传入可省一次重复处理。
        返回: 进球时间戳（秒）或 None
        """
        # fps 归一化防御：0/None 视为无效值，回退实例值（构造时默认 30.0），
        # 防止下游多处 /fps 除零、self.fps*5 TypeError，或把无效值写回实例状态
        if not fps:
            fps = self.fps if self.fps > 0 else 1.0
        # fps 变化时重算按秒定义的时间窗口：
        # 构造时可能用默认 fps=30（调用方不传），实际视频帧率在首次 feed 才知道，
        # 不重算的话 60fps 视频的窗口会按 30fps 算（减半），与 persistent_trigger
        # 等按 self.fps 计算的窗口语义割裂
        if fps != self.fps:
            self.fps = float(fps)
            self.above_timeout_frames = max(10, int(1.5 * self.fps))
            self.yolo_window_frames = max(5, int(0.34 * self.fps))

        if frame is None:
            return None

        # 保存当前帧供「备用档补检」使用（两段式兜底；未装配 probe 时无任何开销）
        self._cur_frame = frame
        self._cur_frame_idx = frame_idx

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

        # 搜索区域 ROI：灰度 + 模糊（只处理篮筐周边 ~5% 画面，不碰整帧，
        # 旧实现对整幅 1080p 做 cvtColor+GaussianBlur 后又丢弃 ~95% 结果）
        # 支持外部传入预计算结果：检测循环里 has_motion_near_hoop 与 feed
        # 各算一次 ROI 属纯浪费（~1-2ms/帧，~2-4% 吞吐）
        frame_roi = frame_roi if frame_roi is not None else self._roi_gray(frame)

        # 缓存 YOLO 球位置到历史（用于双确认的时间窗口检查）
        # 进球是一个过程（~0.3秒），即使触发瞬间 YOLO 漏检，前后帧检测到也能确认。
        # 放在冷却检查之前：冷却 3s 内检出的球位置也是有效样本，
        # 冷却刚结束触发的补篮候选需要窗口内有球才能通过 YOLO 确认
        # 本帧全部球位置都写入：_check_yolo_near_hoop 关心的是「筐邻域有没有球」，
        # 只写最高分那一个会让筐边真球被画面上其它球/误报挤掉（见方法 docstring）
        for _bp in (ball_pos or ()):
            self.ball_pos_history.append(
                (ball_frame if ball_frame is not None else frame_idx,
                 _bp[0], _bp[1]))
        # 保留最近 yolo_window_frames 帧
        cutoff = frame_idx - self.yolo_window_frames
        while self.ball_pos_history and self.ball_pos_history[0][0] < cutoff:
            self.ball_pos_history.popleft()

        # 冷却期检查
        if self.last_goal_frame >= 0:
            gap_sec = (frame_idx - self.last_goal_frame) / fps
            if gap_sec < self.min_gap_sec:
                self.diag["reject_cooldown"] += 1
                self.last_blob_box = None
                # 冷却早退不走下方"无斑块"分支，in_hoop 计数若不重置会残留：
                # 冷却结束后第一帧（如抢篮板球员持球进框）残留计数立即满足
                # "连续 min_in_hoop_frames 帧"，跳过防噪约束直接触发 loose 误报
                self.blob_in_hoop_frames = 0
                return None

        # 在篮筐周边找运动斑块
        blob = self._find_moving_blob(frame_roi)

        # ====== 自适应阈值预热：只采样不判定 ======
        if self.auto_threshold and not self._warmup_done:
            # 按样本计数判定，不按帧号换算时间：
            # iter_frames 的 fidx 由 pts 浮点换算，可能整体偏移（首帧从 1 起而非 0），
            # 用 (frame_idx - start)/fps >= 30s 判定会因偏移差一帧而永远不触发。
            # 样本数 = 真实数据量，N 个样本 @fps 就是 N/fps 秒，与帧号偏移无关。
            _n_needed = int(self._warmup_target_sec * max(fps, 1.0))
            if len(self._warmup_p95s) >= _n_needed:
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
            # 只存 ROI 灰度图（与 baseline_gray 同格式，~90KB），
            # 旧实现 frame.copy() 拷整帧 BGR ~6MB，60s 窗口内可能拷贝上百次
            if self.last_diff_ratio < self._baseline_candidate_diff:
                self._baseline_candidate_frame = frame_roi.copy()
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
                    # 候选已是 ROI 灰度模糊图，直接采用（无需再处理整帧）
                    self._set_baseline_roi(cand)
                    self.last_baseline_frame_idx = self._baseline_candidate_idx
                else:
                    self.set_baseline(frame)
                    self.last_baseline_frame_idx = frame_idx
                self.diag["baseline_updates"] += 1
                # 重置候选
                self._baseline_candidate_frame = None
                self._baseline_candidate_diff = float("inf")
                self._baseline_candidate_idx = -1
                # 重置持续计数，避免"球员常驻"触发后每帧都重复更新基准帧
                self.blob_persistent_frames = 0

        if blob is None:
            # 没检测到运动物体：轨迹断档即清空历史
            # （旧实现每帧只淘汰 1 条，断档后重出现的斑块会把上段轨迹的 y 值
            #   混进 _check_downward_trend 的首尾位移检查，跨段位移可能误通过）
            self.diag["reject_no_blob"] += 1
            self.blob_history.clear()
            self.blob_in_hoop_frames = 0
            self.blob_persistent_frames = 0  # 无斑块，重置持续计数
            self.blob_above_hoop = False  # 上方状态只属于当前连续轨迹段，断档即失效
            self.last_blob_box = None
            return None

        # 有运动斑块，增加持续计数（用于"球员常驻"触发基准帧更新）
        self.blob_persistent_frames += 1
        cx, cy, bx1, by1, bx2, by2, area = blob
        self.last_blob_box = (bx1, by1, bx2, by2)

        # 记录斑块 y 历史（deque(maxlen=30) 自动淘汰最旧记录）
        self.blob_history.append((frame_idx, cy))

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

        # 斑块在篮筐框内（中心 y 位于框带内）
        elif self.hoop_top <= cy <= self.hoop_bot:
            # 仅当水平位置也进框（in_x，含 ±30% 框宽余量）才算"进框"：
            # diag["in_hoop"] 与 blob_in_hoop_frames 同一口径累加（B9）——
            # 旧实现不管 in_x 都 +1，把垂直恰好穿过筐带但水平离框很远的
            # 无关斑块（如球员躯干经过筐下）也计成"筐内"，诊断日志/FAQ 的
            # "筐内次数"相对真实进框信号虚高，无法与 loose/side 判定路径对应
            if in_x:
                self.diag["in_hoop"] += 1
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
                            # loose 进球后复位"曾在上方"标记：否则本次下穿轨迹结束后，
                            # flag 无限期残留，几秒后任意无关"下方下移"斑块会走路径1
                            # 直接按 visual 注册（无 YOLO 部署时无兜底）
                            self.blob_above_hoop = False
                            self.last_above_frame = -999
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

        self.goals.append(ts)
        self.last_goal_frame = frame_idx
        self.blob_history.clear()
        if source == "loose":
            self.diag["loose_goal"] += 1  # 独立计数器（旧实现复用 side_goal，两条路径混在一起无法排查）
        return True

    def _check_downward_trend(self, n=4):
        """检查最近 n 帧斑块是否整体向下运动。"""
        if len(self.blob_history) < n:
            n = len(self.blob_history)
            if n < 2:
                return True
        recent = list(self.blob_history)[-n:]  # deque 不支持切片，转 list 取尾部
        ys = [r[1] for r in recent]
        # y 整体增加（向下）且至少下降 5 像素（放宽，适配斜向运动）
        return (ys[-1] - ys[0]) > 5

    def _probe_missing_yolo(self, frame_idx):
        """两段式兜底：默认档整窗无球证据时，用备用档在当前帧补检一次。

        返回是否写入了新证据。**每帧最多补检一次**（_last_probe_frame 去重，
        避免同一轨迹连续多帧反复推理），且只在 yolo_probe 已装配时生效——
        未装配时本方法恒返回 False，判定行为与旧版逐字节一致。

        写入的历史项与 feed 同格式 (frame_idx, cx, cy)，因此补检命中后，
        本帧以及时间窗内后续帧的 _check_yolo_near_hoop 都会看到它。
        """
        if self.yolo_probe is None or self._cur_frame is None:
            return False
        if self._last_probe_frame == frame_idx:
            return False
        self._last_probe_frame = frame_idx
        self.diag["probe_called"] += 1
        try:
            balls = self.yolo_probe(self._cur_frame, frame_idx)
        except Exception:
            # 补检是兜底路径，失败不能影响主判定（主判定此时已是"否决"）
            return False
        if not balls:
            return False
        for _bp in balls:
            self.ball_pos_history.append((frame_idx, _bp[0], _bp[1]))
        return True

    def _check_yolo_near_hoop(self):
        """检查 YOLO 历史中，篮筐附近是否有球。

        返回: (有球: bool, 用于诊断: 'confirmed'/'probe'/'rejected'/'skipped')

        接受范围按篮筐框外扩，且**向下额外放宽**：进球是单向过程，球必定
        穿过筐口落到下方，实测漏检样本的球心落在筐下 ~97px、筐左 ~92px，
        原先四周统一 1.0 倍外扩刚好把它们挡在外面（差 10~27px），导致
        真实进球被硬否决。横向 1.5 倍、向下 2.0 倍可覆盖该偏移。

        整窗没球时先走一次「备用档补检」（见 _probe_missing_yolo）：默认档
        （裁剪@640）会系统性压低真球置信度，整窗一帧都不过阈值的事件才会漏球
        （如 2026.08.31-3rd 的 449.16s）；补检命中则本帧按 confirmed 记账。
        """
        if not self.yolo_confirm:
            return True, "skipped"
        x_lo, y_lo, x_hi, y_hi = self.yolo_accept_box()

        def _has_ball():
            return any(x_lo <= bx <= x_hi and y_lo <= by <= y_hi
                       for (_, bx, by) in self.ball_pos_history)

        if _has_ball():
            return True, "confirmed"
        if self._probe_missing_yolo(self._cur_frame_idx):
            if _has_ball():
                self.diag["probe_confirmed"] += 1
                return True, "probe"
        return False, "rejected"

    def yolo_accept_box(self):
        """YOLO 证据的接受框 (x_lo, y_lo, x_hi, y_hi)，整帧坐标。

        横向 ±1.5×hoop_w、上 1.0×hoop_h、下 2.0×hoop_h（向下额外放宽的理由
        见 _check_yolo_near_hoop 的 docstring）。

        **与检测端共用同一份几何**：services/detection._yolo_input_box 用它决定
        「裁剪推理」的输入范围，输入范围必须 ⊇ 本框，否则会重现
        「球在筐附近、却不在推理输入里」的静默漏检。
        """
        return (self.hoop_x1 - self.hoop_w * 1.5,
                self.hoop_y1 - self.hoop_h * 1.0,
                self.hoop_x2 + self.hoop_w * 1.5,
                self.hoop_y2 + self.hoop_h * 2.0)

    # ============ 断点续识别：状态序列化 ============

    def get_state(self) -> dict:
        """导出检测器动态状态（用于断点续识别）。

        仅保存运行时会变化的状态；构造参数（hoop 等）由 run_detect 用相同
        参数重建检测器后传入 set_state 恢复。例外：diff_threshold 在自适应
        预热完成时会被改写，必须随之保存（否则恢复后两条阈值路径分裂）。
        """
        state = {
            # ---- 进球状态机 ----
            "goals": list(self.goals),
            "last_goal_frame": self.last_goal_frame,
            "blob_above_hoop": self.blob_above_hoop,
            "blob_in_hoop": self.blob_in_hoop,
            "blob_in_hoop_frames": self.blob_in_hoop_frames,
            "blob_persistent_frames": self.blob_persistent_frames,
            "blob_history": list(self.blob_history),       # deque → list
            "last_blob_box": list(self.last_blob_box) if self.last_blob_box is not None else None,
            "last_above_frame": self.last_above_frame,
            "last_in_hoop_frame": self.last_in_hoop_frame,
            "last_diff_ratio": self.last_diff_ratio,
            # ---- YOLO 球位置历史 ----
            "ball_pos_history": list(self.ball_pos_history),  # deque → list
            # ---- 基准帧（滚动基准帧可能已偏离初始标定帧，必须保存）----
            "baseline_gray_b64": None,
            "last_baseline_frame_idx": self.last_baseline_frame_idx,
            # ---- 滚动基准帧候选（60s 窗口内运动量最小帧）----
            # 候选不序列化的话，恢复后候选池从恢复点清零重积：恢复的
            # blob_persistent_frames 已超 5s 阈值时首次触发只能取恢复后第一帧
            # 当基准，若该帧篮下有球员 → 该区域 diff≈0 形成检测盲区（漏检）
            "_baseline_candidate_frame_b64": None,
            "_baseline_candidate_diff": (None if self._baseline_candidate_diff == float("inf")
                                         else self._baseline_candidate_diff),
            "_baseline_candidate_idx": self._baseline_candidate_idx,
            # ---- 自适应阈值 / 预热 ----
            "_warmup_done": self._warmup_done,
            "_auto_threshold_value": self._auto_threshold_value,
            "_warmup_p95_median": self._warmup_p95_median,
            "_warmup_sample_count": self._warmup_sample_count,
            # diff_threshold：预热完成时被改写为 clamp 后的自适应值，必须保存；
            # _warmup_p95s：预热中途断点续跑需要样本列表继续累计（半开区间未
            # 完成预热的场景，恢复后不重新收集导致阈值计算偏差）
            "diff_threshold": int(self.diff_threshold),
            "_warmup_p95s": list(self._warmup_p95s),
            # ---- 诊断计数器 ----
            "diag": dict(self.diag),
        }
        if self.baseline_gray is not None:
            # ROI 灰度图 ~40KB，PNG 编码后更小；base64 便于 JSON 存储
            ok, buf = cv2.imencode(".png", self.baseline_gray)
            if ok:
                state["baseline_gray_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
        if self._baseline_candidate_frame is not None:
            ok, buf = cv2.imencode(".png", self._baseline_candidate_frame)
            if ok:
                state["_baseline_candidate_frame_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
        return state

    def set_state(self, state: dict) -> None:
        """从 get_state() 导出的字典恢复检测器动态状态。"""
        if not state:
            return
        # ---- 进球状态机 ----
        self.goals = list(state.get("goals", []))
        self.last_goal_frame = int(state.get("last_goal_frame", -1))
        self.blob_above_hoop = bool(state.get("blob_above_hoop", False))
        self.blob_in_hoop = bool(state.get("blob_in_hoop", False))
        self.blob_in_hoop_frames = int(state.get("blob_in_hoop_frames", 0))
        self.blob_persistent_frames = int(state.get("blob_persistent_frames", 0))
        bh = state.get("blob_history", [])
        self.blob_history = deque(bh, maxlen=30) if bh else deque(maxlen=30)
        lbb = state.get("last_blob_box")
        self.last_blob_box = tuple(lbb) if lbb is not None else None
        self.last_above_frame = int(state.get("last_above_frame", -999))
        self.last_in_hoop_frame = int(state.get("last_in_hoop_frame", -999))
        self.last_diff_ratio = float(state.get("last_diff_ratio", 0.0))
        # ---- YOLO 球位置历史 ----
        bph = state.get("ball_pos_history", [])
        self.ball_pos_history = deque(bph) if bph else deque()
        # ---- 基准帧 ----
        b64 = state.get("baseline_gray_b64")
        if b64:
            try:
                raw = base64.b64decode(b64)
                arr = np.frombuffer(raw, dtype=np.uint8)
                self.baseline_gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            except Exception:
                self.baseline_gray = None  # 解码失败：由 feed 首帧重建
        self.last_baseline_frame_idx = int(state.get("last_baseline_frame_idx", -1))
        # ---- 滚动基准帧候选 ----
        cand_b64 = state.get("_baseline_candidate_frame_b64")
        if cand_b64:
            try:
                raw = base64.b64decode(cand_b64)
                arr = np.frombuffer(raw, dtype=np.uint8)
                cand = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if cand is None:
                    raise ValueError("candidate frame decode failed")
                self._baseline_candidate_frame = cand
            except Exception:
                # 候选帧解码失败：只恢复标量，候选置空
                # （下次触发基准更新时回退用当帧，候选重新累积）
                self._baseline_candidate_frame = None
        else:
            self._baseline_candidate_frame = None
        cand_diff = state.get("_baseline_candidate_diff")
        self._baseline_candidate_diff = float(cand_diff) if cand_diff is not None else float("inf")
        self._baseline_candidate_idx = int(state.get("_baseline_candidate_idx", -1))
        # ---- 自适应阈值 / 预热 ----
        # 旧版断点（_warmup_done 键尚不存在时生成）一律视为"预热已完成"（B8）：
        # 当前流水线里预热由 run_detect 的独立前置 pass 完成，能走到 set_state
        # 的 checkpoint 要么 auto_threshold=False（无预热概念）、要么已带
        # _auto_threshold_value（预热值已持久化，缺 _warmup_done 的旧断点正是
        # 预热完成后保存的）。若把缺失键默认成 not auto_threshold=False，
        # 恢复后的正式检测器会把恢复点起前 ~30s 当成预热期跳过进球判定、
        # 并用新样本重算阈值覆盖断点阈值 → 双重重预热、结果与断点漂移。
        self._warmup_done = bool(state.get("_warmup_done", True))
        self._auto_threshold_value = state.get("_auto_threshold_value")
        self._warmup_p95_median = state.get("_warmup_p95_median")
        self._warmup_sample_count = int(state.get("_warmup_sample_count", 0))
        # ---- 阈值 / 预热样本（旧格式断点无这些键时保持构造值，向后兼容）----
        if "diff_threshold" in state:
            self.diff_threshold = int(state["diff_threshold"])
        _p95s = state.get("_warmup_p95s")
        if _p95s is not None:
            self._warmup_p95s = [float(v) for v in _p95s]
        # ---- 诊断计数器 ----
        saved_diag = state.get("diag", {})
        for k in self.diag:
            if k in saved_diag:
                self.diag[k] = int(saved_diag[k])
