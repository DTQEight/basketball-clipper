"""chonyy/basketball-shot-detection 原始进球检测逻辑 demo。

算法原理（chonyy 原版状态机）：
  球的位置相对篮筐分三段：
    - ABOVE: 球在篮筐上沿之上（y < hoop_top）
    - IN_HOOP: 球在篮筐框内（hoop_top <= y <= hoop_bot）
    - BELOW: 球在篮筐下沿之下（y > hoop_bot）

  状态机：
    IDLE → 球到 ABOVE → 球进 IN_HOOP → 球到 BELOW = GOAL
    （必须按顺序经历三个阶段，防止误判）

  附带约束：
    - 球的 x 坐标必须在篮筐 x 范围内（含容差）
    - 最小进球间隔（默认 3 秒）防止重复计数

依赖 YOLO 检测球，不依赖基准帧差法。

用法:
    E:\\basketball-project\\env\\python.exe demo_chonyy.py <视频路径> [开始帧] [结束帧]

示例:
    E:\\basketball-project\\env\\python.exe demo_chonyy.py "E:\\basketball-project\\cache\\2026.07.05 2nd.mp4" 0 3000
"""
import sys
import time
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from video_io import get_video_info, read_frame
from app import get_ball_model


# ==================== chonyy 原始进球检测状态机 ====================
class ChonyyGoalDetector:
    """chonyy/basketball-shot-detection 原始进球检测逻辑。

    基于单帧球位置的状态机：球从篮筐上方→进入篮筐→下方 = 进球。
    不用光流、不用基准帧差、不用音频，纯粹的"球位置穿越"判断。
    """

    # 状态
    STATE_IDLE = 0          # 等待球出现
    STATE_ABOVE = 1         # 球在篮筐上方
    STATE_IN_HOOP = 2       # 球进入篮筐区域

    def __init__(self, hoop_box, min_gap_sec=3.0, x_margin_ratio=0.5):
        """
        hoop_box: (x1, y1, x2, y2) 篮筐框
        min_gap_sec: 最小进球间隔（秒），防止重复计数
        x_margin_ratio: x 方向容差（篮筐宽度的比例）
        """
        self.hoop_x1, self.hoop_y1, self.hoop_x2, self.hoop_y2 = [int(v) for v in hoop_box]
        self.hoop_top = self.hoop_y1
        self.hoop_bot = self.hoop_y2
        self.hoop_cx = (self.hoop_x1 + self.hoop_x2) / 2
        self.hoop_w = self.hoop_x2 - self.hoop_x1

        self.min_gap_sec = float(min_gap_sec)
        self.x_margin = int(self.hoop_w * x_margin_ratio)

        # 状态机
        self.state = self.STATE_IDLE
        self.last_goal_frame = -1
        self.fps = 30.0

        # 球轨迹历史（调试用）
        self.ball_history = []  # [(frame_idx, cx, cy), ...]
        self.goals = []         # 进球时间戳列表

    def _ball_zone(self, cx, cy):
        """判断球相对篮筐的位置区域。"""
        # x 必须在篮筐范围内（含容差），否则视为不在篮筐附近
        if not (self.hoop_x1 - self.x_margin <= cx <= self.hoop_x2 + self.x_margin):
            return "OUT"

        if cy < self.hoop_top:
            return "ABOVE"
        elif cy > self.hoop_bot:
            return "BELOW"
        else:
            return "IN_HOOP"

    def feed(self, ball_pos, frame_idx, fps):
        """喂入一帧的球检测结果。

        ball_pos: (cx, cy, x1, y1, x2, y2, conf) 或 None
        frame_idx: 当前帧号
        fps: 帧率
        返回: 进球时间戳（秒）或 None
        """
        self.fps = fps

        # 冷却期不处理
        if self.last_goal_frame >= 0:
            gap_sec = (frame_idx - self.last_goal_frame) / fps
            if gap_sec < self.min_gap_sec:
                return None

        if ball_pos is None:
            # 没检测到球，状态保持但允许衰减
            return None

        cx, cy = ball_pos[0], ball_pos[1]
        self.ball_history.append((frame_idx, cx, cy))
        if len(self.ball_history) > 60:
            self.ball_history.pop(0)

        zone = self._ball_zone(cx, cy)
        ts = frame_idx / fps

        # ====== chonyy 状态机 ======
        if self.state == self.STATE_IDLE:
            if zone == "ABOVE":
                self.state = self.STATE_ABOVE

        elif self.state == self.STATE_ABOVE:
            if zone == "IN_HOOP":
                self.state = self.STATE_IN_HOOP
            elif zone == "BELOW":
                # 跳过了 IN_HOOP（可能帧率太低没采到），也认为是进球
                self._confirm_goal(frame_idx, fps)
                return ts
            elif zone == "OUT":
                # 球飞走了，重置
                self.state = self.STATE_IDLE

        elif self.state == self.STATE_IN_HOOP:
            if zone == "BELOW":
                # 完整穿越：上→中→下 = 进球
                self._confirm_goal(frame_idx, fps)
                return ts
            elif zone == "ABOVE":
                # 球又回到上方（弹出来），重置
                self.state = self.STATE_ABOVE
            elif zone == "OUT":
                self.state = self.STATE_IDLE

        return None

    def _confirm_goal(self, frame_idx, fps):
        """确认进球。"""
        ts = frame_idx / fps
        self.goals.append(ts)
        self.last_goal_frame = frame_idx
        self.state = self.STATE_IDLE

    def get_state_name(self):
        return {0: "IDLE", 1: "ABOVE", 2: "IN_HOOP"}.get(self.state, "UNKNOWN")


# ==================== 可视化 ====================
def draw_frame(frame, ball_pos, hoop_box, detector, frame_idx, fps):
    """在帧上画检测信息。"""
    out = frame.copy()

    # 画篮筐框（绿）+ 上沿/下沿线
    x1, y1, x2, y2 = hoop_box
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(out, (x1 - 30, y1), (x2 + 30, y1), (0, 255, 255), 1)  # 上沿线
    cv2.line(out, (x1 - 30, y2), (x2 + 30, y2), (255, 0, 255), 1)  # 下沿线
    cv2.putText(out, "HOOP", (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 画球（红）
    if ball_pos is not None:
        _, _, bx1, by1, bx2, by2, bconf = ball_pos
        cv2.rectangle(out, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 2)
        cv2.circle(out, (int(ball_pos[0]), int(ball_pos[1])), 5, (0, 165, 255), -1)
        cv2.putText(out, f"BALL {bconf:.2f}", (int(bx1), max(int(by1) - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 画状态机信息（左上角）
    state_name = detector.get_state_name()
    ts = frame_idx / fps
    info_lines = [
        f"Frame: {frame_idx} ({ts:.1f}s)",
        f"State: {state_name}",
        f"Goals: {len(detector.goals)}",
        f"FPS: {fps:.1f}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(out, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # 进球时刻闪烁红色边框
    if detector.goals and abs(ts - detector.goals[-1]) < 1.0:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 0, 255), 8)

    return out


# ==================== 主流程 ====================
def run_demo(video_path, hoop_box, start_frame=0, end_frame=None,
             ball_conf=0.3, min_gap_sec=3.0, save_output=True):
    """运行 chonyy 原始进球检测 demo。

    video_path: 视频路径
    hoop_box: (x1, y1, x2, y2) 篮筐框
    start_frame: 起始帧
    end_frame: 结束帧（None 表示到视频末尾）
    ball_conf: YOLO 球检测置信度
    min_gap_sec: 最小进球间隔（秒）
    save_output: 是否保存可视化视频
    """
    print(f"\n{'='*60}")
    print(f"chonyy/basketball-shot-detection 原始进球检测 demo")
    print(f"{'='*60}")
    print(f"视频: {video_path}")
    print(f"篮筐框: {hoop_box}")
    print(f"检测范围: 帧 {start_frame} - {end_frame or '末尾'}")
    print(f"球检测置信度: {ball_conf}")
    print(f"最小进球间隔: {min_gap_sec}s")
    print(f"{'='*60}\n")

    info = get_video_info(video_path)
    fps = info["fps"]
    total = info["total"]
    if end_frame is None or end_frame > total:
        end_frame = total
    print(f"视频信息: {total} 帧 | {fps:.1f} fps | {info['codec']}")

    # 初始化检测器
    detector = ChonyyGoalDetector(hoop_box, min_gap_sec=min_gap_sec)

    # 初始化 YOLO 模型
    print("加载 YOLO 模型...")
    model, weights_path = get_ball_model()
    print(f"使用权重: {weights_path}")

    # 可视化输出视频（保存到项目目录内，避免沙箱拦截）
    out_writer = None
    if save_output:
        out_dir = ROOT / "demo_output"
        out_dir.mkdir(exist_ok=True)
        out_path = str(out_dir / (Path(video_path).stem + "_chonyy_demo.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps,
                                     (info["width"], info["height"]))
        print(f"保存可视化到: {out_path}")

    print(f"\n开始检测...")
    t0 = time.time()
    processed = 0
    last_print = t0

    for fidx in range(start_frame, end_frame):
        frame = read_frame(video_path, fidx, total=total, fps=fps)
        if frame is None:
            continue

        # YOLO 检测球
        ball_pos = None
        try:
            res = model.predict(frame, conf=ball_conf, imgsz=1280,
                                device="cuda:0", verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
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
                if best is not None:
                    x1, y1, x2, y2 = xyxy[best]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    ball_pos = (float(cx), float(cy), float(x1), float(y1),
                                float(x2), float(y2), float(confs[best]))
        except Exception as e:
            pass

        # 喂入检测器
        goal_ts = detector.feed(ball_pos, fidx, fps)

        # 进球提示
        if goal_ts is not None:
            print(f"  ⚽ 进球! 帧 {fidx} ({goal_ts:.1f}s) - 状态: {detector.get_state_name()}")

        # 可视化 + 保存
        if out_writer is not None:
            annotated = draw_frame(frame, ball_pos, hoop_box, detector, fidx, fps)
            out_writer.write(annotated)

        processed += 1
        # 进度打印（每 5 秒一次）
        now = time.time()
        if now - last_print > 5:
            speed = processed / (now - t0)
            eta = (end_frame - fidx) / speed if speed > 0 else 0
            print(f"  进度: {fidx}/{end_frame} ({fidx/end_frame*100:.1f}%) "
                  f"| {speed:.1f} fps | ETA {eta:.0f}s | 进球: {len(detector.goals)}")
            last_print = now

    elapsed = time.time() - t0
    if out_writer is not None:
        out_writer.release()

    print(f"\n{'='*60}")
    print(f"检测完成")
    print(f"{'='*60}")
    print(f"处理帧数: {processed}")
    print(f"耗时: {elapsed:.1f}s ({processed/elapsed:.1f} fps)")
    print(f"检测到进球: {len(detector.goals)} 个")
    for i, ts in enumerate(detector.goals):
        print(f"  [{i+1}] {ts:.1f}s (帧 {int(ts*fps)})")
    print(f"{'='*60}\n")

    return detector.goals


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python demo_chonyy.py <视频路径> [开始帧] [结束帧]")
        print("示例: python demo_chonyy.py \"E:\\basketball-project\\cache\\2026.07.05 2nd.mp4\" 0 3000")
        sys.exit(1)

    video = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # 篮筐框（从 config.yaml 读取上次标定值）
    HOOP_BOX = (1575, 246, 1634, 301)

    print(f"使用篮筐框: {HOOP_BOX}（来自 config.yaml 上次标定）")
    print()

    goals = run_demo(video, HOOP_BOX, start_frame=start, end_frame=end)
