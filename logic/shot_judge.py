"""进球判断状态机。

移植自 chonyy/basketball-shot-detection 的 detect_shot() 核心逻辑，
适配 YOLO 的 xyxy 输出格式，去掉了 OpenPose 头部排除部分
（固定机位下球与人头部不太会混淆，必要时可加回 person 检测过滤）。

原项目核心判断思路（utils.py 第 158-217 行）：
  1. 球升到篮筐高度以上 → 进入"投篮中"状态，记录轨迹
  2. 球下落到篮筐高度以下且位移小 → 判定是否进框
  3. 球的 x 坐标在篮筐 [xmin, xmax] 范围内 → 进球，否则未进
"""
import math


def distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


class ShotJudge:
    def __init__(self, hoop_box, ball_motion_threshold=100, hoop_above_margin=30):
        """hoop_box: 标定好的篮筐框 [x1,y1,x2,y2]。
        """
        self.hoop_xmin = min(hoop_box[0], hoop_box[2])
        self.hoop_xmax = max(hoop_box[0], hoop_box[2])
        self.hoop_ymin = min(hoop_box[1], hoop_box[3])
        self.hoop_ymax = max(hoop_box[1], hoop_box[3])
        self.hoop_height = self.hoop_ymin  # 篮筐上沿高度（原项目取 ymin）
        self.ball_motion_threshold = ball_motion_threshold
        self.hoop_above_margin = hoop_above_margin

        self.prev_ball = None
        self.is_shooting = False
        self.trajectory = []
        self.score_count = 0
        self.miss_count = 0

    def feed(self, ball_box):
        """喂入一帧的球检测框 [x1,y1,x2,y2]，无检测传 None。

        返回: None | 'SCORE' | 'MISS'
        """
        if ball_box is None:
            return None

        x1, y1, x2, y2 = ball_box[:4]
        x_coor = (x1 + x2) / 2
        y_coor = (y1 + y2) / 2
        ymin = y1

        event = None

        # 1. 球升到篮筐高度以上 → 进入投篮状态，记录轨迹
        if ymin < self.hoop_height:
            if not self.is_shooting:
                self.is_shooting = True
            self.trajectory.append((x_coor, y_coor))

        # 2. 球下落到篮筐附近且位移小 → 判定
        elif (ymin >= self.hoop_height - self.hoop_above_margin
              and self.prev_ball is not None
              and distance([x_coor, y_coor], self.prev_ball) < self.ball_motion_threshold):
            if self.is_shooting:
                # 3. 关键判断：球的 x 坐标是否在篮筐 [xmin, xmax] 范围内
                if self.hoop_xmin <= x_coor <= self.hoop_xmax:
                    event = "SCORE"
                    self.score_count += 1
                else:
                    event = "MISS"
                    self.miss_count += 1
                self.is_shooting = False
                self.trajectory.clear()

        self.prev_ball = [x_coor, y_coor]
        return event
