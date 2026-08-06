# basketball-clipper

篮球录像进球检测与自动剪辑工具，专为**固定机位**比赛录像设计。手动标定篮筐后，自动检测进球时刻并剪辑集锦。

## 核心特性

- **固定机位免微调**：手动框选篮筐即可，无需重新训练模型
- **三种进球检测算法**：
  - `chonyy`：状态机（球依次经过 ABOVE → IN_HOOP → BELOW），适合正面视角
  - `diff`：基准帧差法（默认），通过帧差 + 连通域分析检测运动球穿越篮筐，适合底角斜向视角
  - `hybrid`：diff 快速筛选 + YOLO 全程确认，兼顾速度与准确率
- **多信号融合**：视觉信号（球穿越篮筐）+ 音频信号（进球音量峰值），可配置 `or/and/fused_only/visual_only` 融合模式
- **视频兼容性强**：PyAV 读取，支持 HEVC 编码和 moov atom 后置的 mp4；GPU 硬解硬编转码到 H.264
- **GPU 加速**：YOLO 推理与转码均调用 CUDA
- **可视化界面**：Gradio 多 Tab 界面，支持实时标定、参数调节、结果预览
- **YOLO 训练流水线**：自动抽帧、标注工具、训练脚本一应俱全

## 环境要求

- Windows / Linux
- Python 3.10+
- NVIDIA GPU（推荐 GTX 1650 4G 及以上），CUDA 环境
- FFmpeg（含 libx264 + nvenc，由 `imageio-ffmpeg` 自带）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 需安装 CUDA 版 PyTorch，参考 https://pytorch.org/get-started/locally/

### 2. 准备权重

将篮球检测 YOLO 权重放到 `weights/` 目录，命名为 `basketball_custom.pt`。若没有，首次运行会自动下载 `yolov8n.pt`（COCO 预训练，检测效果有限，建议自行训练）。

### 3. 启动交互式 Demo（推荐）

```bash
python demo_interactive.py
# 浏览器打开 http://127.0.0.1:7870
```

操作流程：
1. 输入视频文件路径（支持超大文件，不走浏览器上传）
2. 滑动到含篮筐的帧，点击画面 2 个点框选篮筐（同时采集无球基准帧）
3. 设置起止帧、检测算法、参数
4. 点击「开始检测」→ 生成可视化视频并在界面内播放

### 4. 启动完整界面

```bash
python app.py
# 浏览器打开 http://127.0.0.1:7862
```

三个 Tab：
- **标定与追踪**：框选篮筐 + YOLO 检测球 + 进球检测
- **YOLO 检测可视化**：调试球检测置信度、TTA
- **完整流程**：一键跑 pipeline 输出集锦 mp4

### 5. 命令行运行

```bash
# chonyy 状态机 demo
python demo_chonyy.py "path/to/video.mp4" 0 3000

# 完整流程（检测 + 剪辑）
python pipeline.py --video path/to/game.mp4 --config config.yaml
```

## 项目结构

```
basketball-clipper/
├── app.py                  # Gradio 完整界面（3 Tab）
├── demo_interactive.py     # 交互式进球检测 demo（推荐入门）
├── demo_chonyy.py          # chonyy 状态机 demo
├── pipeline.py             # 命令行主流程：检测 → 剪辑
├── tracker.py              # 基准帧差法 GoalDetector + 音频融合
├── video_io.py             # PyAV 视频读取（HEVC/moov 兼容）
├── transcoder.py           # GPU 硬解硬编转码
├── audio_detector.py       # 音频峰值检测
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖
├── auto_train.py           # 自动抽帧 + 伪标注 + 训练
├── train.py                # YOLO 训练脚本
├── label_tool.py           # 手动标注工具
├── cutter/ffmpeg_cutter.py # ffmpeg 剪辑
├── detector/               # 球与篮筐检测器
├── logic/shot_judge.py     # 进球判断逻辑
└── weights/                # YOLO 权重（不入库，用 Release 发布）
```

## 进球检测算法详解

### diff 基准帧差法（默认，适合底角视角）

1. 标定时取一帧无球画面作为**基准帧**
2. 每帧与基准帧做灰度差分 → 二值化 → 形态学去噪 → 连通域分析
3. 在篮筐周边搜索区域找最大运动连通域 = 运动的篮球
4. 跟踪斑块：斑块先经过篮筐上沿 → 后经过篮筐下沿 = 视觉进球
5. 融合阶段：视觉进球 ± 时间窗口内有音频峰值 = 高置信度进球

**可调参数**（demo 界面「diff 高级参数」面板）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 帧差阈值 | 25 | 低=灵敏（漏检多降到 15）/ 高=严格（误报多提到 35）|
| 最小斑块面积 | 50 | 过滤噪声 |
| 最大斑块面积 | 5000 | 过滤大物体（人）|
| 搜索范围 | 60px | 篮筐周边搜索运动物体的范围，漏检则扩大 |

### chonyy 状态机（适合正面视角）

球的位置相对篮筐分三段，必须按顺序经历：`IDLE → ABOVE → IN_HOOP → BELOW = GOAL`。依赖 YOLO 检测球，不依赖基准帧差。

### hybrid 混合模式

diff 算法快速筛选候选进球 → YOLO 全程检测球确认是否在篮筐附近穿越。第一遍 YOLO 检测结果缓存，第二遍直接复用，避免重复推理。

## YOLO 训练流程

### 1. 手动标注（推荐，质量高）

```bash
# 追加标注 200 帧
python label_tool.py --video "video.mp4" --frames 200

# 指定时间范围抽帧
python label_tool.py --video "video.mp4" --frames 100 --start 1000 --end 2000

# 用现有模型预标注辅助
python label_tool.py --video "video.mp4" --frames 100 --prelabel
```

快捷键：`n` 下一帧 / `p` 上一帧 / `s` 跳过 / `u` 撤销 / `a` 预标注 / `d` 删除 / `w` 保存 / `q` 保存退出

### 2. 自动训练（伪标注，快速但需人工清洗）

```bash
python auto_train.py --video "video.mp4" --frames 300 --epochs 50
```

自动抽帧 → 颜色 + 形状约束伪标注 → 训练 → 输出 `weights/basketball_custom.pt`

### 3. 训练建议

- 推理分辨率 `imgsz=1280`，提升小目标检出
- 置信度阈值 `0.3-0.4` 平衡检出率与误报
- 加入硬负样本（橙色非球物体）减少误检
- 数据增强：HSV 色调、旋转、缩放、Mosaic、Mixup

## 配置说明

`config.yaml` 关键字段：

```yaml
model:
  weights: weights/basketball_yolov8n.pt   # 权重路径
  device: cuda:0                           # GPU 设备
  conf_threshold: 0.4                      # 置信度阈值
  classes: [basketball, hoop]

hoop:
  auto_calibrate: true                     # 自动标定篮筐
  manual_box: [1575, 246, 1634, 301]       # 手动标定的篮筐框

judge:
  hoop_above_margin: 30                    # 篮筐上沿容差
  ball_motion_threshold: 100               # 球运动阈值
  frame_batch: 3                           # 帧批处理

cutter:
  pre_roll: 5                              # 进球前保留秒数
  post_roll: 5                             # 进球后保留秒数
  min_gap: 8                               # 最小剪辑间隔
  codec: libx264                           # 输出编码
```

## 常见问题

**Q: 上传大视频中断？**
A: `demo_interactive.py` 改用文件路径输入（不走浏览器上传），支持任意大小视频。缓存目录已设到 `E:\bball_cache`，避免 C 盘空间不足。

**Q: 视频显示 "video not playable"？**
A: 非 H.264 编码（如 HEVC）浏览器不支持。点击界面「转码预览」按钮，或用 `transcoder.py` 转码到 H.264。

**Q: Gradio 报 `ERR_ABORTED` 访问结果视频失败？**
A: Gradio 安全机制限制非允许路径。已将 `E:\bball_cache\demo_output` 和整个 `E:\bball_cache\gradio` 加入 `allowed_paths`，输出目录用 Gradio 临时目录。若仍有问题，关闭旧标签页用无痕窗口访问。

**Q: YOLO 检测不到球 / 误检橙色腿为球？**
A: ① 用 `label_tool.py` 手动标注高质量数据重训；② 降低置信度阈值；③ YOLO 检测已加几何过滤（长宽比 0.4-2.5、面积范围），可进一步收紧。

**Q: 进球漏检多？**
A: 用 diff 算法，在「diff 高级参数」面板：降低帧差阈值（如 15）、减小最小斑块面积、扩大搜索范围。查看诊断信息中「运动斑块检测率」，低于 50% 说明参数太严格。

**Q: 进球误报多？**
A: 提高帧差阈值（如 35）、增大最小斑块面积、开启音频融合 `and` 模式要求视觉+音频双确认。

## 技术参考

- 进球检测算法参考：chonyy/basketball-shot-detection
- 基准帧差法论文：Camera-based Basketball Scoring Detection Using CNN
- 多信号融合参考：ClarkWang1214/basketball-highlights
- 视频读取：PyAV（替代 OpenCV，兼容 HEVC 和 moov 后置 mp4）

## License

MIT
