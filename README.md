# basketball-clipper

篮球录像进球检测与自动剪辑工具。专为**固定机位**比赛录像设计，手动标定篮筐后自动检测进球时刻并 GPU 加速剪辑集锦。

## 核心特性

| # | 能力 | 关键细节 |
|---|------|----------|
| 1 | 固定机位免微调标筐 | 2 点框选篮筐即可，无需重训 |
| 2 | diff 帧差 + YOLO 双通道 | 高召回候选 + 三条进球路径硬否决 |
| 3 | 自适应阈值 + 滚动基准帧 | 30s 预热 P95 中位数；60s 自动换基准帧 |
| 4 | 高召回候选 + √/× 人工标记 | 卡片绿/红样式反馈；顶部统计；toggle 取消 |
| 5 | 训练池直通标签飞轮 | labels.kept / deleted 时间戳增量写 |
| 6 | 帧选择器（Catch-up 防抖） | 预览滑条选帧 + current_frame 与标定对齐 |
| 7 | 原画质集锦 + GPU 加速 | NVENC 探测，h264_nvenc cq=20 / libx264 crf=18 |
| 8 | GPU 硬性要求 + 启动自检 | 主线程真实 YOLO 推理自检；CUDA 不可用拒绝检测 |
| 9 | 提速模式 + 条件跳过 YOLO | 每 3 帧推理 / 篮筐无运动跳过，整体提速 40% |
| 10 | 视频兼容 + HEVC 容错解码 | PyAV；HEVC/moov 后置；NAL 损坏视频逐包跳过 |
| 11 | 跨平台 + 自定义缓存 | Windows/Linux；BBALL_CACHE_ROOT |
| 12 | 历史记录 + 片段缓存 | 每视频独立 JSON 持久化；重启复用预览；标签/标定跨会话恢复 |
| 13 | 文件夹批量 + 流水线确认 | 逐个标定后批量识别，视频完成即可预览/导出；标定从历史回填免重标 |
| 14 | 人物分类 + 按人物导出 | 卡片彩色徽章归属人物；个人集锦 `{视频名}-{人物}-highlights.mp4`；全局名单跨场次复用 |
| 15 | 断点续跑 + 检测状态序列化 | 300 帧自动存档；续跑跳过预热、恢复阈值/基准帧/滚动候选完整状态 |
| 16 | NiceGUI 可视化界面 | 深色主题卡片式；检测中可随时取消；三级调试日志 |

更完整说明 👉 [doc/ALGORITHM.md](doc/ALGORITHM.md) / [doc/BENCHMARKS.md](doc/BENCHMARKS.md)

## 环境要求

- Windows / Linux
- Python 3.10+
- NVIDIA GPU（**必需**，推荐 GTX 1650 4G 及以上；CUDA 不可用时服务拒绝检测，不做 CPU 降级）
- FFmpeg（含 libx264 + h264_nvenc，由 `imageio-ffmpeg` 自带）

## 快速开始

```bash
# 1. 安装依赖（需 Python 3.10+ / NVIDIA GPU）
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 2. 把 YOLO 权重放到 weights/basketball_custom.pt（首次运行会自动下载 yolov8n.pt）
# 3. 启动服务
python demo_nicegui.py    # Windows 双击 start.bat / Linux 运行 ./start.sh
# 浏览器打开 http://127.0.0.1:7871
```

**单视频流程**：输入视频路径 → 加载 → 帧选择器选帧 + 2 点标定篮筐 → 设置参数 → 开始识别 → 卡片 √/× 标记 + 👤 人物分类 → 导出集锦（可按人物筛选）

**批量流程**：输入文件夹 → 逐个标定（保存标定；已跑过的文件夹自动从历史回填标定）→ 批量识别 → 每个视频完成即可流水线查看/标记/分类/导出

详细参数说明 👉 [doc/ALGORITHM.md#可调参数](doc/ALGORITHM.md#可调参数)；输出规格 👉 [doc/OUTPUT.md](doc/OUTPUT.md)

## 项目结构

```
basketball-clipper/
├── demo_nicegui.py         # 入口：NiceGUI 界面 + 全流程编排
├── app.py                  # YOLO 模型加载与推理
├── tracker.py              # GoalDetector：diff + 自适应阈值 + YOLO 硬否决
├── video_io.py             # PyAV 视频读取（HEVC / moov 后置 / NAL 容错）
├── services/
│   ├── detection.py        # 检测调度 / 进度回调 / 三级日志 / 标记与集锦 / 人物分类
│   ├── state.py            # 历史记录（含 labels 标签池）/ 断点 / 片段缓存 / 全局人物名单
│   └── video_utils.py      # 视频元信息 / 帧转码
├── cutter/ffmpeg_cutter.py # ffmpeg 剪辑（NVENC 探测 / 流拷贝拼接）
├── start.bat / start.sh    # Windows / Linux 一键启动
└── doc/                    # 文档（算法 / 性能 / FAQ / 训练 / 输出 / 更新日志）
```

完整数据流图 👉 以下链接：[训练指南](doc/TRAINING.md) · [FAQ](doc/FAQ.md)

## 更新日志

最新版本 **2026.09.05**：人物分类工作流（彩色徽章 + 按人物导出个人集锦 + 全局名单跨场次复用）+ 批量标定跨会话回填 + 断点续跑 12 项缺陷修复（含自适应阈值失效、标签数据丢失、续跑重复帧）。

完整变更记录（含 08.18~08.19 版本进化对比报告、陌生场次泛化验证、三代纵向对比、审查修复、P2/P3 记录不修等）👉 [doc/CHANGELOG.md](doc/CHANGELOG.md)  
性能与识别质量对比报告（含 3rd/4th 新旧版识别差异球、陌生场次 4 节批量实测）👉 [doc/BENCHMARKS.md](doc/BENCHMARKS.md)
