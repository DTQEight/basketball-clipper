# basketball-clipper

篮球录像进球检测与自动剪辑工具。专为**固定机位**比赛录像设计，手动框住篮筐+篮网（标定）后自动检测进球时刻并 GPU 加速剪辑集锦。

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
| 14 | 人物分类 + 按人物导出 | 卡片彩色徽章归属人物；个人集锦 `{视频名}-{人物}-highlights.mp4`；**整场集锦**四节合并 `{文件夹名}-{人物}-highlights.mp4`；全局名单跨场次复用 |
| 15 | 断点续跑 + 检测状态序列化 | 300 帧自动存档；续跑跳过预热、恢复阈值/基准帧/滚动候选完整状态 |
| 16 | NiceGUI 可视化界面 | 深色主题卡片式；检测中可随时取消；三级调试日志 |
| 17 | 多线程解码（FRAME 线程） | PyAV 由 SLICE 改 FRAME，解码 130 → 285 帧/秒，端到端提速 1.36×，候选结果逐项不变 |
| 18 | 四臂集成 AI 复核 + 自动 √ | A(手工特征 LGBM) + B(SimCLR 时序) + Flow(SimCLR 光流) + VM(VideoMAE) 加权集成，检测后自动给高分候选打 √ 免人工确认；**只标记不删候选**、人工标记优先；六场真实录像 286 候选实测自动 √ 精度 0.988、召回 80.6%；新场地单场（59 候选）高带 15 个精度 1.000、低带零漏球，人工复核量 59 → 26（−56%） |

更完整说明 👉 [doc/ALGORITHM.md](doc/ALGORITHM.md) / [doc/BENCHMARKS.md](doc/BENCHMARKS.md)

## 环境要求

- Windows / Linux
- Python 3.10+
- NVIDIA GPU（**必需**，推荐 GTX 1650 4G 及以上；CUDA 不可用时服务拒绝检测，不做 CPU 降级）
  - 支持的显卡世代：Maxwell（GTX 750/900 系）～ Blackwell（**RTX 50 系**）全覆盖，取决于 torch 构建的算力列表，见下方安装说明
- FFmpeg（含 libx264 + h264_nvenc，由 `imageio-ffmpeg` 自带）

## 快速开始

```bash
# 1. 安装依赖（需 Python 3.10+ / NVIDIA GPU）
pip install -r requirements.txt
# torch 必须用 cu128 且固定 2.7.1：
#   cu121 只编译到 sm_90，RTX 50 系（sm_120）会报 "no kernel image is available"；
#   torch>=2.11 的 cu128 构建移除了 Maxwell/Pascal/Volta，会丢失 GTX 10 系及更老显卡的支持。
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# 2. 把 YOLO 权重放到 weights/basketball_ft.pt（缺失时自动回退 yolov8n.pt 并下载）
#    AI 复核的模型权重 training/model_*.pt 不在仓库内（.gitignore 排除 *.pt），
#    缺失时对应臂自动禁用，检测与人工标记不受影响
# 3. 启动服务
python demo_nicegui.py    # Windows 双击 start.bat / Linux 运行 ./start.sh
# 浏览器打开 http://127.0.0.1:7871
```

**单视频流程**：输入视频路径 → 加载 → 帧选择器选帧 + 2 点框住篮筐+篮网 → 设置参数 → 开始识别 → **AI 复核自动 √ 高分候选**（A 臂逐帧复检约 5~6 秒/候选）→ 卡片 √/× 标记 + 👤 人物分类 → 导出集锦（可按人物筛选）

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
│   ├── goal_verifier.py    # 四臂集成 AI 复核：逐候选打分 + 达标自动 √
│   ├── state.py            # 历史记录（含 labels 标签池）/ 断点 / 片段缓存 / 全局人物名单
│   └── video_utils.py      # 视频元信息 / 帧转码
├── training/               # 离线训练流水线：四臂特征抽取 / 训练 / 集成搜索与阈值标定
├── cutter/ffmpeg_cutter.py # ffmpeg 剪辑（NVENC 探测 / 流拷贝拼接）
├── start.bat / start.sh    # Windows / Linux 一键启动
└── doc/                    # 文档（算法 / 性能 / FAQ / 训练 / 输出 / 更新日志）
```

完整数据流图 👉 以下链接：[训练指南](doc/TRAINING.md) · [FAQ](doc/FAQ.md)

## 更新日志

最新版本 **2026.09.22**：修复三处影响可用性的问题——**手机 HDR 源（10-bit HLG）的预览片段在浏览器里播不了**（NVENC 回退软编时 libx264 按输入位深编出 `h264 High 10`，Chrome 的 H.264 解码链只吃 8-bit → 黑屏且无任何报错；编码参数改为钉死 8-bit SDR 并用 `setparams` 把颜色标记落到 bt709）、**加载 `E:\ball\*.mov` 永久卡死**（PyAV 帧线程死锁，`read_frame` 关闭帧级多线程解码；同时修掉 `start.bat` 用 PowerShell 管道转发 stdout 导致控制台不消费时整条事件循环冻住）；另新增历史记录留档 **AI 当次分带 `verify_snapshot`**（人工确认会把 `labels.auto_*` 搬进 `kept`/`deleted`，模型当次判了什么此前不可回溯、判对率趋势无从统计）。随后补上**真正的 HDR→SDR 映射**——8-bit 化只是位深降级，BT.2020 原色当 BT.709 用会让球场青/粉明显偏淡；实测选定 `hable npl=400 + curves` 链（不映射 mean 150 / 教科书 npl=1000 mean 63 / 选定 mean 113，三者都零爆白，纯色彩学转换会爆掉 17.5% 像素），**BT.2390 因打包的 ffmpeg 未编 libplacebo 不可用**（`mobius` 属同族膝盖曲线，已实测留档备切）；映射**只作用于预览片段与集锦/单球导出**，检测与特征抽取仍直接读源文件。
上一版 **2026.09.22**：**四臂 AI 复核从特性分支整体合入主线**（`feat/4arm-goal-verifier` → main，双父合并 `0178539`）——发布线的检测与 AI 复核自此同属一条代码线，不再需要两处手工同步。合入同时做了三路代码审核（四臂模块 / 训练脚本 / 合并自洽性），修掉两处严重问题：**重检测会清空用户已标的人工 √/×**（`_sync_marks` 把"本次新片段没有人工标记"当成"清空"，已加 `write_manual` 开关 + 复现用例与接线守卫）、**`training/train_temporal.py` 整表覆盖 `model_temporal_meta.json` 会丢掉 `ensemble` 段**（线上三带阈值与权重静默回落到未标定的代码默认值，已改为读-改-写 + 原子替换）；另修 `invalidate_stale` 漏清 `auto_reject`、`build_dataset` 丢掉 `label_source`（模型自动 √ 会静默混进真值）两处。
上一版 **2026.09.21**：两段式兜底（默认裁剪@640 + 即将被否决时抹黑画布@1280 补检，真球召回 116/119 → 119/119）、球检测权重换为「按部署几何重训」版（双盲测日真球 103/103、候选 +1.4%、耗时 −6.2%）、静止球证据门（堵住「静止假球让 YOLO 硬否决失效」导致的连续 87 秒误报）、A 臂按新权重重抽重训（OOF AUC 0.8338 → 0.8404）并补上口径指纹漏掉的 A 臂模型与球检测权重、修复单球导出点了没反应。
更早 **2026.09.20**：条件跳过判据改为与斑块门同构（消灭「斑块触发却跳过 YOLO」的静默漏球）、裁剪推理（只推筐接受框 + `imgsz` 1280 → 640，单场实测耗时 −34%）、预览切片打点（把 `preview` 计时拆成 detect / preview / verify / tail 四段）。
更早 **2026.09.19**：**四臂集成 AI 复核上线**（检测后自动 √ 高分候选，五场真实录像实测自动通过精度 100%）、漏检根因修复（YOLO 接受框外扩）、RTX 50 系显卡支持（cu121 → cu128）、Windows 安装程序（Inno Setup）、FRAME 线程解码（端到端提速 1.36×、结果零变化）。

完整变更记录（含 08.18~08.19 版本进化对比报告、陌生场次泛化验证、三代纵向对比、审查修复、P2/P3 记录不修等）👉 [doc/CHANGELOG.md](doc/CHANGELOG.md)  
性能与识别质量对比报告（含解码线程优化前后对比、3rd/4th 新旧版识别差异球、陌生场次 4 节批量实测、球检测权重三代对比与双盲测日方法论）👉 [doc/BENCHMARKS.md](doc/BENCHMARKS.md)
