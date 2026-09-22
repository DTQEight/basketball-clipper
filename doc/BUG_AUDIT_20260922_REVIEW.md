# Bug 清单核验对比（对 `BUG_AUDIT_20260922.md` 的复审）

> 被核验清单：[BUG_AUDIT_20260922.md](BUG_AUDIT_20260922.md)（GLM-5.3 产出，基线 main @ `71af72d`，29 条）
> 核验方式：5 路独立复审（3 路分区逐条核验 + 1 路低危核验 + 1 路独立全面重审），核验者须实际打开代码，关键项做最小复现；本文作者另对最关键的 5 处逐行确认。
> 结论：**27 条有实质依据，2 条误报；但 13 条存在行号/根因/严重度偏差，需要修正**。另有 **20 条新发现**（含 1 条高危）。
> 状态：仅审核，未改动任何代码。

---

## 一、总览

| 分区 | 条数 | ✅属实 | 🟡部分属实（需修正） | ❌误报 |
|---|---|---|---|---|
| 🔴 R1–R9 | 9 | 4（R1 R2 R7 R9） | 5（R4 R5 R6 R8 + R9 严重度） | 1（R3） |
| 🟡 M1–M12 | 12 | 7（M1 M3 M4 M6 M7 M9 M12） | 4（M2 M5 M8 M11） | 1（M10） |
| 🟢 L1–L8 | 8 | 5（L2 L4 L5 L6 L7） | 3（L1 L3 L8） | 0 |
| **合计** | **29** | **16** | **11** | **2** |

**主要偏差类型**：
1. **1 条 🔴 误报**（R3）——依据的"刷新会取消 UI 协程"与 NiceGUI 3.16 实际行为相反，实测证伪。
2. **1 条 🟡 误报**（M10）——该代码分支在应用内不可达。
3. **6 条严重度偏高**（R4 R5 R6 R8 R9 → 应降 🟡；M6 M11 L7 建议降）——多为"方向对但后果夸大"。
4. **4 条行号/根因有偏差**（M2、L3、L8、R3 涉及的注释本身）。

---

## 二、需要修正的原清单条目

### 2.1 ❌ 误报（建议从清单删除或降级）

#### R3「检测/批量运行中刷新页面 → 任务收尾全丢」→ **❌误报，降为 🟢**
- **原断言**：`clear_live` 只在原页面协程里（`demo_nicegui.py:1120/:943`），刷新后协程死掉 → live 槽永不清空 → 按钮永显"取消"、进度停格、卡片不出现、点取消导致整场重检。
- **核验证伪**：NiceGUI 3.16 的事件协程属于**全局** background_tasks（`nicegui/events.py:486`），`client.delete()` **不取消**它们；delete 后协程照常恢复，`set_text` 退化为静默 no-op（`nicegui/element.py:415-432` 注释明写 "an async callback resuming after the teardown is not a user bug"）。故 `state.clear_live('detect')` 仍会执行，新页面 timer（:1820/:1831）正常复位——"永显取消/进度停格/整场重检"三条均不成立。
- **真实残留（🟢）**：新页面不会自动 `_refresh_result_cards()`，"暂无进球结果"需用户手动刷新/加载历史。
- **误报根源（值得单独修）**：代码注释本身写反了。`demo_nicegui.py:939 / 1111 / 1439 / 1525`、`services/detection.py:1439 / 1655`、`doc/CHANGELOG.md:1289` 都声称"刷新/断连会取消 UI 协程"，与 3.16 实际行为相反 —— 既误导了这轮审核，也会误导以后"把收尾挪进 finally 就能救场"的改法。
- **行动**：改成 🟢 并改写上述注释。

#### M10「静止球窗口不随 fps 重算」→ **❌误报，建议删除**
- **原断言**：`tracker.py:391-394` 只重算 above_timeout/yolo_window，static_ball_frames/gap 定格 → VFR 视频窗口时长偏一倍。
- **核验证伪**：重算块确实只含两项，但 `detection.py:764/853` 构造 tracker 时与 `feed` 均传同一 `fps`（:777/:1034），该分支在应用内**不可达**；"VFR 偏一倍"无依据（全程同一标称 fps）。
- **保留价值**：属 API 陷阱 + 测试覆盖缺口（`test_tracker.py:235` 只断言那两项）。建议移入"清理/加固项"，不进 bug 清单。

---

### 2.2 🟡 行号或根因需修正

| 条目 | 原断言 | 核验修正 |
|---|---|---|
| **M2** | 固化点在 `detection.py:2097-2099` | 实际固化点为 **`:2100-2102`**；且需"该片段本轮再打分失败"（`score_clips:744` continue / `refresh_auto:457` 跳过）才留存 —— `refresh_auto` 对非 manual 片段会 `c.pop("mark")`（:471）覆盖掉旧标记。**触发条件比原文窄** |
| **M8a** | 取消按钮 disable 被 0.4s 复位"防重复点击失效" | 属实但**后果仅观感**：再点仍走 cancel 分支（幂等 set），无功能影响 → 建议降 🟢。同源问题批量按钮也一样（`demo:874` + `:1831-1833`） |
| **L3** | 位置 `goal_verifier.py:404-415` | 实际 **`:399-416`**；且"不含 reject_thr"**不构成缺陷**（reject_thr 不参与 score 计算）。真正的问题只有：keep_thr 与 ENS_WEIGHTS 混入指纹（改阈值即触发全量重打分）+ **无条件**哈希互斥的兜底模型 mtime（:403-413） |
| **L8②** | `extract_features.py:290` 的 `grays[0]` IndexError 崩训练主流程；引 `detection.py:683` break | **两部分均不成立**：① `:257-259` 有 `if f1-f0 < 8: continue`，且 `f1=min(total,...)`，两条件互斥 → `grays[0]` 不可达；② `detection.py:683` 是三行注释，真正的 break 在 `goal_verifier.py:685`，且只影响**剩余候选的 A 臂**（B/flow/VM 仍打分，combine 会重归一化）→ 严重度大幅夸大。建议只保留"加 `if not grays:` 守卫"作为加固项 |

### 2.3 严重度偏高（建议下调）

| 条目 | 原级 | 建议 | 理由 |
|---|---|---|---|
| **R4** 混合编码器 concat 花屏 | 🔴 | 🟡 | 方向对（回退确从第 k 段起生效、concat `-c copy` 确无一致性校验），但① 依赖"导出中途 NVENC 运行时失败"这一条件；② `-c copy` 报错会回落重编码兜底；③ 原文"本机 GTX 1650 无 NVENC、首段即整体回退"**事实存疑**——代码自注 `ffmpeg_cutter.py:139` 称 1650 有 1 个 NVENC 编码器，复审实测探测编码 exit=0。⚠️ 与本项目历史记录（"NVENC 全程失败 No capable devices found"）冲突，**需实测确认**该机 NVENC 到底能不能用，再定级 |
| **R5** eval_thresholds 循环评估 | 🔴 | 🟡 | 循环口径属实（`:34` 用 `label_sets`、`:45-46` 打分 AUC，auto_kept 必成 TP），但该脚本**只读打印、不写任何模型或阈值** → "污染线上模型 / keep_thr 标定被误导"言过其实 |
| **R6** train_lgbm 覆盖生产模型 | 🔴 | 🟡 | 文件路径与覆盖行为属实（`train_lgbm.py:27-28` = `goal_verifier.py:51-52`，无备份无开关），但这是**手动运行的离线训练脚本**，产出即线上模型属预期设计；真正缺陷仅是"无备份无确认"。另：mtime 进指纹导致重打分是**设计行为**（口径变了就该重算），不算 bug |
| **R8** annotate.py 导入名错 | 🔴 | 🟡 | 导入名错属实（`ffmpeg_cutter` 只有 `build_encode_args`，无下划线版）→ 永远走 except 兜底。但后果写错了：标注片段由 `_open_clip`（`annotate.py:376`）`os.startfile` 交**系统默认播放器**，不经浏览器 → 应为"默认播放器可能播不出/色偏"，且不影响模型与线上产物 |
| **R9** 指纹不含臂组合 | 🔴 | 🟡 | 机制全部属实（指纹 :404-416 只哈希 7 个模型文件 mtime + `weights/*.pt` mtime + ENS_WEIGHTS + thr；combine :708-715 重归一化；`_tried` 闩死 :95/:166/:298 异常分支不重置），但需"某臂加载失败"才触发，且模块只打标不删候选（留人工兜底）→ 条件性风险 |
| **M6** 历史排序未防护 | 🟡 | 中偏低 | 属实（`state.py:666` 无 try 的 `float(ts)`），但应用内唯一写入者是 `time.time()`（:1029），触发需**外部改写 JSON** |
| **M11** 标定 TOCTOU | 🟡 | 低 | 守卫其实存在（`if state.current_task() is not None: return` + `_refuse_if_busy()`），只是检查与 `await run.io_bound(...)` 之间不持锁，窗口极窄 |
| **L7** tracker 无身份关联 | 🟢 | 保持/更低 | 属实（`blob_history` 仅在 `blob is None` 时清空 :512-517），但需同时通过 in_x、size_ok、circularity、blob_above_hoop、YOLO 硬否决、静止球剔除多重闸门，实际误检概率低 |

### 2.4 ✅ 完全属实（可放心按原文修）

R1、R2、R7、M1、M3、M4、M7、M9、M12、L2、L4、L5、L6 及 L7 的机制描述。

**其中 R1 的触发条件需扩大（重要）**：
- 原文只说"clips ⊂ goals（预览切片失败/片段被清理后重生成不完整）"。
- 核验发现**更高频的路径**：单视频**重检测之后**，`add_history` 只把旧标签重映射进历史记录，clips 上不带任何人工标记（只有批量/历史路径 `:2286/:2331` 才调 `_restore_labels_to_clips` 回填）→ 重检测后**第一次点 √/× 就把上一轮全部 kept/deleted 覆盖成"仅本次点的那个"**。
- 另：`:2101` 那条另有门槛（需 AI 开启且至少一个 auto 标记，`:2100`）。
- **修法需覆盖这两条路径**，不能只防"clips 子集"。

**R2 需补一句**：断连后并非立即删，NiceGUI `reconnect_timeout=3.0s`（`client.py:382-390`）后才 delete → 表述应为"刷新约 3 秒后断点被删"。

---

## 三、原清单遗漏的新发现

### 🔴 新增高危

#### N1. 提前 EOF 被当成"检测完成"：结果静默截断，且断点被删 ☐
- **位置**：`video_io.py:294-301`（容器层 demux 异常 → 计数后 `return`，静默结束迭代）+ `services/detection.py:1115`（只在 `processed == 0` 时判失败）+ `:1244`（成功分支 `delete_checkpoint(video_path)` 删全部断点）
- **触发**：截断/拷贝中断的 mp4（代码注释自述存在此场景，并有 `TestIterFramesDemuxTolerance` 专测）、VFR、容器 `frames` 头虚高 → `iter_frames` 在区间**中途**结束（自然 EOF 甚至不计 `decode_errors`）。
- **后果**：`processed > 0` → 走成功分支：写历史 + 删全部断点 + UI 显示"检测完成 | 处理 N 帧"，后半段进球永久缺失**且无法续跑**。
- **根因**：缺少 `processed` 与目标区间（`n_frames`）的完整性校验；`:1124-1126` 对 `decode_errors>0` 只打 WARNING 即"容错继续"——但该容错设计针对的是"中途损坏 NAL 可跳过"，不适用于"读到数据末尾"。
- **修法方向**：`processed < n_frames` 时按"未完成"处理（保留断点、历史/状态标 partial）。
- **置信度**：高（已逐行确认三处联动，且全仓无完整性校验）。

### 🟡 新增中危

#### N2. 断点指纹对 int/float 表示敏感 → "继续识别"实际从头跑 ☐
- **位置**：`services/state.py:321-333`（指纹用 `repr(items)`）+ `services/detection.py:700-710`（`_cp_params` 中 `min_gap_sec`、`ball_conf` **未做类型归一化**，而相邻所有键都显式 `int()/float()`）
- **触发**：`min_gap` 是 `ui.slider(min=1.0,max=10.0,value=2.0,step=0.5)`（`demo:390`）。默认值 2.0 是 float；用户拖动后 JS 回传整数值 `2` → Python `int` → `repr` 从 `2.0` 变 `2` → 指纹不同 → `load_checkpoint` 找不到断点。而断点弹窗用 `params=None` 探测（`demo:1025-1036`）→ 仍显示"可从断点继续" → 用户点继续却**静默从头重跑**。
- **修法**：指纹构建时统一 `float()`/`int()` 归一化（或只哈希数值本体）。
- **待确认**：需实测 NiceGUI 对 float 滑块的整数值回传类型（JS 无 int/float 之分，`2.0` 序列化为 `2` 属常见行为）。

#### N3. 断点指纹不含球检测权重 → 换 `weights/*.pt` 后混口径续跑 ☐
- **位置**：`services/state.py:312-318`（`_CHECKPOINT_PARAM_KEYS` 15 项，**不含任何模型/权重标识**）
- **触发**：README 的正常工作流就是替换 `weights/basketball_ft.pt`。AI 侧指纹已专门把 `weights/*.pt` mtime 计入（`goal_verifier.py:410-414`），断点侧却漏了 → 换权重后"继续识别"，断点前后用**两套权重**推理，结果口径分裂且无任何提示（同路径文件被替换亦无法识别）。
- **修法**：指纹加入检测权重文件 mtime（或 md5）。

#### N4. 循环评估口径扩散到标定侧（比原 R5 更该修）☐
- **位置**：`training/eval_quantile_rule.py:33`、`training/eval_arm_ablation.py:64`（同样直接用 `state.label_sets` 当真值）
- **更严重的一处**：`training/build_dataset.py:31` 默认把 `auto_kept` 当正样本 → 流入 `training/recalib_ensemble.py:63` 的 **OOF 阈值标定** → 标定侧同样存在"模型自己的判断当人工真值"的闭环，直接影响线上阈值正确性。
- **建议**：与 R5 合并为一条"训练/标定侧 label 口径"问题统一处理。

#### N5. M3 的连带面比原文更广 ☐
- `refresh_auto` 清掉内存 mark 后**未回写 `kept_goal_indices`**（`:2045` 载入值）→ 不止整场导出，**单视频集锦也按旧 auto √ 导出**。

#### N6. 「本轮打分失败即保留上一口径标记」语义冲突 ☐
- `services/goal_verifier.py:457`（refresh_auto 跳过）/ `:744`（score_clips continue）：某片段本轮打分失败时保留旧 `mark`，与"无分就该重跑/重推"的原则矛盾。建议无分时清 mark 或标 `stale`。

### 🟢 新增低危

| # | 问题 | 位置 | 说明 |
|---|---|---|---|
| N7 | 批量的"正在取消..."同样被 0.4s 复位 | `demo:874` + `:1831-1833` | 与 M8a 同源 |
| N8 | 单球导出成品被 7 天清理误删 | `state.py:126` | `_purge_old_clips` 只豁免 `-highlights.mp4`，而单球 HQ 导出落同一 `demo_output/` 且名为 `{源名}-goal-{ts}s.mp4`（`detection.py:1564-1567`）→ 一周后静默消失（docstring 却称"成品不清理"）。**已逐行确认** |
| N9 | `os.listdir` 无 try → OSError 穿透 UI | `video_utils.py:48` | 与 L6 的"静默返回空"不一致（网络盘中途断开/权限不足时抛给 UI 回调） |
| N10 | 集锦消息"N 球 → M 段"不反映实际写入 | `ffmpeg_cutter.py:462-464` + `detection.py:1696-1703` | 只有日志记跳过段数，消息用 `merge_segments` 的期望段数 → 用户以为全部入片 |
| N11 | 预览不校验文件存在即报成功 | `demo:1430-1436` | `clip_action` 恒返回理论路径；片段被驱逐/清理后点预览 → 无画面却提示成功 |
| N12 | 对话框元素累积不回收 | `demo:1044-1054 / 1716-1729` | 事件处理器内新建 `ui.dialog()`，`await` 后只 hide 不 delete → DOM/元素数随点击线性增长 |
| N13 | `goal_verifier.py:685` 的 break 语义偏重 | `goal_verifier.py:685` | 单批 `extract` 异常即放弃其后**全部候选**的 A 臂打分（其余臂仍在），使部分 clip 分数缺 A 臂、跨候选不可比 |
| N14 | `extract_features.py:290` 脆弱（当前不可达） | `extract_features.py:290` | 现逻辑不可达，但一旦 `iter_frames` 因解码错误只返回部分帧仍会 IndexError 且无 except 包裹 → 建议加 `if not grays:` 守卫 |
| N15 | `extract_features.py:336-337` 死代码 | `extract_features.py:336-337` | 残留 `... if False else ...` 三元式，无害但误导 |
| N16 | 注释路径漂移 | `annotate.py:17/34` | 仍写 `E:\basketball-project\training`，实为 `e:\basketball-clipper\training` |
| N17 | 单次 √/× 重写全部历史 | `state.py:784`（`save_history(records)`）+ `:687-691` | 每次标记触发 **O(N) 次原子写**（N=历史条数），随历史线性变慢，并与并发 `get_labels` 抢文件（代码自身的长重试即为此症状的补丁） |

---

## 四、对原清单的总体评价

**可信度较高的部分**：存储层与状态机类问题（M1–M7、M9、R1、R2）几乎全部经代码逐行证实，行号准确度高（多数只差 1–6 行），可以直接作为修复依据。

**需要保留怀疑的部分**：
1. **UI 并发/生命周期类断言**（R3、M8、M11）——这批最不可靠。R3 已被实测证伪，M8a/M11 被降级。共同根源是**依据了仓库里错误的注释**而非 NiceGUI 实际行为。以后再审此类问题应从 NiceGUI 源码取证。
2. **"会不会真触发"的判断**（M5、M6、M10、L8②）——原清单倾向于把"代码缺陷"直接等于"线上风险"。M5 现网 36 条历史 basename 互不相同，M6 需外部改 JSON，M10 分支不可达，都是**理论成立、当前不触发**。
3. **严重度分层**——9 条 🔴 中实际只有 4 条够格（R1 标签清空、R2 断点误删、R7 标签冻结，以及新发现的 N1 提前 EOF）。其余 5 条 🔴 应降 🟡：R4/R5/R6/R8/R9 都不是"错误结果/崩溃/数据丢失"级。

**原清单最有价值的贡献**：R1（含核验后扩大的触发条件）、R2、R7、M3、M4、M6、M9、M12 —— 这些是真实且会咬人的问题。

**方法论教训（建议写入项目记忆）**：本次 R3 误报的直接原因是**代码注释与框架实际行为相反**。审核结论不应建立在仓库自带注释上，尤其是涉及第三方框架生命周期的断言。

---

## 五、修正后的修复优先级

1. **数据丢失级（真 🔴）**
   - R1 人工标签被清空（**扩大触发条件**：重检测后首次打标即清空）
   - R2 断点误删（补 `is None` 分支）
   - **N1 提前 EOF 静默截断 + 删断点**（新增，建议与 R2 一起改，同属断点生命周期）
   - R7 改标后重训用旧标签
2. **口径正确性（🟡，影响模型）**
   - N4 + R5 训练/标定侧 label 口径（**含 `build_dataset.py:31` → `recalib_ensemble.py:63` 标定闭环**，优先级应高于原 R5）
   - R8 annotate 导入名（一行）+ N16 路径注释
   - R9 指纹加臂组合 + R6 训练覆盖前备份
3. **存储健壮性（🟡）**
   - M6 历史排序毒化 + N17 单次标记重写全部历史 + M5/M7 查找兜底对称化
   - N2 指纹类型归一化 + N3 指纹纳入检测权重
4. **UI 一致性（🟡→🟢）**
   - M3（含 N5 单视频集锦连带）、M4 标记优先级统一、M9 保存失败要提示、M12 移出事件循环
   - N13 注释订正（R3 根因）→ M8a/M8b、N7、N11、N12
5. **加固/清理（🟢）**
   - N8 清理豁免单球导出、N9/N10/N14/N15、L1/L2/L3、M10 转入加固项

---

## 附：核验未发现问题、可放行的部分

`merge_segments` 跨源/合并与 pre/post roll 钳制、`_atomic_write_json` 原子性与残留 tmp 命名、NVENC 信号量全部 with 配对、`put_clip_cache` 驱逐边界、tracker 静态球门/冷却早退顺序、`VideoReader`/`read_frame` 的 pts 换算与容器释放、`ui.timer` 随 client 删除自动 cancel、`recalib_ensemble.py` 只读不写生产、中文文件名 md5 安全化。
