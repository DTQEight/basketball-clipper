# Bug 审核终审对比（第三道验证，2026-09-22）

> 三份文档的关系：
> - [BUG_AUDIT_20260922.md](BUG_AUDIT_20260922.md) —— 第一轮全面审核清单（29 条，GLM-5.3 产出）
> - [BUG_AUDIT_20260922_REVIEW.md](BUG_AUDIT_20260922_REVIEW.md) —— 第二轮复盘（对该清单的逐条核验 + 新发现 N1–N17）
> - 本文 —— **第三道独立验证**：不信任前两轮的任何结论，对复盘中最关键/最大胆的判定重新取证（亲自读 NiceGUI 源码、跑 NVENC 实测），并给出三轮合并后的最终结论。
>
> 状态：仅审核，未改动任何代码。实测均为只读探测（lavfi 黑帧 → null muxer）。

---

## 一、第三道验证结果：复盘的核心判定全部站得住

| 复盘判定 | 验证方式 | 结果 |
|---|---|---|
| **R3 判为误报**（刷新不会取消协程） | 亲自读 NiceGUI 3.16 源码 | ✅ **确认**。`events.py:486`：事件 handler 的 awaitable 经 `background_tasks.create_or_defer(...)` 调度为**全局**任务，不属 client；`element.py:415-432` 注释原文："Silent when the *client* has been deleted (e.g. browser reload race past ``reconnect_timeout``) or already garbage-collected: **an async callback resuming after the teardown is not a user bug**"。刷新后 `clear_live` 照常执行，原清单"永显取消/整场重检"不成立 |
| **M10 判为误报**（静止球窗口分支不可达） | 亲自读构造点 | ✅ **确认**。`detection.py:764/:853` 两处 `GoalDetector(...)` 构造均传 `fps=fps`（`:660` 取自 `video_state`，全程同一标称值）→ 构造时已按真实 fps 初始化，中途改 fps 的分支应用内不可达 |
| **R1 触发条件扩大**（重检测后首次打标即清空） | 亲自 grep 调用点 | ✅ **确认**。`_restore_labels_to_clips`（`:2124` 定义）仅有两个调用点 `:2286/:2331`（批量/历史加载）；单视频 `run_detect` 完成后**无回填** → clips 不带人工标记 → 首次 √/× 即整体替换旧 kept/deleted |
| **N8 单球导出被 7 天清理误删** | 亲自读两处 | ✅ **确认**。`detection.py:1564-1567` 输出 `{源名}-goal-{ts:.1f}s.mp4` 落 `DEMO_OUTPUT_DIR`；`state.py:126` 只豁免 `-highlights.mp4` 后缀 → 单球成品一周后被当预览片段删除 |
| **R8 降级**（os.startfile 而非浏览器） | 亲自读 | ✅ **确认**。`annotate.py:376` `os.startfile(ev["clip_path"])` —— Windows 系统默认播放器，不经浏览器。"浏览器播不出"的后果描述确实写错 |
| **R5 降级**（只读打印不写阈值） | 亲自读全文 | ✅ **确认**。`eval_thresholds.py` 全文 63 行纯只读（读 clip_cache/history → print），无任何写操作；`:34` 确用 `state.label_sets`（循环评估机制本身成立） |
| **L8② 否定**（grays[0] 原触发条件不成立） | 亲自读 | ✅ **确认**。`extract_features.py:255-259`：`f1=min(total,...)` + `if f1-f0<8: continue`，"ts 超时长"场景两条件互斥，不可达 |
| **N1 提前 EOF 静默截断**（新增高危） | 前一轮已逐行确认三处联动 | ✅ **维持**。`video_io.py:294-301` 静默 return + `detection.py:1115` 只查 `processed==0` + `:1244` 成功即删全部断点，全仓无 `processed < n_frames` 完整性校验 |

## 二、⚠️ 一处需要修正：R4 的 NVENC 前提被实测推翻

**实测**（本机开发环境，`imageio_ffmpeg` v7.1 静态版 ffmpeg-win-x86_64-v7.1.exe）：

```
ffmpeg -f lavfi -i color=black:s=256x256:d=0.1 -frames:v 3 -c:v h264_nvenc -f null -
→ exit=0，无任何报错
```

**结论：本机开发环境 NVENC 可用。**

- 原始清单 R4 写"本机 GTX 1650 无 NVENC 单元、首段即整体回退"——**这个前提是错的**。"No capable devices found" 的记录来自**打包 exe 环境**（PyInstaller 单文件），不是开发环境。GTX 1650 存在无 NVENC 的 TU117 核心与有 NVENC 的核心两种批次，本机属后者。
- 对 R4 定级的影响：复盘把它 🔴→🟡 的理由之一是"触发概率低（本机不用 NVENC）"。实测推翻此理由——**平时就在走 NVENC（main profile）**，一旦导出中途某段 NVENC 运行时失败（2 路会话配额被预览线程池占满、驱动瞬断）回退 libx264（high profile），混编 concat 场景**真实可触发**。
- 维持 🟡 但应标"中偏高"：触发仍需"运行中失败"这一条件，且 `-c copy` 报错时有整体重编码兜底；静默成功时才有花屏风险。
- **顺带澄清一个环境事实**（值得写入项目记忆）：开发环境与打包 exe 环境的 NVENC 可用性**不同**。代码 `_detect_nvenc` 的运行时双探测恰好能正确处理这种差异（打包环境探测失败→全程软编；开发环境→走 NVENC），所以这不是 bug，但性能预期与排障时要区分两个环境。

## 三、复盘文档自身的两处小瑕疵

1. **计数不一致**：头部写"另有 **20 条新发现**"，实际列出 N1–N17 共 **17 条**（1 高危 + 5 中危 + 11 低危）。数字应为 17。
2. **N14 措辞过强**："现逻辑不可达"只对"ts 超时长"这一原触发场景成立；若视频解码**全失败**（`f0<f1` 但 `iter_frames` 产出 0 帧，即 `TestIterFramesDemuxTolerance` 覆盖的场景），`grays` 仍为空 → `grays[0]` IndexError 仍可达。N14 列为加固项的**方向正确**，但表述应改为"原触发场景不可达，解码失败场景仍可达"。

另：N2（slider int/float 指纹敏感）机制层面成立（JS `JSON.stringify(2.0)` = `"2"` → Python `int`，NiceGUI 事件经 JSON 传输），未起服务实测回传类型，维持"待确认"；但修法（指纹归一化 `float()`）无论实测结果如何都是正确的。

## 四、三轮合并后的最终结论

### 铁案（三轮一致或经实测/源码取证，可直接修）

| 级别 | 条目 | 一句话 |
|---|---|---|
| 🔴 | **R1** 人工标签被清空 | 触发条件以"重检测后首次打标"为主路径（比原清单的 clips⊂goals 更高频），修法必须覆盖两条路径 |
| 🔴 | **R2** 断点被误删 | `await dlg` 返回 None 落入删除分支；补 `is None` 分支即修 |
| 🔴 | **N1** 提前 EOF 静默截断 | `processed<n_frames` 无校验，成功分支还删断点；与 R2 同属断点生命周期，建议一起修 |
| 🔴 | **R7** 改标后重训用旧标签 | features.jsonl label 冻结在首次提取 |
| 🟡 | M1 M3(含N5) M4 M7 M9 M12 N2 N3 N4 N17 | 口径/存储类，行号与根因已核实 |
| 🟡 | R4（中偏高） R5 R6 R8 R9 | 方向属实、后果有界；R4 前提已实测更正 |
| 🟢 | L2 L4 L5 L6 L7 N7–N16 | 低危/加固 |

### 维持误报/降级结论
- **R3** 误报（降 🟢：仅"新页面不自动刷卡片"）→ 附带修 6 处写反的注释（`demo:939/1111/1439/1525`、`detection:1439/1655`、`CHANGELOG:1289`）
- **M10** 误报（转入加固项：补 tracker fps 重算测试）
- **M6 M11 M8a** 降级；**L8②** 否定（保留 N14 守卫加固）

### 修复优先级（终版）

1. **R1 + R2 + N1**（数据丢失级，断点/标签生命周期，同源一起修）
2. **R7 + N4/R5**（训练标签口径，标定闭环 `build_dataset.py:31` → `recalib_ensemble.py:63` 优先）
3. **R8 一行修复 + N2/N3 指纹归一化与纳入权重**
4. **M3(含N5)/M4 标记口径统一、M9 保存失败提示、M6 + N17 历史健壮性**
5. **R4 混编标记、M1/M2、其余 🟢 与注释订正**

## 五、本轮修复状态（2026-09-22 执行）

> 全部改动已跑 `pytest tests -q`（244 passed）。新增回归测试：`tests/test_training_labels.py`，
> 并在 `test_state.py` / `test_detection_guard.py` 追加 14 条。

### ✅ 已修

| 条目 | 修法要点 |
|---|---|
| **R1** 人工标签被清空 | `state.update_history_labels` 新增 `manual_scope`（覆盖范围）：范围内按新值写，**范围外的旧 √/× 原样保留**；`_persist_marks` 传入本次 clips 的 ts 集合。覆盖了"重检测后首次打标"与"clips 子集"两条路径 |
| **R2** 断点被误删 | `await dlg` 返回 `None`（外因关闭）单独分支 `return`，不再与"用户选从头开始"(False) 混同 |
| **N1** 提前 EOF 静默截断 | 新增 `_is_partial_decode()`；判定为 partial 时**保留断点**（可续跑补齐）、UI 与日志显式标注结果不完整 |
| **R7** 改标后重训用旧标签 | 新增 `_relabel_features()`：续跑时比对 dataset 当前标签，**就地改写** features.jsonl 的 label（特征不变，不重算昂贵的特征提取） |
| **R8** annotate 导入名错 | 改名 `build_encode_args`；并补上根因之一——顶部注入 `sys.path`（脚本以 `training/annotate.py` 运行时项目根不在导入路径上，即使名字对也 ImportError）；兜底参数补 `-pix_fmt yuv420p`；切片滤镜改用 `build_view_filter`（HDR→SDR 同口径） |
| **R5 / N4** 训练标定口径 | `eval_thresholds` / `eval_quantile_rule` / `eval_arm_ablation` 改用**严格人工标签**（kept/deleted）；`load_dataset_events` 保留 `label_source`；`refresh_oof` 输出 `label_strict` 并在 REPORT 里并列 strict 口径（标注"标定请看 strict_*"）；`recalib_ensemble --from-refresh` 自动切换到严格口径标定 |
| **R6** 训练覆盖生产模型 | `train_lgbm` 覆盖前把现役 `model_lgbm.txt` / `model_meta.json` 备份到 `training/backup/`（带时间戳） |
| **R9** 指纹不含臂组合 | 新增 `_available_arms()`，臂组合计入 `model_fingerprint` → 缺臂降级分数与全量分数**不再是同一指纹**，臂恢复后会重算 |
| **M1** B 臂兜底连累 Flow 臂 | ImageNet 兜底包 try（只禁本臂）；B 臂推理段补 `if b_nets:` + try（与 Flow 臂对称，且避免 `None` 崩溃） |
| **M2** 作废时未清 auto 标记 | `invalidate_stale` 同时清 `mark/mark_source`（auto 来源） |
| **M3 / N5** 阈值变更后 auto 标签不回写 | 历史加载路径改为**无条件** `_sync_marks`（原 `if any(auto)` 会跳过，留下旧阈值判定） |
| **M4** 整场 vs 单视频标记优先级相反 | `_clips_from_record` 改为与单视频一致：**人工 × > 人工 √ > 模型 √ > 模型 ×** |
| **M5 / M7** 历史记录查找不对称 | `_find_history_record` 的 basename 兜底改为**唯一匹配才生效**（防同名视频串记录）；`add_history` 复用它（迁移目录后仍能继承旧标签、不再新旧两条并存） |
| **M6** 排序毒化全部历史 | `_sort_key` 的 `float(timestamp)` 加 try（坏值记 0），与 time 字符串分支一致 |
| **M9** 人物分类异常被吞 | 捕获异常 + 读返回值，落盘失败时提示"⚠ 未能写入历史，重启后会丢失" |
| **M12** √/× 阻塞事件循环 | `clip_action` 走 `run.io_bound`；卡片刷新改 `ui.timer(0, once=True)` |
| **N2** 指纹对 int/float 敏感 | 指纹构建时数值统一归一化为 float（bool 单独判），杜绝 `2` vs `2.0` 导致"继续识别"静默从头跑 |
| **N3** 断点指纹不含检测权重 | 新增 `_ball_weights_fingerprint()`，`weights/*.pt` 的 name+mtime 计入断点指纹 |
| **N6** 无分时保留旧口径标记 | `refresh_auto` 对无分片段清掉 auto 标记 |
| **N8** 单球导出被 7 天清理误删 | `_purge_old_clips` 追加豁免 `-goal-*.mp4` |
| **N9** `os.listdir` 无 try | `scan_video_files` 包 try（网络盘断开/权限不足不再穿透到 UI 回调） |
| **N10** 集锦消息不反映实际段数 | `cut_clips` 新增尾部关键字参数 `stats`（默认 None，向后兼容）；有跳过时消息注明"实际写入 N 段、跳过 M 段失败" |
| **N11** 预览失效无提示 | 路径不存在时提示"片段已失效，请重新检测/加载" |
| **N12** 对话框元素累积 | `await` 后在 `finally` 里 `dlg.delete()` |
| **N13** A 臂 break 语义偏重 | 单批提特征失败改为 `continue`（只跳本批，不再放弃其后全部候选的 A 臂） |
| **N15 / N16 / N14** | 清死代码 `if False else`；注释路径订正；`grays` 空值守卫 |
| **M10（加固）** | 判定为误报（分支不可达），未改逻辑 |
| **M8a / M8b / N7** | `_CANCEL_REQ` 标记：已受理的取消不被 0.4s timer 复位；进度面板不再强制顶掉用户正在看的片段/集锦 |
| **M11** 标定 check-then-act | 选帧/标定/重置加 path 与帧号二次校验，不一致则放弃并提示 |
| **L1** YOLO 采样相位 | `processed % step` → 绝对帧号 `fidx % step`（从头跑逐字节等价，仅纠正续跑相位） |
| **L4 / L5** | 单球导出锁改模块级；AI 开关多标签同步（`set_value` 会触发 `on_change`，已加守卫防回环） |
| **R3 注释订正** | 订正 11 处写反的注释（`demo_nicegui.py` 5 处 + `services/detection.py` 5 处 + `doc/CHANGELOG.md` 2 处）：NiceGUI 事件协程是**全局** background task，刷新/断连**不会**取消它 |

### ⏸ 评估后决定不改（含理由）

| 条目 | 不改理由 |
|---|---|
| **L3** 指纹含 keep_thr 与未启用兜底模型 mtime | 指纹变化会使**全部缓存分数失效**、强制重跑四臂推理（每候选约 6.5s），而收益仅是省掉一次重算——代价远大于收益。`refresh_auto` 已能在不重算分数的情况下重推分带 |
| **L7** tracker 无身份关联 | 属跟踪算法的结构性局限，实际需同时通过 in_x、size_ok、circularity、blob_above_hoop、YOLO 硬否决、静止球剔除多重闸门才可能误检，概率低；改动风险远大于收益。本轮未改动代码 |
| **R4 的"混合编码器"** | 已修（回退时打 `mixed_encoder` 标记 → 拼接跳过 `-c copy` 直接整体重编码）。**注**：本机开发环境实测 NVENC 可用（`exit=0`），触发条件是"导出中途 NVENC 运行时失败" |
| **M12 的既往全量写盘** | 除 io_bound 外，`N17` 已把标记落盘从 O(N) 次原子写降为**只写被改的那一条** |

### 仍待决策（需先定方案，本轮未动）

- **M12 更深层的 UI 架构**（多标签共享状态的一致性）——属产品级改造
- 永久盲测池 / 权重 manifest / 运行趋势表等长期建议（原审核文档"长期建议"节）

---

## 六、方法论记录

三轮审核的分歧几乎全部集中在两类：
1. **依赖仓库注释做断言**（R3 的根源——注释与 NiceGUI 实际行为相反）→ 涉及第三方框架生命周期的结论必须从框架源码取证。
2. **"代码缺陷"≠"线上风险"**（M5/M6/M10/L8②：理论成立、当前数据不触发）→ 定级必须回答"谁、多久会遇到一次"。
3. **环境差异会被记忆混淆**（R4：打包环境的 NVENC 失败记录被套到开发环境）→ 涉及硬件/环境的事实，实测优先于记忆。
