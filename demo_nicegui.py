"""篮球进球检测与自动剪辑交互式 Demo（NiceGUI）。

功能：
  1. 输入视频文件路径 → 加载
  2. 滑动到含篮筐的帧，点击画面 2 个点标定篮筐
  3. 设置起止帧、置信度、最小进球间隔
  4. 点击「开始检测」→ diff + YOLO 双确认检测进球
  5. 每个进球生成独立预览片段，人工确认保留/删除
  6. 生成集锦视频（GPU 硬编加速）
  7. 历史记录支持加载后直接剪辑（无需重新检测）

架构：
  - services/state.py     全局状态 + 历史记录 + 片段缓存
  - services/video_utils.py 视频工具函数
  - services/detection.py  检测/剪辑/批量业务逻辑
  - demo_nicegui.py        UI 层（本文件）

用法:
    python demo_nicegui.py
浏览器打开 http://127.0.0.1:7871
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
from nicegui import ui

from services import state, detection, video_utils

# 启动时从磁盘恢复片段缓存
state.init_clip_cache()


# ============ NiceGUI 界面 ============

@ui.page('/')
def main_page():
    ui.dark_mode().enable()
    ui.add_head_html('''
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
    /* ============ Dark Tech Theme (v2 design-taste-frontend) ============ */
    :root {
      /* 背景层级（off-black, 非 pure black） */
      --bg-canvas:   #0A0A0A;   /* 最底层 */
      --bg-surface:  #141414;   /* 卡片层 */
      --bg-elevated: #1F1F1F;   /* 悬浮层 */
      --border-subtle: #2A2A2A; /* 低对比分隔 */
      --border-strong: #3A3A3A; /* 强分隔 */
      /* 文字层级 */
      --text-primary:   #FAFAFA; /* off-white */
      --text-secondary:  #A3A3A3; /* zinc-400 */
      --text-tertiary:   #525252; /* zinc-600 */
      /* 单一 accent: Electric Cyan (v2 推荐 Electric Blue 类) */
      --accent:          #22D3EE;
      --accent-hover:    #67E8F9;
      --accent-muted:    rgba(34, 211, 238, 0.12);
      /* 语义色（仅状态用，不破坏单 accent 锁定） */
      --ok:    #22C55E;
      --err:   #EF4444;
      --busy:  #22D3EE;
      /* 字体栈 */
      --font-sans: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'Geist Mono', 'SF Mono', Menlo, monospace;
    }
    body {
      background: var(--bg-canvas) !important;
      margin: 0;
      overflow: hidden;
      font-family: var(--font-sans);
      -webkit-font-smoothing: antialiased;
      font-feature-settings: "tnum" 1, "ss01" 1;
    }
    .nicegui-content { max-width: 100% !important; padding: 0 !important; font-family: var(--font-sans); }

    /* 数字 tabular-nums 对齐 */
    .font-mono, [class*="font-mono"] { font-family: var(--font-mono) !important; font-variant-numeric: tabular-nums; }

    /* ============ 滚动条（accent 主题化） ============ */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(34, 211, 238, 0.25); border-radius: 3px; transition: background 0.2s; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(34, 211, 238, 0.5); }

    /* ============ Quasar 组件覆盖 ============ */
    .q-expansion-item__header { white-space: nowrap; }
    /* 输入框聚焦 accent 边框 + 微发光 */
    .q-field--outlined.q-field--focused .q-field__control {
      border-color: var(--accent) !important;
      box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.15);
    }
    .q-field--outlined.q-field--focused .q-field__label { color: var(--accent) !important; }

    /* ============ 按钮交互态 ============ */
    .q-btn { transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1); }
    .q-btn:active { transform: scale(0.98); }
    /* 透明 accent 描边按钮 hover */
    .btn-accent-outline:hover {
      background: var(--accent-muted);
      border-color: var(--accent);
      color: var(--accent);
    }

    /* ============ 进球卡片：左侧色条 + hover 上浮 ============ */
    .result-card {
      position: relative;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      border-left: 3px solid transparent !important;
    }
    .result-card:hover {
      background: var(--bg-elevated) !important;
      border-left-color: var(--accent) !important;
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .result-card:active { transform: translateY(0); }

    /* ============ 视频区边框 + 屏幕感 ============ */
    .q-video, video {
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      box-shadow: inset 0 0 0 1px rgba(0,0,0,0.5), 0 0 20px rgba(34, 211, 238, 0.04);
    }

    /* ============ 进度条 accent 渐变 ============ */
    .q-linear-progress__track { background: rgba(34, 211, 238, 0.12) !important; }
    .q-linear-progress__model {
      background: linear-gradient(90deg, #0E7490 0%, #22D3EE 50%, #67E8F9 100%) !important;
      box-shadow: 0 0 8px rgba(34, 211, 238, 0.4);
    }

    /* ============ 历史记录选中态 ============ */
    .history-row {
      transition: all 0.15s ease;
    }
    .history-row:hover { background: var(--bg-elevated); }
    .history-row.selected {
      border-color: var(--accent) !important;
      background: var(--accent-muted);
    }

    /* ============ 背景噪点纹理（去 digital flatness） ============ */
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Cfilter id='n'%3E%3CfeTurbulence baseFrequency='0.9'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
      pointer-events: none;
      z-index: 0;
    }
    /* 确保内容层在噪点层之上 */
    .nicegui-content, .q-page, .q-layout { position: relative; z-index: 1; }

    /* 减少动效偏好 */
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
      }
    }
    </style>
    ''')

    with ui.column().classes('w-full h-[100dvh] p-0 gap-0').style('overflow: hidden; background: var(--bg-canvas)'):
        # ====== 主容器：左右分栏 ======
        with ui.row().classes('w-full h-full'):

            # ========== 左侧面板 ==========
            with ui.column().classes('w-[350px] min-w-[350px] border-r').style('height: 100%; overflow: hidden; background: var(--bg-surface); border-color: var(--border-subtle)'):

                # 标题行 + 折叠功能区按钮（始终可见）
                with ui.row().classes('w-full items-center justify-between px-3 py-2 border-b').style('flex-shrink: 0; border-color: var(--border-subtle)'):
                    ui.label('🏀 进球集锦助手').classes('text-sm font-bold').style('color: var(--text-primary)')
                    collapse_btn = ui.button('▲ 收起功能区', on_click=lambda: _toggle_func_collapse()).classes(
                        'text-xs').style('background: var(--bg-elevated); color: var(--text-secondary)')

                # 功能区域（固定高度，紧凑）
                with ui.column().classes('w-full px-3 py-2 gap-1 border-b').style('flex-shrink: 0; border-color: var(--border-subtle)') as func_container:

                    # 输入框 + 加载按钮（一行）
                    path_input = ui.input(value=state.DEFAULT_VIDEO, placeholder='文件路径').classes('w-full').props('dense')
                    info_text = ui.label('').classes('text-gray-400 text-xs font-mono')
                    calib_status = ui.label('').classes('text-gray-400 text-xs font-mono')
                    with ui.row().classes('w-full gap-2'):
                        ui.button('加载', on_click=lambda: _on_load()).classes('flex-1 text-sm').props('ripple').style('background: var(--bg-elevated); color: var(--text-secondary)')
                        ui.button('重置', on_click=lambda: _on_reset()).classes('text-sm').props('ripple').style('background: var(--bg-elevated); color: var(--text-secondary)')

                    # 开始识别
                    detect_btn = ui.button('开始识别', on_click=lambda: _on_detect()).classes(
                        'w-full font-bold text-sm').props('ripple').style('background: var(--accent); color: var(--bg-canvas)')

                    # 文件夹批量模式面板（加载文件夹后显示）
                    batch_panel = ui.column().classes('w-full gap-1 hidden')
                    with batch_panel:
                        batch_select = ui.select(options={}, value=None).classes('w-full').props('outlined dense dark')
                        with ui.row().classes('w-full gap-2'):
                            ui.button('保存标定', on_click=lambda: _on_batch_save_calib()).classes('flex-1 text-xs').props('ripple').style('background: var(--bg-elevated); color: var(--text-secondary)')
                            batch_run_btn = ui.button('批量识别', on_click=lambda: _on_batch_run()).classes('flex-1 text-xs').props('ripple').style('background: var(--accent); color: var(--bg-canvas)')

                    # 结果状态
                    result_status = ui.label('').classes('text-gray-400 text-xs')

                    # 折叠区域：参数 / 集锦 / 历史 合并到一个框（展开时占满整框）
                    with ui.row().classes('w-full border rounded-lg overflow-hidden').style('gap: 0; flex-wrap: wrap; border-color: var(--border-subtle)'):
                        exp_params = ui.expansion('参数', group='leftpanel').classes('w-1/3 text-gray-300 text-xs').style('min-width: 0').props('duration=0')
                        with exp_params:
                            with ui.column().classes('gap-1 w-full p-1 max-h-[300px] overflow-y-auto'):
                                yolo_3frame_switch = ui.switch('提速模式 (YOLO每3帧推理一次, 可能略漏检)', value=False).classes('w-full')
                                ui.label('默认每2帧（推荐, 更准）').classes('text-gray-500 text-[10px] -mt-1 mb-1')
                                skip_yolo_switch = ui.switch('条件跳过 (篮筐无运动时跳过YOLO, 大幅提速)', value=False).classes('w-full')
                                ui.label('篮筐区域无运动像素时跳过 YOLO 推理').classes('text-gray-500 text-[10px] -mt-1 mb-1')
                                with ui.row().classes('gap-2 w-full'):
                                    start_frame = ui.number(label='起始帧', value=0, format='%d').classes('flex-1')
                                    end_frame = ui.number(label='结束帧(0=末尾)', value=0, format='%d').classes('flex-1')
                                with ui.row().classes('gap-2 w-full'):
                                    ball_conf = ui.slider(min=0.1, max=0.9, value=0.3, step=0.05).classes('flex-1')
                                    ui.label().bind_text_from(ball_conf, 'value', lambda v: f'置信度: {v:.2f}').classes('text-gray-400 text-xs')
                                with ui.row().classes('gap-2 w-full'):
                                    min_gap = ui.slider(min=1.0, max=10.0, value=3.0, step=0.5).classes('flex-1')
                                    ui.label().bind_text_from(min_gap, 'value', lambda v: f'进球间隔: {v:.1f}s').classes('text-gray-400 text-xs')
                                with ui.expansion('高级', icon='tune').classes('w-full text-gray-400'):
                                    auto_threshold_switch = ui.switch('自适应阈值', value=True).classes('text-xs')
                                    ui.label('前30秒预热自动计算阈值').classes('text-gray-500 text-[10px] -mt-1')
                                    diff_threshold = ui.slider(min=5, max=40, value=15, step=5).classes('w-full')
                                    diff_threshold_label = ui.label().classes('text-gray-400 text-xs')
                                    min_circularity = ui.slider(min=0.0, max=0.8, value=0.35, step=0.05).classes('w-full')
                                    ui.label().bind_text_from(min_circularity, 'value', lambda v: f'圆形度: {v:.2f}').classes('text-gray-400 text-xs')
                                    min_in_hoop_frames = ui.slider(min=1, max=6, value=2, step=1).classes('w-full')
                                    ui.label().bind_text_from(min_in_hoop_frames, 'value', lambda v: f'进框帧数: {v}').classes('text-gray-400 text-xs')
                                    min_blob_area = ui.slider(min=10, max=200, value=30, step=10).classes('w-full')
                                    ui.label().bind_text_from(min_blob_area, 'value', lambda v: f'最小斑块: {v}').classes('text-gray-400 text-xs')
                                    search_margin = ui.slider(min=20, max=150, value=80, step=10).classes('w-full')
                                    ui.label().bind_text_from(search_margin, 'value', lambda v: f'搜索范围: {v}px').classes('text-gray-400 text-xs')

                                    def _sync_diff_threshold(auto_on):
                                        # 直接用 .style 切颜色，比 classes(add/remove) 更稳，且不存在切换顺序导致颜色残留
                                        _cyan = '#22D3EE'
                                        _gray = 'var(--text-secondary)'
                                        if auto_on:
                                            diff_threshold.props('disable')
                                            diff_threshold_label.set_text('帧差阈值: 自动（预热后计算）')
                                            diff_threshold.style(f'color: {_cyan}')
                                        else:
                                            diff_threshold.props(remove='disable')
                                            diff_threshold_label.set_text(f'帧差阈值: {diff_threshold.value}')
                                            diff_threshold.style(f'color: {_gray}')
                                    auto_threshold_switch.on_value_change(lambda e: _sync_diff_threshold(e.value))
                                    def _on_diff_threshold_change(e):
                                        if not auto_threshold_switch.value:
                                            diff_threshold_label.set_text(f'帧差阈值: {e.value}')
                                    diff_threshold.on_value_change(_on_diff_threshold_change)
                                    _sync_diff_threshold(True)
                        exp_hl = ui.expansion('集锦', group='leftpanel').classes('w-1/3 text-gray-300 text-xs').style('min-width: 0').props('duration=0')
                        with exp_hl:
                            with ui.column().classes('gap-1 w-full p-1'):
                                hl_pre_roll = ui.slider(min=0, max=10, value=5, step=1).classes('w-full')
                                ui.label().bind_text_from(hl_pre_roll, 'value', lambda v: f'提前: {v:.0f}s').classes('text-gray-400 text-xs')
                                hl_post_roll = ui.slider(min=0, max=10, value=5, step=1).classes('w-full')
                                ui.label().bind_text_from(hl_post_roll, 'value', lambda v: f'延后: {v:.0f}s').classes('text-gray-400 text-xs')
                                hl_min_gap = ui.slider(min=1, max=30, value=8, step=1).classes('w-full')
                                ui.label().bind_text_from(hl_min_gap, 'value', lambda v: f'合并间隔: {v:.0f}s').classes('text-gray-400 text-xs')
                        exp_hist = ui.expansion('历史', group='leftpanel').classes('w-1/3 text-gray-300 text-xs').style('min-width: 0').props('duration=0')
                        with exp_hist:
                               history_list = ui.column().classes('w-full gap-1 max-h-[120px] overflow-y-auto')

                    # 展开的折叠区占满整框，其余两个隐藏；折叠后恢复并排
                    _exp_list = [exp_params, exp_hl, exp_hist]
                    def _sync_expand():
                        # 从实际状态重新计算，避免触发顺序导致不一致
                        active = None
                        for _e in _exp_list:
                            if _e.value:
                                active = _e
                                break
                        if active is None:
                            for _e in _exp_list:
                                _e.classes(add='w-1/3', remove='hidden w-full')
                        else:
                            for _e in _exp_list:
                                if _e is active:
                                    _e.classes(add='w-full', remove='w-1/3 hidden')
                                else:
                                    _e.classes(add='hidden', remove='w-1/3 w-full')
                    for _e in _exp_list:
                        _e.on_value_change(lambda evt: _sync_expand())
                    # 展开「历史」时自动加载记录，无需手动点刷新
                    exp_hist.on_value_change(lambda e: _refresh_history() if e.value else None)
                    _sync_expand()

                # 导出集锦按钮（固定在列表上方，仅列表有内容时显示）
                with ui.row().classes('w-full px-3 pt-2 flex-shrink-0 hidden') as export_row:
                    ui.button('导出集锦', on_click=lambda: _on_highlights()).classes(
                        'w-full text-sm font-bold').props('ripple').style('background: var(--accent); color: var(--bg-canvas)')

                # 进球列表区域（独立滚动）
                with ui.column().classes('w-full p-2 gap-1 overflow-y-auto flex-1').style('min-height: 0'):
                    # 结果列表容器
                    result_container = ui.column().classes('w-full gap-1')

                # ====== 顶部功能区折叠控制 ======
                _func_state = {"collapsed": False}

                def _set_func_collapsed(collapsed: bool):
                    """折叠/展开顶部功能区，把空间让给进球列表。"""
                    _func_state["collapsed"] = collapsed
                    if collapsed:
                        func_container.classes(add='hidden')
                        collapse_btn.set_text('▾ 展开功能区')
                        collapse_btn.style('background: transparent; color: var(--accent); border: 1px solid var(--accent)')
                    else:
                        func_container.classes(remove='hidden')
                        collapse_btn.set_text('▲ 收起功能区')
                        collapse_btn.style('background: var(--bg-elevated); color: var(--text-secondary); border: none')

                def _toggle_func_collapse():
                    _set_func_collapsed(not _func_state["collapsed"])

            # ========== 右侧面板 ==========
            with ui.column().classes('flex-1 p-4 gap-3').style('background: var(--bg-canvas)'):

                # 视频预览区
                preview_image = ui.interactive_image(
                    source='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
                ).classes('w-full rounded-xl bg-black').style('aspect-ratio: 16/9; object-fit: contain')

                result_video_el = ui.video(src='').classes('w-full rounded-xl hidden').style('aspect-ratio: 16/9')
                highlights_video_el = ui.video(src='').classes('w-full rounded-xl hidden').style('aspect-ratio: 16/9')

                # 进度显示区（检测时显示，居中）
                progress_container = ui.column().classes('w-full hidden items-center justify-center').style('aspect-ratio: 16/9')
                with progress_container:
                    progress_text = ui.label('检测中...').classes('text-lg font-semibold').style('color: var(--accent)')
                    progress_bar = ui.linear_progress(show_value=False).classes('w-64 mt-3')
                    progress_detail = ui.label('').classes('text-gray-400 text-xs mt-2')

                # 底部留白
                ui.label('').classes('h-4')

    # ====== 事件处理函数 ======
    def _set_status(text, kind='info'):
        """设置结果状态文本并切换颜色（ok=绿 / err=红 / busy=青 / info=灰）。"""
        result_status.set_text(text)
        _color_map = {
            'ok': 'var(--ok)',
            'err': 'var(--err)',
            'busy': 'var(--busy)',
        }
        result_status.style(f'color: {_color_map.get(kind, "var(--text-secondary)")}')

    # ========== 公共辅助：切换右侧视频区可见性（避免每处重复 4 行 classes 切换，也避免遗漏） ==========
    def _show_right_pane(mode):
        """右侧 4 个元素互斥显示：'preview' | 'result' | 'highlights' | 'progress' | None（preview fallback）。"""
        modes = {'preview', 'result', 'highlights', 'progress'}
        if mode not in modes:
            mode = 'preview'
        # 全部先隐藏（一次性 remove/add，比每处写 4 行稳）
        for el, name in [(preview_image, 'preview'),
                         (result_video_el, 'result'),
                         (highlights_video_el, 'highlights'),
                         (progress_container, 'progress')]:
            if name == mode:
                el.classes(remove='hidden')
            else:
                el.classes(add='hidden')

    # ====== 全局任务互斥（B1）：检测/批量/集锦/加载运行期间，禁止其他改 state 的操作 ======
    _busy = {"task": None}

    def _task_label(task):
        return {'detect': '开始识别', 'batch': '批量识别',
                'highlights': '生成集锦', 'load': '加载视频'}.get(task, '后台任务')

    def _try_acquire(task):
        """尝试占用任务锁；已有任务运行则提示并返回 False。"""
        if _busy["task"] is not None:
            _set_status(f'「{_task_label(_busy["task"])}」正在进行，请等待完成或取消', 'err')
            return False
        _busy["task"] = task
        return True

    def _refuse_if_busy():
        """轻量操作守卫：有任务运行时提示并返回 True（调用方直接 return）。"""
        if _busy["task"] is not None:
            _set_status(f'「{_task_label(_busy["task"])}」正在进行，请稍后再试', 'err')
            return True
        return False

    async def _on_load():
        if _refuse_if_busy():
            return
        path = path_input.value
        # 文件夹路径 → 批量标定 + 批量识别模式
        if path and os.path.isdir(path.strip().strip('"')):
            files = video_utils.scan_video_files(path)
            if not files:
                _set_status('文件夹内没有找到视频文件', 'err')
                return
            # 切换到批量模式：立即清空上一个视频的 state + 同步 UI 空列表
            # （否则 _on_batch_load_video 异步生成预览期间，UI 会一直残留上一个视频的进球卡片）
            state.last_goal_clips.clear()
            state.last_goals.clear()
            state.kept_goal_indices.clear()
            _refresh_result_cards()
            state.batch_files = files
            state.batch_calibs = {}
            state.batch_current_video = None
            batch_panel.classes(remove='hidden')
            _refresh_batch_list()
            # 自动加载第一个视频（同时同步下拉框）
            await _on_batch_load_video(files[0])
            _set_status(f'批量模式 | 扫描到 {len(files)} 个视频，逐个标定后批量识别', 'info')
            return
        # 单视频文件路径 → 原有流程（清空批量状态，避免写历史误带 batch_idx）
        batch_panel.classes(add='hidden')
        state.batch_files = []
        state.batch_calibs = {}
        state.batch_current_video = None
        # 重要：state.last_goal_clips/last_goals/kept_goal_indices 的清空
        #       已下沉到 detection.load_video 业务层（切视频即清空），UI 层不再重复
        #       以避免 frame is None 分支漏清空导致旧数据残留
        frame, info = detection.load_video(path)
        if frame is not None:
            # 切换视频：仅清 UI 卡片容器缓存（state 已由 load_video 清空）
            _refresh_result_cards()
            b64 = video_utils.frame_to_base64(frame)
            preview_image.set_source(b64)
            _show_right_pane('preview')
        else:
            # 加载失败也必须刷新卡片（state 已在 load_video 里清空，UI 要同步显示空列表）
            _refresh_result_cards()
        info_text.set_text(info)
        calib_status.set_text('请点击画面 2 个点标定篮筐' if frame is not None else info)

    def _refresh_batch_list():
        """刷新批量下拉框：文件名前带标定状态标记（✓已标定 / ○未标定），保留当前选中。"""
        if not state.batch_files:
            return
        cur_val = batch_select.value if batch_select.value in state.batch_files else None
        # 用 dict（值->标签）作为 options：dict 时 select 的值才是纯路径字符串；
        # 若用 [(值,标签)] 列表，NiceGUI 会把整个元组当值，导致加载失败
        batch_select.set_options(
            {f: (f'✓ {os.path.basename(f)}' if f in state.batch_calibs
                 else f'○ {os.path.basename(f)}') for f in state.batch_files},
            value=cur_val)

    _batch_loading = False  # 防重入（set_value 可能触发 change 事件）

    async def _on_batch_load_video(path=None):
        """加载批量视频（从下拉或列表点击）。

        若该视频已有历史检测记录（批量识别完成后再点击），
        自动加载检测结果和预览片段，无需再去历史记录里找。
        """
        nonlocal _batch_loading
        # 兼容旧版（值,标签）元组，防御性解包
        if isinstance(path, (tuple, list)):
            path = path[0]
        if _batch_loading:
            return
        if _refuse_if_busy():
            return
        if path is None:
            path = batch_select.value
        if not path:
            # 下拉框被重置（如重新扫描）时静默跳过，避免误报
            if state.batch_current_video is None:
                _set_status('请先选择视频', 'err')
            return
        if path == state.batch_current_video:
            # 已是当前视频：仅同步下拉框显示，避免重复加载
            if batch_select.value != path:
                batch_select.set_value(path)
            return
        _batch_loading = True
        if not _try_acquire('load'):
            _batch_loading = False
            return
        try:
            batch_select.set_value(path)
            # 若该视频已有检测结果，生成预览片段可能耗时，用 io_bound 避免阻塞 UI
            from nicegui import run

            def _progress_callback(pct, msg):
                try:
                    progress_text.set_text(msg)
                except Exception:
                    pass

            try:
                frame, info, status = await run.io_bound(
                    detection.on_batch_load_video, path, _progress_callback)
            except Exception as _e:
                import traceback
                frame, info, status = None, "", f"❌ 加载视频异常: {_e}\n{traceback.format_exc()}"
            if frame is not None:
                b64 = video_utils.frame_to_base64(frame)
                preview_image.set_source(b64)
                _show_right_pane('preview')
            info_text.set_text(info)
            calib_status.set_text(status)
            # 刷新结果卡片（已检测过的视频会显示进球列表）
            _refresh_result_cards()
        finally:
            _busy["task"] = None
            _batch_loading = False

    def _on_batch_save_calib():
        if _refuse_if_busy():
            return
        status = detection.on_batch_save_calib()
        calib_status.set_text(status)
        _refresh_batch_list()
        _set_status(status, 'ok' if '已保存' in status else 'err')

    async def _on_batch_run():
        if _busy["task"] == 'batch':
            # 批量中点击 → 请求取消，run_batch_detect 轮询后中断
            state.cancel_requested = True
            batch_run_btn.set_text('正在取消...')
            batch_run_btn.disable()
            return
        if not state.batch_files:
            _set_status('请先加载文件夹', 'err')
            return
        if not _try_acquire('batch'):
            return
        state.cancel_requested = False
        batch_run_btn.set_text('取消')
        batch_run_btn.enable()
        # 显示进度
        _show_right_pane('progress')
        progress_bar.set_value(0)
        progress_text.set_text('正在批量识别...')
        progress_detail.set_text('')

        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                progress_text.set_text('批量识别中...')
                progress_detail.set_text(msg)
            except Exception:
                pass

        from nicegui import run
        try:
            status, ok = await run.io_bound(
                detection.run_batch_detect,
                int(start_frame.value or 0), int(end_frame.value or 0),
                ball_conf.value, min_gap.value,
                diff_threshold.value, min_circularity.value, int(min_in_hoop_frames.value),
                min_blob_area.value, search_margin.value,
                progress_callback=_progress_callback,
                auto_threshold=auto_threshold_switch.value,
                yolo_step=3 if yolo_3frame_switch.value else 2,
                skip_yolo_no_motion=skip_yolo_switch.value)
        except Exception as _e:
            import traceback
            status = f"❌ 批量识别异常: {_e}\n{traceback.format_exc()}"
            ok = False
        finally:
            _busy["task"] = None
        batch_run_btn.set_text('批量识别')
        batch_run_btn.enable()
        _show_right_pane('preview')
        _set_status(status, 'ok' if ok else 'err')
        # 批量结束后显示最后一个视频的结果
        _refresh_result_cards()
        _refresh_batch_list()

    def _on_image_click(e):
        """点击预览图标定篮筐。

        使用 ui.interactive_image 的 on_mouse 事件，e.image_x/e.image_y
        已由前端按 显示尺寸/原始尺寸 比例换算为原始帧坐标。
        """
        if _busy["task"] is not None:
            calib_status.set_text('任务进行中，暂不能标定')
            return
        try:
            x = int(round(e.image_x))
            y = int(round(e.image_y))
            nat_w = int(state.video_state.get('width', 0) or 0)
            nat_h = int(state.video_state.get('height', 0) or 0)
            if nat_w > 0:
                x = max(0, min(x, nat_w - 1))
            if nat_h > 0:
                y = max(0, min(y, nat_h - 1))
            frame, status = detection.click_calibrate(x, y)
            if frame is not None:
                b64 = video_utils.frame_to_base64(frame)
                preview_image.set_source(b64)
            calib_status.set_text(status)
        except Exception as ex:
            calib_status.set_text(f'点击解析失败: {ex}')

    # 注册鼠标事件（interactive_image 默认监听 click，on_mouse 回调直接返回换算后的坐标）
    preview_image.on_mouse(_on_image_click)

    # 下拉框选择视频后立即加载（无需再点「加载」按钮），直接用事件携带的新值
    async def _on_batch_select_change(e):
        await _on_batch_load_video(e.value)
    batch_select.on_value_change(_on_batch_select_change)

    def _on_reset():
        if _refuse_if_busy():
            return
        status = detection.reset_hoop()
        calib_status.set_text(status)
        # 刷新预览
        if state.video_state["path"]:
            frame, _ = detection.preview_frame(state.video_state["current_frame"])
            if frame is not None:
                preview_image.set_source(video_utils.frame_to_base64(frame))

    async def _on_detect():
        if _busy["task"] == 'detect':
            # 检测中点击 → 请求取消，run_detect 轮询后中断
            state.cancel_requested = True
            detect_btn.set_text('正在取消...')
            detect_btn.disable()
            return
        if not _try_acquire('detect'):
            return
        state.cancel_requested = False
        detect_btn.set_text('取消')
        detect_btn.enable()
        # 显示进度条
        _show_right_pane('progress')
        progress_bar.set_value(0)
        progress_text.set_text('正在加载模型...')
        progress_detail.set_text('')

        # 进度回调函数（修复 Bug#3：更新 progress_detail，让用户看到当前处理的帧/阶段）
        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                progress_text.set_text(msg)
                progress_detail.set_text(msg)
            except Exception:
                pass

        # 使用 run.io_bound 在后台线程执行，避免阻塞事件循环
        from nicegui import run
        try:
            status, ok = await run.io_bound(
                detection.run_detect,
                int(start_frame.value or 0), int(end_frame.value or 0),
                ball_conf.value, min_gap.value,
                diff_threshold.value, min_circularity.value, int(min_in_hoop_frames.value),
                min_blob_area.value, search_margin.value,
                progress_callback=_progress_callback,
                auto_threshold=auto_threshold_switch.value,
                yolo_step=3 if yolo_3frame_switch.value else 2,
                skip_yolo_no_motion=skip_yolo_switch.value)
        except Exception as _e:
            import traceback
            status = f"❌ 检测异常: {_e}\n{traceback.format_exc()}"
            ok = False
        finally:
            _busy["task"] = None
        # 隐藏进度条，显示预览图（无论成功/失败/取消，都回到一致的 preview 态，避免视频重叠）
        _show_right_pane('preview')
        if state.cancel_requested:
            _set_status(status, 'info')  # 用户取消属中性提示，不用红色
            _refresh_result_cards()      # 同步清空列表
        else:
            _set_status(status, 'ok' if ok else 'err')
        detect_btn.set_text('开始识别')
        detect_btn.enable()
        if ok:
            _refresh_result_cards()

    def _refresh_result_cards():
        """刷新结果卡片列表。"""
        result_container.clear()
        if not state.last_goal_clips:
            with result_container:
                ui.label('暂无进球结果').classes('text-gray-300 text-xs text-center w-full py-4')
                ui.label('请先加载视频 → 标定篮筐 → 开始识别').classes('text-gray-400 text-xs text-center w-full')
            export_row.classes(add='hidden')  # 无结果时隐藏导出按钮
            _set_func_collapsed(False)  # 列表为空时展开功能区
            return
        export_row.classes(remove='hidden')  # 有结果时显示导出按钮
        _set_func_collapsed(True)  # 进球列表出来后自动折叠顶部功能区，把空间让给列表
        for i, clip in enumerate(state.last_goal_clips):
            ts = clip["ts"]
            # 预览片段实际为进球时刻 ±3s（见 _generate_preview_clips 的 clip_half）
            start_ts = max(0.0, ts - 3)
            end_ts = ts + 3
            t_min, t_sec = int(start_ts // 60), start_ts % 60
            end_min, end_sec = int(end_ts // 60), end_ts % 60
            with result_container:
                with ui.card().props('flat').classes('result-card w-full rounded-lg px-3 py-2').style('margin: 0; background: var(--bg-surface); border: 1px solid var(--border-subtle)'):
                    # 第一行：时间戳徽章（mono + tabular-nums）
                    with ui.row().classes('w-full items-center gap-2 mb-1'):
                        ui.label(f'{t_min}:{t_sec:04.1f} - {end_min}:{end_sec:04.1f}').classes(
                            'text-sm font-bold font-mono').style('color: var(--accent)')
                    # 第二行：操作按钮（图标化，hover 才显文字）
                    with ui.row().classes('w-full gap-1'):
                        ui.button('预览', on_click=lambda e, idx=i: _on_preview_clip(idx)).classes(
                            'flex-1 text-xs rounded-lg py-1').props('ripple flat').style('color: var(--text-secondary); border: 1px solid var(--border-subtle)')
                        ui.button('导出', on_click=lambda e, idx=i: _on_export_clip(idx)).classes(
                            'flex-1 text-xs rounded-lg py-1').props('ripple flat').style('color: var(--text-secondary); border: 1px solid var(--border-subtle)')
                        ui.button('删除', on_click=lambda e, idx=i: _on_delete_clip(idx)).classes(
                            'flex-1 text-xs rounded-lg py-1').props('ripple flat').style('color: var(--err); border: 1px solid rgba(239, 68, 68, 0.3)')

    def _on_preview_clip(idx):
        path, status = detection.clip_action("preview", idx)
        if path and os.path.exists(path):
            result_video_el.set_source(path)
            _show_right_pane('result')
            # 自动播放
            result_video_el.run_method('play')
        _set_status(status, 'info')

    def _on_export_clip(idx):
        path, status = detection.clip_action("export", idx)
        if path and os.path.exists(path):
            ui.download(path)
        _set_status(status, 'ok' if path and os.path.exists(path) else 'err')

    def _on_delete_clip(idx):
        if _refuse_if_busy():
            return
        # 点击直接删除，不弹确认框
        _, status = detection.clip_action("delete", idx)
        _refresh_result_cards()
        _set_status(status, 'info')

    async def _on_highlights():
        if not _try_acquire('highlights'):
            return
        from nicegui import run
        # 在右侧预览区显示进度
        _show_right_pane('progress')
        progress_bar.set_value(0)
        progress_text.set_text('正在生成集锦...')
        progress_detail.set_text('')

        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                progress_text.set_text(msg)
                progress_detail.set_text(msg)
            except Exception:
                pass

        _set_status('正在生成集锦...', 'busy')
        try:
            path, status = await run.io_bound(
                detection.generate_highlights, hl_pre_roll.value, hl_post_roll.value,
                hl_min_gap.value, _progress_callback)
        except Exception as _e:
            import traceback
            path, status = None, f"❌ 集锦生成异常: {_e}\n{traceback.format_exc()}"
        finally:
            _busy["task"] = None

        if path and os.path.exists(path):
            ui.download(path)
            highlights_video_el.set_source(path)
            _show_right_pane('highlights')
        else:
            _show_right_pane('preview')
        _set_status(status, 'ok' if path and os.path.exists(path) else 'err')

    # 历史记录选中索引（用 dict 包一层，让闭包内外都能读写；普通变量需要 nonlocal 但跨多个函数不便）
    _selected_history_idx = {"idx": None}

    def _refresh_history():
        history_list.clear()
        records = state.load_history()
        if not records:
            with history_list:
                ui.label('暂无历史记录').classes('text-gray-400 text-xs')
            return
        for i, r in enumerate(records):
            video = r.get("video", "")
            name = os.path.basename(video) if video else "未知"
            goals = len(r.get("goals", []))
            selected = (_selected_history_idx["idx"] == i)
            row_cls = 'history-row selected' if selected else 'history-row'
            with history_list:
                def _make_click(idx):
                    async def _on_click():
                        _selected_history_idx["idx"] = idx
                        _refresh_history()
                        await _on_load_history()
                        exp_hist.set_value(False)  # 加载完成后自动收起历史面板
                    return _on_click
                with ui.row().classes(f'w-full items-center gap-1 p-1 rounded cursor-pointer border {row_cls}').on('click', _make_click(i)):
                    ui.label(f'{i+1}.').classes('text-xs font-bold font-mono').style('color: var(--accent)')
                    ui.label(name).classes('text-xs flex-1 truncate').style('color: var(--text-primary)')
                    ui.label(f'{goals}球').classes('text-xs font-mono').style('color: var(--text-secondary)')

    async def _on_load_history():
        if _refuse_if_busy():
            return
        idx = _selected_history_idx["idx"]
        if idx is None:
            _set_status('请先点击选择一条历史记录', 'err')
            return

        # 显示进度
        _show_right_pane('progress')
        progress_bar.set_value(0)
        progress_text.set_text('正在加载历史记录...')
        progress_detail.set_text('')

        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                progress_text.set_text(msg)
                progress_detail.set_text(msg)
            except Exception:
                pass

        from nicegui import run
        try:
            result = await run.io_bound(detection.on_load_history, int(idx), _progress_callback)
            frame, info, status = result
        except Exception as _e:
            import traceback
            frame, info, status = None, "", f"❌ 加载历史异常: {_e}\n{traceback.format_exc()}"

        _show_right_pane('preview')
        if frame is not None:
            preview_image.set_source(video_utils.frame_to_base64(frame))
        # 同步路径输入框，显示当前加载的视频
        if state.video_state["path"]:
            path_input.set_value(state.video_state["path"])
        info_text.set_text(info)
        _set_status(status, 'ok' if frame is not None else 'err')
        _refresh_result_cards()

    # 页面初始化：不自动加载历史列表（空白初始状态），点「刷新」才加载
    _refresh_result_cards()


# ============ 启动 ============

if __name__ == "__main__":
    _out_dir = str(Path(state.CACHE_ROOT) / "demo_output")
    os.makedirs(_out_dir, exist_ok=True)

    # 预加载 YOLO 模型到 GPU 并在主线程预热推理。
    # 原因：检测在后台线程首次初始化 CUDA 上下文可能导致驱动层崩溃
    # （Windows 事件日志: nvcuda64.dll 0xC0000409），预热后后台线程只复用主线程上下文。
    _warmup_log = os.path.join(state.CACHE_ROOT, "warmup_status.log")
    try:
        import torch
        from app import get_ball_model, get_device
        model, _ = get_ball_model()
        device = get_device()
        if device != "cpu":
            warm = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(warm, conf=0.5, imgsz=640, device=device, verbose=False)
            torch.cuda.empty_cache()
            _msg = f"WARMUP-OK: {device} context created on main thread"
        else:
            _msg = "WARMUP-SKIP: CUDA not available, using CPU"
    except Exception as e:
        _msg = f"WARMUP-FAIL: {e}"
    with open(_warmup_log, "w", encoding="utf-8") as _f:
        _f.write(_msg)
    print(_msg, flush=True)

    ui.run(host="127.0.0.1", port=7871, title="进球集锦助手",
           dark=True, reload=False)
