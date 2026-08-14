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
    <style>
    body { background: #0f172a !important; margin: 0; overflow: hidden; }
    .nicegui-content { max-width: 100% !important; padding: 0 !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #1a2433; }
    ::-webkit-scrollbar-thumb { background: #3b4a63; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #4a5a78; }
    /* 折叠区头部标题不换行 */
    .q-expansion-item__header { white-space: nowrap; }
    /* 输入框聚焦时金色边框 */
    .q-field--outlined.q-field--focused .q-field__control { border-color: #FFB320 !important; }
    .q-field--outlined.q-field--focused .q-field__label { color: #FFB320 !important; }
    /* 透明金色按钮悬停时金色填充 */
    .btn-gold-outline:hover { background: rgba(255, 179, 32, 0.12); border-color: #FFB320; color: #FFB320; }
    </style>
    ''')

    with ui.column().classes('w-full h-[100dvh] bg-[#0f172a] p-0 gap-0').style('overflow: hidden'):
        # ====== 主容器：左右分栏 ======
        with ui.row().classes('w-full h-full'):

            # ========== 左侧面板 ==========
            with ui.column().classes('w-[350px] min-w-[350px] bg-[#1a2433] border-r border-[#2d3a4f]').style('height: 100%; overflow: hidden'):

                # 标题行 + 折叠功能区按钮（始终可见）
                with ui.row().classes('w-full items-center justify-between px-3 py-2 border-b border-[#2d3a4f]').style('flex-shrink: 0'):
                    ui.label('🏀 进球集锦助手').classes('text-white text-sm font-bold')
                    collapse_btn = ui.button('▲ 收起功能区', on_click=lambda: _toggle_func_collapse()).classes(
                        'bg-[#2d3a4f] text-gray-200 text-xs')

                # 功能区域（固定高度，紧凑）
                with ui.column().classes('w-full px-3 py-2 gap-1 border-b border-[#2d3a4f]').style('flex-shrink: 0') as func_container:

                    # 输入框 + 加载按钮（一行）
                    path_input = ui.input(value=state.DEFAULT_VIDEO, placeholder='文件路径').classes('w-full').props('dense')
                    info_text = ui.label('').classes('text-gray-400 text-xs font-mono hidden')
                    calib_status = ui.label('').classes('text-gray-400 text-xs font-mono hidden')
                    with ui.row().classes('w-full gap-2'):
                        ui.button('加载', on_click=lambda: _on_load()).classes('flex-1 bg-[#2d3a4f] text-gray-200 text-sm')
                        ui.button('重置', on_click=lambda: _on_reset()).classes('bg-[#2d3a4f] text-gray-200 text-sm')

                    # 开始识别
                    detect_btn = ui.button('开始识别', on_click=lambda: _on_detect()).classes(
                        'w-full bg-[#FFB320] text-black font-bold text-sm')

                    # 文件夹批量模式面板（加载文件夹后显示）
                    batch_panel = ui.column().classes('w-full gap-1 hidden')
                    with batch_panel:
                        batch_select = ui.select(options={}, value=None).classes('w-full').props('outlined dense dark')
                        with ui.row().classes('w-full gap-2'):
                            ui.button('保存标定', on_click=lambda: _on_batch_save_calib()).classes('flex-1 bg-[#2d3a4f] text-gray-200 text-xs')
                            batch_run_btn = ui.button('批量识别', on_click=lambda: _on_batch_run()).classes('flex-1 bg-[#FFB320] text-black text-xs')

                    # 结果状态
                    result_status = ui.label('').classes('text-gray-400 text-xs')

                    # 折叠区域：参数 / 集锦 / 历史 合并到一个框（展开时占满整框）
                    with ui.row().classes('w-full border border-[#2d3a4f] rounded-lg overflow-hidden').style('gap: 0; flex-wrap: wrap'):
                        exp_params = ui.expansion('参数', group='leftpanel').classes('w-1/3 text-gray-300 text-xs').style('min-width: 0').props('duration=0')
                        with exp_params:
                            with ui.column().classes('gap-1 w-full p-1 max-h-[300px] overflow-y-auto'):
                                yolo_3frame_switch = ui.switch('提速模式 (YOLO每3帧推理一次, 可能略漏检)', value=False).classes('w-full')
                                ui.label('默认每2帧（推荐, 更准）').classes('text-gray-500 text-[10px] -mt-1 mb-1')
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
                                        if auto_on:
                                            diff_threshold.props('disable')
                                            diff_threshold_label.set_text('帧差阈值: 自动（预热后计算）')
                                            diff_threshold_label.classes(add='text-[#FFB320]', remove='text-gray-400')
                                        else:
                                            diff_threshold.props(remove='disable')
                                            diff_threshold_label.set_text(f'帧差阈值: {diff_threshold.value}')
                                            diff_threshold_label.classes(add='text-gray-400', remove='text-[#FFB320]')
                                    auto_threshold_switch.on_value_change(lambda e: _sync_diff_threshold(e.value))
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
                    def _sync_expand(exp=None):
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
                        _e.on_value_change(lambda evt, x=_e: _sync_expand(x))
                    # 展开「历史」时自动加载记录，无需手动点刷新
                    exp_hist.on_value_change(lambda e: _refresh_history() if e.value else None)
                    _sync_expand()

                # 导出集锦按钮（固定在列表上方，仅列表有内容时显示）
                with ui.row().classes('w-full px-3 pt-2 flex-shrink-0 hidden') as export_row:
                    ui.button('导出集锦', on_click=lambda: _on_highlights()).classes(
                        'w-full bg-[#FFB320] text-black text-sm font-bold')

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
                        collapse_btn.classes(remove='bg-[#2d3a4f] text-gray-200').classes(
                            add='btn-gold-outline border border-[#FFB320]/60 bg-transparent text-[#FFB320]')
                    else:
                        func_container.classes(remove='hidden')
                        collapse_btn.set_text('▲ 收起功能区')
                        collapse_btn.classes(remove='btn-gold-outline border border-[#FFB320]/60 bg-transparent text-[#FFB320]').classes(
                            add='bg-[#2d3a4f] text-gray-200')

                def _toggle_func_collapse():
                    _set_func_collapsed(not _func_state["collapsed"])

            # ========== 右侧面板 ==========
            with ui.column().classes('flex-1 bg-[#0f172a] p-4 gap-3'):

                # 视频预览区
                preview_image = ui.interactive_image(
                    source='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
                ).classes('w-full rounded-xl bg-black').style('aspect-ratio: 16/9; object-fit: contain')

                result_video_el = ui.video(src='').classes('w-full rounded-xl hidden').style('aspect-ratio: 16/9')
                highlights_video_el = ui.video(src='').classes('w-full rounded-xl hidden').style('aspect-ratio: 16/9')

                # 进度显示区（检测时显示，居中）
                progress_container = ui.column().classes('w-full hidden items-center justify-center').style('aspect-ratio: 16/9')
                with progress_container:
                    progress_text = ui.label('检测中...').classes('text-[#FFB320] text-lg font-semibold')
                    progress_bar = ui.linear_progress(show_value=False).classes('w-64 mt-3')
                    progress_bar.style('background-color: #2d3a4f; color: #FFB320')
                    progress_detail = ui.label('').classes('text-gray-400 text-xs mt-2')

                # 底部留白
                ui.label('').classes('h-4')

    # ====== 事件处理函数 ======
    def _set_status(text, kind='info'):
        """设置结果状态文本并切换颜色（ok=绿 / err=红 / busy=金 / info=灰）。"""
        result_status.set_text(text)
        result_status.classes(
            remove='text-green-400 text-red-400 text-[#FFB320] text-gray-400')
        if kind == 'ok':
            result_status.classes(add='text-green-400')
        elif kind == 'err':
            result_status.classes(add='text-red-400')
        elif kind == 'busy':
            result_status.classes(add='text-[#FFB320]')
        else:
            result_status.classes(add='text-gray-400')

    def _on_load():
        path = path_input.value
        # 文件夹路径 → 批量标定 + 批量识别模式
        if path and os.path.isdir(path.strip().strip('"')):
            files = video_utils.scan_video_files(path)
            if not files:
                _set_status('文件夹内没有找到视频文件', 'err')
                return
            state.batch_files = files
            state.batch_calibs = {}
            state.batch_current_video = None
            batch_panel.classes(remove='hidden')
            _refresh_batch_list()
            # 自动加载第一个视频（同时同步下拉框）
            _on_batch_load_video(files[0])
            _set_status(f'批量模式 | 扫描到 {len(files)} 个视频，逐个标定后批量识别', 'info')
            return
        # 单视频文件路径 → 原有流程
        batch_panel.classes(add='hidden')
        frame, info = detection.load_video(path)
        if frame is not None:
            # 切换视频：清空上一轮的进球结果，避免列表残留旧视频数据
            state.last_goal_clips.clear()
            state.last_goals.clear()
            state.kept_goal_indices.clear()
            _refresh_result_cards()
            b64 = video_utils.frame_to_base64(frame)
            preview_image.set_source(b64)
            preview_image.classes(remove='hidden')
            result_video_el.classes(add='hidden')
            highlights_video_el.classes(add='hidden')
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

    def _on_batch_load_video(path=None):
        """加载批量视频（从下拉或列表点击）。"""
        nonlocal _batch_loading
        # 兼容旧版（值,标签）元组，防御性解包
        if isinstance(path, (tuple, list)):
            path = path[0]
        if _batch_loading:
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
        try:
            batch_select.set_value(path)
            frame, info, status = detection.on_batch_load_video(path)
            if frame is not None:
                # 切换批量视频：清空上一轮的进球结果
                state.last_goal_clips.clear()
                state.last_goals.clear()
                state.kept_goal_indices.clear()
                _refresh_result_cards()
                b64 = video_utils.frame_to_base64(frame)
                preview_image.set_source(b64)
                preview_image.classes(remove='hidden')
                result_video_el.classes(add='hidden')
                highlights_video_el.classes(add='hidden')
            info_text.set_text(info)
            calib_status.set_text(status)
        finally:
            _batch_loading = False

    def _on_batch_save_calib():
        status = detection.on_batch_save_calib()
        calib_status.set_text(status)
        _refresh_batch_list()
        _set_status(status, 'ok' if '已保存' in status else 'err')

    _batch_running = {"active": False}

    async def _on_batch_run():
        if _batch_running["active"]:
            # 批量中点击 → 请求取消，run_batch_detect 轮询后中断
            state.cancel_requested = True
            batch_run_btn.set_text('正在取消...')
            batch_run_btn.disable()
            return
        if not state.batch_files:
            _set_status('请先加载文件夹', 'err')
            return
        _batch_running["active"] = True
        state.cancel_requested = False
        batch_run_btn.set_text('取消')
        batch_run_btn.enable()
        # 显示进度
        progress_container.classes(remove='hidden')
        preview_image.classes(add='hidden')
        result_video_el.classes(add='hidden')
        highlights_video_el.classes(add='hidden')
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
        status, ok = await run.io_bound(
            detection.run_batch_detect,
            start_frame.value, end_frame.value, ball_conf.value, min_gap.value,
            diff_threshold.value, min_circularity.value, int(min_in_hoop_frames.value),
            min_blob_area.value, search_margin.value,
            progress_callback=_progress_callback,
            auto_threshold=auto_threshold_switch.value,
            yolo_step=3 if yolo_3frame_switch.value else 2)

        _batch_running["active"] = False
        batch_run_btn.set_text('批量识别')
        batch_run_btn.enable()
        progress_container.classes(add='hidden')
        preview_image.classes(remove='hidden')
        _set_status(status, 'ok' if ok else 'err')
        # 批量结束后显示最后一个视频的结果
        _refresh_result_cards()
        _refresh_batch_list()

    def _on_image_click(e):
        """点击预览图标定篮筐。

        使用 ui.interactive_image 的 on_mouse 事件，e.image_x/e.image_y
        已由前端按 显示尺寸/原始尺寸 比例换算为原始帧坐标。
        """
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
    batch_select.on_value_change(lambda e: _on_batch_load_video(e.value))

    def _on_reset():
        status = detection.reset_hoop()
        calib_status.set_text(status)
        # 刷新预览
        if state.video_state["path"]:
            frame, _ = detection.preview_frame(state.video_state["current_frame"])
            if frame is not None:
                preview_image.set_source(video_utils.frame_to_base64(frame))

    _detecting = {"active": False}

    async def _on_detect():
        if _detecting["active"]:
            # 检测中点击 → 请求取消，run_detect 轮询后中断
            state.cancel_requested = True
            detect_btn.set_text('正在取消...')
            detect_btn.disable()
            return
        _detecting["active"] = True
        state.cancel_requested = False
        detect_btn.set_text('取消')
        detect_btn.enable()
        # 显示进度条
        progress_container.classes(remove='hidden')
        preview_image.classes(add='hidden')
        result_video_el.classes(add='hidden')
        highlights_video_el.classes(add='hidden')
        progress_bar.set_value(0)
        progress_text.set_text('正在加载模型...')
        progress_detail.set_text('')

        # 进度回调函数
        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                progress_text.set_text(msg)
            except Exception:
                pass

        # 使用 run.io_bound 在后台线程执行，避免阻塞事件循环
        from nicegui import run
        status, ok = await run.io_bound(
            detection.run_detect,
            start_frame.value, end_frame.value, ball_conf.value, min_gap.value,
            diff_threshold.value, min_circularity.value, int(min_in_hoop_frames.value),
            min_blob_area.value, search_margin.value,
            progress_callback=_progress_callback,
            auto_threshold=auto_threshold_switch.value,
            yolo_step=3 if yolo_3frame_switch.value else 2)

        _detecting["active"] = False
        # 隐藏进度条，显示结果
        progress_container.classes(add='hidden')
        preview_image.classes(remove='hidden')
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
            t_min, t_sec = int(ts // 60), ts % 60
            end_ts = ts + 10
            end_min, end_sec = int(end_ts // 60), end_ts % 60
            with result_container:
                with ui.card().props('flat').classes('w-full bg-[#1f2b3d] rounded-lg px-2 py-1.5').style('margin: 0; border: 1px solid #334155'):
                    # 第一行：序号 + 时间范围（紧凑）
                    with ui.row().classes('w-full items-center gap-2 mb-1'):
                        ui.label(str(i+1)).classes(
                            'bg-[#FFB320] text-black font-bold text-xs w-5 h-5 flex items-center justify-center rounded-full')
                        ui.label(f'{t_min}:{t_sec:04.1f} - {end_min}:{end_sec:04.1f}').classes('text-white text-sm font-bold font-mono')
                    # 第二行：操作按钮（删除用危险色区分，填满宽度）
                    with ui.row().classes('w-full gap-1'):
                        ui.button('预览', on_click=lambda e, idx=i: _on_preview_clip(idx)).classes(
                            'flex-1 border border-gray-600 bg-transparent text-gray-300 text-xs rounded-lg py-1')
                        ui.button('导出', on_click=lambda e, idx=i: _on_export_clip(idx)).classes(
                            'flex-1 border border-gray-600 bg-transparent text-gray-300 text-xs rounded-lg py-1')
                        ui.button('删除', on_click=lambda e, idx=i: _on_delete_clip(idx)).classes(
                            'flex-1 border border-red-500/60 bg-transparent text-red-400 text-xs rounded-lg py-1')

    def _on_preview_clip(idx):
        path, status = detection.clip_action("preview", idx)
        if path and os.path.exists(path):
            result_video_el.set_source(path)
            preview_image.classes(add='hidden')
            result_video_el.classes(remove='hidden')
            highlights_video_el.classes(add='hidden')
            # 自动播放
            result_video_el.run_method('play')
        _set_status(status, 'info')

    def _on_export_clip(idx):
        path, status = detection.clip_action("export", idx)
        if path and os.path.exists(path):
            ui.download(path)
        _set_status(status, 'ok' if path and os.path.exists(path) else 'err')

    def _on_delete_clip(idx):
        # 点击直接删除，不弹确认框
        detection.clip_action("delete", idx)
        _refresh_result_cards()
        _set_status(f'已删除第 {idx+1} 个片段 | 剩余 {len(state.last_goal_clips)} 个', 'info')

    async def _on_highlights():
        from nicegui import run
        # 在右侧预览区显示进度
        progress_container.classes(remove='hidden')
        preview_image.classes(add='hidden')
        result_video_el.classes(add='hidden')
        highlights_video_el.classes(add='hidden')
        progress_bar.set_value(0)
        progress_text.set_text('正在生成集锦...')
        progress_detail.set_text('')

        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                progress_text.set_text(msg)
            except Exception:
                pass

        _set_status('正在生成集锦...', 'busy')
        path, status = await run.io_bound(
            detection.generate_highlights, hl_pre_roll.value, hl_post_roll.value,
            hl_min_gap.value, _progress_callback)

        progress_container.classes(add='hidden')
        if path and os.path.exists(path):
            ui.download(path)
            highlights_video_el.set_source(path)
            preview_image.classes(add='hidden')
            result_video_el.classes(add='hidden')
            highlights_video_el.classes(remove='hidden')
        else:
            preview_image.classes(remove='hidden')
        _set_status(status, 'ok' if path and os.path.exists(path) else 'err')

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
            selected = (_selected_history_idx.get("idx") == i)
            row_cls = ('border border-[#FFB320] bg-[#2a2f3e]'
                       if selected else 'border border-transparent hover:bg-[#2a2f3e]')
            with history_list:
                def _make_click(idx=i):
                    async def _on_click():
                        _selected_history_idx["idx"] = idx
                        _refresh_history()
                        await _on_load_history()
                        exp_hist.set_value(False)  # 加载完成后自动收起历史面板
                    return _on_click
                with ui.row().classes(f'w-full items-center gap-1 p-1 rounded cursor-pointer {row_cls}').on('click', _make_click(i)):
                    ui.label(f'{i+1}.').classes('text-[#FFB320] text-xs font-bold')
                    ui.label(name).classes('text-gray-300 text-xs flex-1 truncate')
                    ui.label(f'{goals}球').classes('text-gray-400 text-xs')

    async def _on_load_history():
        idx = _selected_history_idx.get("idx")
        if idx is None:
            _set_status('请先点击选择一条历史记录', 'err')
            return

        # 显示进度
        progress_container.classes(remove='hidden')
        preview_image.classes(add='hidden')
        progress_bar.set_value(0)
        progress_text.set_text('正在加载历史记录...')

        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                progress_text.set_text(msg)
            except Exception:
                pass

        from nicegui import run
        result = await run.io_bound(detection.on_load_history, int(idx), _progress_callback)
        frame, info, status = result

        progress_container.classes(add='hidden')
        preview_image.classes(remove='hidden')
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
