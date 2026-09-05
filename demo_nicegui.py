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
import asyncio
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
from nicegui import ui

from services import state, detection, video_utils

# 启动时从磁盘恢复片段缓存
state.init_clip_cache()
# 存量人物名单回填：全局名单上线前的旧分类（仅写在各视频 labels.persons）
# 一次性补收进 persons.json（幂等），使旧场次建的人物在新场次可选
state.harvest_persons_from_history()


# ============ 人物分类配色 ============
# 人物徽章/快捷选择/统计行统一从该调色板取色（暗色主题高可读）。
# 分配策略：按人物名排序循环取色 —— 同屏人物（通常 2-6 个）不撞色；
# 新增人物可能使整体色相偏移，但保证任何时刻同屏颜色唯一。
_PERSON_PALETTE = [
    ('#22d3ee', 'rgba(34, 211, 238, 0.14)'),   # cyan（与全局 accent 同色，排第一）
    ('#a78bfa', 'rgba(167, 139, 250, 0.14)'),  # violet
    ('#f472b6', 'rgba(244, 114, 182, 0.14)'),  # pink
    ('#fbbf24', 'rgba(251, 191, 36, 0.14)'),   # amber
    ('#4ade80', 'rgba(74, 222, 128, 0.14)'),   # green
    ('#fb923c', 'rgba(251, 146, 60, 0.14)'),   # orange
    ('#38bdf8', 'rgba(56, 189, 248, 0.14)'),   # sky
    ('#e879f9', 'rgba(232, 121, 249, 0.14)'),  # fuchsia
]


def _person_color_map(clips):
    """从 clips 列表构建 {人物名: (文字色, 背景色)} 映射（按名字排序循环取色）。"""
    names = sorted({c.get("person") for c in (clips or []) if c.get("person")})
    return {n: _PERSON_PALETTE[i % len(_PERSON_PALETTE)]
            for i, n in enumerate(names)}


# ============ YOLO/CUDA 启动自检 ============

YOLO_SELFCHECK = {"ok": False, "msg": ""}


def _yolo_selfcheck():
    """服务启动时在主线程做一次真实 YOLO 推理自检。

    作用：
      1. 提前在主线程创建 CUDA 上下文（后台线程首次创建可能触发驱动层崩溃）
      2. 驱动升级/环境损坏导致推理失败时，启动即可发现并在页面横幅提示，
         避免跑完整个视频（几十分钟）才发现 GPU 推理一直失败
    策略：不做 CPU 降级 —— CUDA 不可用或推理失败均判为失败（红色横幅），
    检测入口也会拒绝在 CPU 上运行（见 services/detection.py run_detect）。
    结果写入 YOLO_SELFCHECK（页面横幅读取）和 cache/warmup_status.log。
    """
    try:
        import torch
        from app import get_ball_model, get_device
        device = get_device()
        if device == "cpu":
            YOLO_SELFCHECK.update(ok=False,
                                  msg="CUDA 不可用（本服务需要 GPU 推理，不支持 CPU 降级）")
            _msg = "WARMUP-FAIL: CUDA not available, CPU fallback disabled"
        else:
            model, _weights = get_ball_model()
            warm = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(warm, conf=0.5, imgsz=640, device=device, verbose=False)
            torch.cuda.empty_cache()
            YOLO_SELFCHECK.update(ok=True, msg=f"YOLO 自检通过（{device}）")
            _msg = f"WARMUP-OK: {device} context created on main thread"
    except Exception as e:
        import traceback
        YOLO_SELFCHECK.update(ok=False, msg=f"YOLO 自检失败：{e}")
        _msg = f"WARMUP-FAIL: {e}\n{traceback.format_exc()}"
    try:
        os.makedirs(state.CACHE_ROOT, exist_ok=True)
        with open(os.path.join(state.CACHE_ROOT, "warmup_status.log"), "w", encoding="utf-8") as f:
            f.write(_msg)
    except Exception:
        pass
    print(_msg, flush=True)  # 自检在 setup_logging 之前执行，直接 print（start 脚本会 tee）


_yolo_selfcheck()


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
        # ====== YOLO 自检横幅（失败时醒目提示 + 确认重启） ======
        if not YOLO_SELFCHECK["ok"]:
            def _do_restart_service():
                """重启服务：分离启动新进程（延迟 1s 等旧进程退出释放端口）后硬退出旧进程。

                旧实现 os.execv 在 Windows 上不保证旧进程 socket 立即释放，
                新进程绑定同一端口可能失败导致服务下线（semantics 与 Unix 不同）。
                """
                try:
                    import subprocess
                    _inner = [sys.executable, '-u', str(ROOT / 'demo_nicegui.py')]
                    subprocess.Popen(
                        [sys.executable, '-u', '-c',
                         'import subprocess,sys,time;time.sleep(1.0);'
                         'sys.exit(subprocess.call(%r))' % (_inner,)],
                        creationflags=state.SBOX)
                except Exception as e:
                    # timer 回调线程内失败不应静默（否则点击无响应无从排查）
                    ui.notify(f'重启失败: {e}，请手动重启服务', type='negative', position='top')
                    return
                # 硬退出：跳过清理钩子立即释放端口/句柄，新进程 1s 后接管
                os._exit(0)

            def _on_confirm_restart():
                restart_dialog.close()
                ui.notify('正在重启服务，请稍后刷新页面…', type='info', position='top')
                # 延迟 1 秒让通知先送达浏览器
                ui.timer(1.0, _do_restart_service, once=True)

            with ui.row().classes('w-full items-center gap-2 px-3 py-2').style(
                    'flex-shrink: 0; background: rgba(239, 68, 68, 0.12); border-bottom: 1px solid #EF4444;'):
                ui.icon('error', color='red-5').classes('text-sm flex-shrink-0')
                ui.label(f'{YOLO_SELFCHECK["msg"]} | 请检查显卡驱动/CUDA 环境后重启服务').classes(
                    'text-xs flex-1').style('color: #FCA5A5')
                ui.button('重启服务', on_click=lambda: restart_dialog.open()).props('dense unelevated').classes(
                    'text-xs flex-shrink-0').style('background: #EF4444; color: white')

            with ui.dialog() as restart_dialog, ui.card().classes('w-80'):
                ui.label('确认重启服务？').classes('text-sm font-bold')
                ui.label('将以相同参数重新启动进程并重新执行 YOLO 自检（适用于修复显卡驱动/CUDA 环境后）。').classes(
                    'text-xs mt-1').style('color: var(--text-secondary)')
                with ui.row().classes('w-full justify-end gap-2 mt-3'):
                    ui.button('取消', on_click=restart_dialog.close).props('flat').classes(
                        'text-xs').style('color: var(--text-secondary)')
                    ui.button('确认重启', on_click=_on_confirm_restart).classes('text-xs').style(
                        'background: #EF4444; color: white')

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

                    # 断点续识别提示（有未完成检测时显示）
                    resume_hint = ui.label('').classes('text-xs text-center w-full').style('color: var(--accent)')
                    resume_hint.visible = False

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
                    async def _on_hist_expand(e):
                        if e.value:
                            await _refresh_history()
                    exp_hist.on_value_change(_on_hist_expand)
                    _sync_expand()

                # 后台批量识别进度条（流水线模式常驻：预览快照片段顶掉右侧进度面板后，
                # 仍可从此处看到后台进度，点击切回完整进度视图）
                with ui.row().classes('w-full px-3 pt-2 flex-shrink-0 items-center gap-2 hidden cursor-pointer') \
                        .on('click', lambda: _show_right_pane('progress')) as batch_progress_strip:
                    ui.icon('sync', color='accent').classes('text-sm animate-spin')
                    batch_mini_bar = ui.linear_progress(value=0, show_value=False).props('instant-feedback').classes('flex-1')
                    batch_mini_text = ui.label('后台识别中').classes('text-xs flex-shrink-0').style('color: var(--text-secondary)')

                # 流水线集锦进度条（批量运行中对快照视频生成集锦时显示，与批量进度条并列互不干扰）
                with ui.row().classes('w-full px-3 pt-1 flex-shrink-0 items-center gap-2 hidden') as hl_progress_strip:
                    ui.icon('movie', color='accent').classes('text-sm')
                    hl_mini_bar = ui.linear_progress(value=0, show_value=False).props('instant-feedback').classes('flex-1')
                    hl_mini_text = ui.label('集锦生成中').classes('text-xs flex-shrink-0').style('color: var(--text-secondary)')

                # 导出集锦按钮（固定在列表上方，仅列表有内容时显示）
                # 人物筛选下拉：全部人物 / 已分类的各个人物（按人物导出集锦）；
                # 全局名单里的其他场次人物也可选（用于整场合并导出，无计数后缀）
                with ui.row().classes('w-full px-3 pt-2 flex-shrink-0 hidden items-center gap-2') as export_row:
                    hl_person_select = ui.select(
                        {'': '全部人物'}, value='').props('dense outlined standout dark'
                        ).classes('text-xs flex-1').style('max-width: 55%')
                    ui.button('导出集锦', on_click=lambda: _on_highlights()).classes(
                        'flex-1 text-sm font-bold').props('ripple').style('background: var(--accent); color: var(--bg-canvas)')
                    ui.button('整场集锦', on_click=lambda: _on_fullgame_highlights()).classes(
                        'flex-1 text-xs font-bold').props('ripple').style(
                        'background: var(--bg-elevated); color: var(--accent); '
                        'border: 1px solid var(--accent)').tooltip(
                        '合并文件夹内全部视频（四节）导出：按当前人物筛选，输出 {文件夹名}-{人物}-highlights.mp4')

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

                # 帧选择器（标定前拖动浏览帧；随预览图同显隐）
                frame_select_container = ui.row().classes('w-full items-center gap-3 hidden')
                with frame_select_container:
                    frame_slider = ui.slider(min=0, max=0, step=1, value=0).classes('flex-1')
                    frame_pos_label = ui.label('帧 0').classes('text-gray-400 text-xs font-mono').style(
                        'min-width: 132px; text-align: right; flex-shrink: 0')

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
        # 帧选择器跟随预览图显隐（仅已加载多帧视频时才有意义）
        if mode == 'preview' and (state.video_state.get('total') or 0) > 1:
            frame_select_container.classes(remove='hidden')
        else:
            frame_select_container.classes(add='hidden')

    # ====== 帧选择器逻辑 ======
    _frame_sel = {"busy": False, "pending": None}

    def _fmt_frame_pos(idx, total, fps):
        m, s = divmod(idx / max(fps, 1e-6), 60)
        return f'帧 {idx} / {max(total - 1, 0)}  {int(m):02d}:{s:04.1f}'

    def _sync_frame_selector():
        """视频加载后同步滑块范围/位置（各加载路径已将 current_frame 重置为 0）。"""
        total = int(state.video_state.get('total') or 0)
        if total <= 1:
            frame_select_container.classes(add='hidden')
            return
        frame_slider._props.update({'min': 0, 'max': total - 1, 'step': 1})
        frame_slider.set_value(0)
        frame_slider.update()
        frame_pos_label.set_text(_fmt_frame_pos(0, total, state.video_state.get('fps') or 30.0))
        frame_select_container.classes(remove='hidden')

    async def _on_frame_slider_change(e):
        """拖动选帧：更新 current_frame 并刷新预览。拖动用 catch-up：解码中只记 pending。"""
        idx = int(e.value)
        total = int(state.video_state.get('total') or 0)
        if total > 0:
            idx = max(0, min(idx, total - 1))
        frame_pos_label.set_text(_fmt_frame_pos(idx, total, state.video_state.get('fps') or 30.0))
        if state.current_task() is not None:
            return
        if _frame_sel["busy"]:
            _frame_sel["pending"] = idx
            return
        _frame_sel["busy"] = True
        try:
            from nicegui import run
            while True:
                target = _frame_sel["pending"]
                _frame_sel["pending"] = None
                if target is None:
                    target = idx
                state.video_state["current_frame"] = int(target)
                result = await run.io_bound(detection.preview_frame, int(target))
                if result is not None and result[0] is not None:
                    preview_image.set_source(video_utils.frame_to_base64(result[0]))
                if _frame_sel["pending"] is None:
                    break
        finally:
            _frame_sel["busy"] = False

    frame_slider.on_value_change(_on_frame_slider_change)

    # ====== 全局任务互斥：锁在 services.state（进程级，跨页面连接/刷新共享）======
    # 旧实现 _busy 是页面函数局部变量：NiceGUI 每个连接/刷新独立执行页面函数，
    # 刷新后旧检测线程还在跑、新页面锁为空可再启动任务 → 并发写全局 state。
    def _task_label(task):
        return {'detect': '开始识别', 'batch': '批量识别',
                'highlights': '生成集锦', 'load': '加载视频'}.get(task, '后台任务')

    def _try_acquire(task):
        """尝试占用任务锁；已有任务运行则提示并返回 0。返回值为 token（传给后台函数）。"""
        token = state.try_acquire_task(task)
        if not token:
            _set_status(f'「{_task_label(state.current_task())}」正在进行，请等待完成或取消', 'err')
        return token

    def _refuse_if_busy():
        """轻量操作守卫：有任务运行时提示并返回 True（调用方直接 return）。"""
        cur = state.current_task()
        if cur is not None:
            _set_status(f'「{_task_label(cur)}」正在进行，请稍后再试', 'err')
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
            _cards_video["path"] = None  # 离开快照查看模式
            _refresh_result_cards()
            state.batch_files = files
            state.batch_calibs = {}
            state.batch_current_video = None
            state.batch_results.clear()  # 清掉上一轮快照，防止新文件夹同名视频显示旧结果
            # 跨会话复用标定：该文件夹此前跑过批量识别的话，历史记录里存有
            # 每个视频的篮筐标定，回填后列表直接显示 ✓、可直接批量识别
            _n_cal = detection.backfill_batch_calibs_from_history()
            batch_panel.classes(remove='hidden')
            _refresh_batch_list()
            # 自动加载第一个视频（同时同步下拉框）
            await _on_batch_load_video(files[0])
            _cal_hint = f'，已从历史恢复 {_n_cal} 个标定' if _n_cal else ''
            _set_status(f'批量模式 | 扫描到 {len(files)} 个视频{_cal_hint}，逐个标定后批量识别', 'info')
            return
        # 单视频文件路径 → 原有流程（清空批量状态，避免写历史误带 batch_idx）
        # 必须占任务锁：io_bound 期间事件循环会让出，双击「加载」会并发跑两个
        # load_video 线程并发改 video_state/calib（批量路径 _on_batch_load_video
        # 同样持锁，旧实现唯独此路径只检查不占锁）
        token = _try_acquire('load')
        if not token:
            return
        batch_panel.classes(add='hidden')
        state.batch_files = []
        state.batch_calibs = {}
        state.batch_current_video = None
        state.batch_results.clear()
        _cards_video["path"] = None  # 离开快照查看模式
        # 重要：state.last_goal_clips/last_goals/kept_goal_indices 的清空
        #       已下沉到 detection.load_video 业务层（切视频即清空），UI 层不再重复
        #       以避免 frame is None 分支漏清空导致旧数据残留
        # 开容器 + seek + 解码第 0 帧在大视频上可达数百 ms~秒级，
        # 走 io_bound 避免阻塞 UI 事件循环（与批量加载路径一致）
        from nicegui import run
        try:
            frame, info = await run.io_bound(detection.load_video, path,
                                             task_token=token)
        except Exception as _e:
            import traceback
            frame, info = None, f"❌ 加载视频异常: {_e}\n{traceback.format_exc()}"
            state.release_task(token)  # io_bound 未启动/启动即异常时后台 finally 不会执行
        # 正常路径锁由 load_video 内部 finally 释放（锁归任务本体）
        if frame is not None:
            # 切换视频：仅清 UI 卡片容器缓存（state 已由 load_video 清空）
            _refresh_result_cards()
            b64 = video_utils.frame_to_base64(frame)
            preview_image.set_source(b64)
            _show_right_pane('preview')
            _sync_frame_selector()
        else:
            # 加载失败也必须刷新卡片（state 已在 load_video 里清空，UI 要同步显示空列表）
            _refresh_result_cards()
        info_text.set_text(info)
        calib_status.set_text('拖动滑块选择帧，点击画面 2 个点标定篮筐' if frame is not None else info)
        # 更新断点续识别提示
        if frame is not None and state.video_state["path"]:
            _cp = state.load_checkpoint(state.video_state["path"])
            if _cp is not None:
                _cf = _cp.get("current_frame", 0)
                _ng = len(_cp.get("detector_state", {}).get("goals", []))
                resume_hint.set_text(f"💡 发现未完成检测（帧 {_cf}，{_ng} 球），点击识别可从断点继续")
                resume_hint.visible = True
            else:
                resume_hint.visible = False
        else:
            resume_hint.visible = False

    def _refresh_batch_list():
        """刷新批量下拉框：三态标记（☑已完成n球 / ✓已标定 / ○未标定），保留当前选中。"""
        if not state.batch_files:
            return
        cur_val = batch_select.value if batch_select.value in state.batch_files else None

        def _label(f):
            name = os.path.basename(f)
            if f in state.batch_results:  # 流水线快照：本轮批量已完成
                return f'☑ {name} ({len(state.batch_results[f]["goals"])}球)'
            if f in state.batch_calibs:
                return f'✓ {name}'
            return f'○ {name}'

        # 用 dict（值->标签）作为 options：dict 时 select 的值才是纯路径字符串；
        # 若用 [(值,标签)] 列表，NiceGUI 会把整个元组当值，导致加载失败
        batch_select.set_options({f: _label(f) for f in state.batch_files},
                                 value=cur_val)

    _batch_loading = False  # 防重入（set_value 可能触发 change 事件）

    # 当前结果卡片显示的视频：None=全局模式（单视频/批量最后结果）
    # 非空=批量快照模式（流水线：后台检测继续跑，前台确认该视频的快照结果）
    _cards_video = {"path": None}
    # 流水线集锦小锁在 services.state（跨页面连接共享），此处不再用页面局部 dict

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
        if path is None:
            path = batch_select.value
        # ===== 流水线模式：批量识别运行中 =====
        # 点已完成视频 → 只切换卡片到该视频快照（纯展示，不动全局 state/标定，不打断后台检测）
        if state.current_task() == 'batch':
            if path and path in state.batch_results:
                batch_select.set_value(path)
                _cards_video["path"] = path
                _refresh_result_cards()
                _set_status(f'查看: {os.path.basename(path)} | '
                            f'{len(state.batch_results[path]["goals"])} 个进球 | 后台检测继续', 'info')
                return
            _set_status('批量识别进行中，仅可点击「☑已完成」的视频查看结果', 'err')
            return
        if _refuse_if_busy():
            return
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
        _cards_video["path"] = None  # 常规加载走全局 state，卡片回到全局模式
        token = _try_acquire('load')
        if not token:
            _batch_loading = False
            return
        try:
            batch_select.set_value(path)
            # 若该视频已有检测结果，生成预览片段可能耗时（缓存未命中时 ~40s/50球），
            # 用 io_bound 避免阻塞 UI；同时切到进度面板给出可见反馈（旧实现
            # 进度写在隐藏容器里，界面静止像卡死）
            from nicegui import run
            _show_right_pane('progress')
            progress_bar.set_value(0.3)
            progress_text.set_text(f'正在加载 {os.path.basename(path)}...')
            progress_detail.set_text('')

            def _progress_callback(pct, msg):
                try:
                    progress_bar.set_value(max(0.3, pct / 100))
                    progress_text.set_text(f'正在加载 {os.path.basename(path)}')
                    progress_detail.set_text(msg)
                except Exception:
                    pass

            try:
                frame, info, status = await run.io_bound(
                    detection.on_batch_load_video, path, _progress_callback,
                    task_token=token)
            except Exception as _e:
                import traceback
                frame, info, status = None, "", f"❌ 加载视频异常: {_e}\n{traceback.format_exc()}"
                state.release_task(token)  # io_bound 未启动/启动即异常时后台 finally 不会执行
            _show_right_pane('preview')
            if frame is not None:
                b64 = video_utils.frame_to_base64(frame)
                preview_image.set_source(b64)
                _sync_frame_selector()
            info_text.set_text(info)
            calib_status.set_text(status)
            # 刷新结果卡片（已检测过的视频会显示进球列表）
            _refresh_result_cards()
            # 更新断点续识别提示
            if frame is not None and state.video_state["path"]:
                _cp = state.load_checkpoint(state.video_state["path"])
                if _cp is not None:
                    _cf = _cp.get("current_frame", 0)
                    _ng = len(_cp.get("detector_state", {}).get("goals", []))
                    resume_hint.set_text(f"💡 发现未完成检测（帧 {_cf}，{_ng} 球），点击识别可从断点继续")
                    resume_hint.visible = True
                else:
                    resume_hint.visible = False
            else:
                resume_hint.visible = False
        finally:
            # 锁已下沉到 on_batch_load_video 内部 finally（锁归任务本体）：
            # 未命中片段缓存时后台会跑 ffmpeg 数十秒，页面刷新取消 UI 协程后
            # 线程仍持锁跑到结束，UI 侧不再提前释放（与 detect/batch/highlights 对齐）
            _batch_loading = False

    def _on_batch_save_calib():
        if _refuse_if_busy():
            return
        status = detection.on_batch_save_calib()
        calib_status.set_text(status)
        _refresh_batch_list()
        _set_status(status, 'ok' if '已保存' in status else 'err')

    async def _on_batch_run():
        if state.current_task() == 'batch':
            # 批量中点击 → 请求取消，run_batch_detect 轮询后中断
            state.cancel_event.set()
            batch_run_btn.set_text('正在取消...')
            batch_run_btn.disable()
            return
        if not state.batch_files:
            _set_status('请先加载文件夹', 'err')
            return
        token = _try_acquire('batch')
        if not token:
            return
        state.cancel_event.clear()
        batch_progress_strip.classes(remove='hidden')  # 显示常驻迷你进度条
        batch_mini_bar.set_value(0)
        batch_mini_text.set_text('后台识别中')
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
                # 主行直接显示完整消息（含当前视频进度 + 整批 ETA），
                # 右侧面板被预览顶掉时迷你条仍有百分比兜底
                progress_text.set_text(msg)
                progress_detail.set_text(msg)
                # 迷你进度条同步（右侧面板被预览顶掉时，用户仍能看到后台进度）
                batch_mini_bar.set_value(pct / 100)
                batch_mini_text.set_text(f'{pct:.0f}%')
            except Exception:
                pass

        def _per_video_done(video_path, goal_count):
            """流水线回调：单个视频完成 → 更新下拉框标记 + 轻提示（不自动切换界面，不打断确认）"""
            try:
                _refresh_batch_list()
                _set_status(f'☑ {os.path.basename(video_path)} 完成 | {goal_count} 个进球 | 可点击下拉查看', 'info')
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
                skip_yolo_no_motion=skip_yolo_switch.value,
                per_video_callback=_per_video_done,
                task_token=token)
        except Exception as _e:
            import traceback
            status = f"❌ 批量识别异常: {_e}\n{traceback.format_exc()}"
            ok = False
            state.release_task(token)  # io_bound 未启动/启动即异常时后台 finally 不会执行
        # 注意：正常路径锁由 run_batch_detect 内部 finally 释放（锁归任务本体，
        # 页面刷新取消 UI 协程时后台线程仍持锁跑到结束）
        batch_run_btn.set_text('批量识别')
        batch_run_btn.enable()
        batch_progress_strip.classes(add='hidden')  # 批量结束，隐藏迷你进度条
        _show_right_pane('preview')
        _set_status(status, 'ok' if ok else 'err')
        # 批量结束后：用户查看中的快照保留显示；否则回全局模式显示最后一个视频的结果
        if _cards_video["path"] not in state.batch_results:
            _cards_video["path"] = None
        _refresh_result_cards()
        _refresh_batch_list()

    async def _on_image_click(e):
        """点击预览图标定篮筐。

        使用 ui.interactive_image 的 on_mouse 事件，e.image_x/e.image_y
        已由前端按 显示尺寸/原始尺寸 比例换算为原始帧坐标。
        read_frame（开容器+seek+解码）走 io_bound：大视频上同步执行
        会冻结 UI 0.5~2 秒，批量标定 2N 次点击体验极差。
        """
        if state.current_task() is not None:
            # 进程级锁（含另一页面连接启动的批量任务）运行中禁止标定：
            # 否则批量线程会清掉用户的 clicks / 覆盖 hoop
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
            from nicegui import run
            frame, status = await run.io_bound(detection.click_calibrate, x, y)
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

    async def _on_reset():
        if _refuse_if_busy():
            return
        status = detection.reset_hoop()
        calib_status.set_text(status)
        # 刷新预览（read_frame 走 io_bound，避免大视频上同步解码卡 UI）
        if state.video_state["path"]:
            from nicegui import run
            result = await run.io_bound(detection.preview_frame, state.video_state["current_frame"])
            if result is not None:
                frame, _ = result
                if frame is not None:
                    preview_image.set_source(video_utils.frame_to_base64(frame))

    async def _on_detect():
        if state.current_task() == 'detect':
            # 检测中点击 → 请求取消，run_detect 轮询后中断
            state.cancel_event.set()
            detect_btn.set_text('正在取消...')
            detect_btn.disable()
            return

        # ===== 断点续识别：检查是否有未完成的检测 =====
        vp = state.video_state["path"]
        if vp and state.has_checkpoint(vp):
            cp = state.load_checkpoint(vp)  # params=None 取最新
            if cp is not None:
                cf = cp.get("current_frame", 0)
                goals = cp.get("detector_state", {}).get("goals", [])
                saved_at = cp.get("saved_at", "")
                msg = (f"发现未完成检测\n"
                       f"  已处理到帧 {cf}\n"
                       f"  已检测 {len(goals)} 个进球\n"
                       f"  保存于 {saved_at}\n\n"
                       f"是否从断点继续？\n"
                       f"（点「取消」将丢弃断点并从头开始）")
                # NiceGUI 对话框
                from nicegui import ui
                _resume_choice = {"val": None}

                async def _on_yes():
                    _resume_choice["val"] = True
                    dlg.close()

                async def _on_no():
                    _resume_choice["val"] = False
                    dlg.close()

                with ui.dialog() as dlg, ui.card().classes('p-4 gap-3'):
                    ui.label('断点续识别').classes('text-lg font-bold')
                    ui.label(msg).classes('text-sm whitespace-pre-line')
                    with ui.row().classes('w-full justify-end gap-2'):
                        ui.button('从头开始', on_click=_on_no).props('ripple').style(
                            'background: var(--bg-elevated); color: var(--text-secondary)')
                        ui.button('继续识别', on_click=_on_yes).props('ripple').style(
                            'background: var(--accent); color: var(--bg-canvas)')
                # persistent：禁止点遮罩/ESC 关闭。对话框被外因关闭后无人写
                # _resume_choice，下方 while 轮询会把检测按钮的 async handler
                # 永久挂死（期间无法再启动任何检测，直到刷新页面）
                dlg.props('persistent')

                def _on_dlg_hide():
                    # 兜底：对话框以任何其他方式关闭且未做选择时按"从头开始"处理，
                    # 保证 while 轮询必然退出
                    if _resume_choice["val"] is None:
                        _resume_choice["val"] = False
                dlg.on_hide(_on_dlg_hide)
                dlg.open()
                # 等待用户选择
                while _resume_choice["val"] is None:
                    await asyncio.sleep(0.1)
                if not _resume_choice["val"]:
                    # 用户选择从头开始：删除该视频所有 checkpoint
                    state.delete_checkpoint(vp)

        token = _try_acquire('detect')
        if not token:
            return
        _cards_video["path"] = None  # 单视频检测结果显示在全局模式
        state.cancel_event.clear()
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
                # 主行短状态 + 详情行完整消息（含帧数/ETA），避免两行重复显示同一内容
                progress_text.set_text('检测中...' if pct < 80 else '生成预览片段...')
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
                skip_yolo_no_motion=skip_yolo_switch.value,
                task_token=token)
        except Exception as _e:
            import traceback
            status = f"❌ 检测异常: {_e}\n{traceback.format_exc()}"
            ok = False
            state.release_task(token)  # io_bound 未启动/启动即异常时后台 finally 不会执行
        # 注意：正常路径锁由 run_detect 内部 finally 释放（锁归任务本体）
        # 隐藏进度条，显示预览图（无论成功/失败/取消，都回到一致的 preview 态，避免视频重叠）
        _show_right_pane('preview')
        if not ok and state.cancel_event.is_set():
            _set_status(status, 'info')  # 用户取消属中性提示，不用红色
        else:
            _set_status(status, 'ok' if ok else 'err')
        detect_btn.set_text('开始识别')
        detect_btn.enable()
        # 无论成功/失败都刷新卡片：失败路径 run_detect 已清空 state，
        # 不刷新会残留上一个视频的卡片（点击预览静默无效）
        _refresh_result_cards()

    # ===== 人物分类对话框（页面级创建一次；内容每次打开时重建） =====
    with ui.dialog() as person_dlg, ui.card().classes('p-4 gap-3 w-96'):
        person_dlg_body = ui.column().classes('w-full gap-2')
    # persistent：禁止点遮罩/ESC 关闭（与断点对话框同一防御口径）
    person_dlg.props('persistent')
    _person_dlg_idx = {"idx": None}

    def _person_apply(name: str):
        """应用人物分类：写 clip + 持久化历史 labels.persons，刷新卡片。"""
        idx = _person_dlg_idx["idx"]
        _person_dlg_idx["idx"] = None
        person_dlg.close()
        if idx is None:
            return
        # 全局模式下有任务运行时拒绝（与 √/× 标记同一规则：避免与检测线程竞态）
        if _cards_video["path"] is None and _refuse_if_busy():
            return
        _, status = detection.clip_action(
            "set_person", idx, video_path=_cards_video["path"], person=name)
        _refresh_result_cards()
        _set_status(status, 'info')

    def _on_person_clip(idx):
        """打开人物分类对话框：已有人物快捷选择 + 自定义新人物 + 清除分类。"""
        vp = _cards_video["path"]
        if vp is not None:
            snap = state.batch_results.get(vp)
            clips = snap["clips"] if snap else []
        else:
            clips = state.last_goal_clips
        if idx < 0 or idx >= len(clips):
            return
        cur = clips[idx].get("person") or ''
        # 本视频已用人物（有色 chips）+ 全局名单里的其他场次人物（灰 chips）
        in_video = sorted({c.get("person") for c in clips if c.get("person")})
        if cur and cur not in in_video:
            in_video.append(cur)
        try:
            global_persons = state.load_persons()
        except Exception:
            global_persons = []
        others = [p for p in global_persons if p and p not in in_video]
        # 人物专属色（与卡片徽章/统计行同源映射；仅本视频已用人物有色）
        color_map = _person_color_map(clips)

        body = person_dlg_body
        body.clear()
        _person_dlg_idx["idx"] = idx
        ts = clips[idx]["ts"]
        with body:
            ui.label('👤 人物分类').classes('text-base font-bold')
            ui.label(f'片段 {ts:.1f}s' + (f' · 当前: {cur}' if cur else ' · 未分类')).classes(
                'text-xs').style('color: var(--text-secondary)')
            if in_video:
                ui.label('本视频人物').classes('text-xs').style('color: var(--text-secondary)')
                with ui.row().classes('w-full flex-wrap gap-1'):
                    for p in in_video:
                        _c = color_map.get(p, _PERSON_PALETTE[0])
                        _sel = (p == cur)
                        ui.button(p, on_click=lambda e, name=p: _person_apply(name)).classes(
                            'text-xs rounded-full px-3 py-1 no-caps').props('ripple flat dense').style(
                            f'color: {_c[0]}; '
                            f'border: 1px solid {_c[0]}; '
                            f'background: {_c[1] if _sel else "transparent"}')
            if others:
                ui.label('其他场次人物').classes('text-xs').style('color: var(--text-secondary)')
                with ui.row().classes('w-full flex-wrap gap-1'):
                    for p in others:
                        _sel = (p == cur)
                        ui.button(p, on_click=lambda e, name=p: _person_apply(name)).classes(
                            'text-xs rounded-full px-3 py-1 no-caps').props('ripple flat dense').style(
                            'color: var(--text-secondary); '
                            f'border: 1px solid {"var(--accent)" if _sel else "var(--border-subtle)"}; '
                            f'background: {"var(--accent-muted)" if _sel else "transparent"}')
            with ui.row().classes('w-full items-center gap-2'):
                new_name = ui.input(placeholder='输入新人物名').props('dense outlined dark').classes(
                    'flex-1').style('color: var(--text-primary)')

                def _add_new():
                    name = (new_name.value or '').strip()
                    if name:
                        _person_apply(name)
                ui.button('添加并使用', on_click=_add_new).props('ripple dense').classes(
                    'text-xs no-caps').style('background: var(--accent); color: var(--bg-canvas)')
                new_name.on('keydown.enter', lambda e: _add_new())
            with ui.row().classes('w-full justify-between gap-2 mt-1'):
                ui.button('清除分类' if cur else '不分类',
                          on_click=lambda: _person_apply('')).props('ripple flat dense').classes(
                    'text-xs no-caps').style(
                    'color: var(--text-secondary); border: 1px solid var(--border-subtle)')

                def _cancel_person():
                    _person_dlg_idx["idx"] = None
                    person_dlg.close()
                ui.button('取消', on_click=_cancel_person).props('ripple flat dense').classes(
                    'text-xs').style('color: var(--text-secondary)')
        person_dlg.open()

    def _refresh_result_cards():
        """刷新结果卡片列表。

        数据源：_cards_video 非空 → 批量快照（流水线确认模式）；
                否则 → 全局 last_goal_clips（单视频/批量最后结果）。
        """
        result_container.clear()
        vp = _cards_video["path"]
        if vp is not None:
            snap = state.batch_results.get(vp)
            clips = snap["clips"] if snap else []
        else:
            clips = state.last_goal_clips
        if not clips:
            with result_container:
                if vp is not None:
                    ui.label(f'当前查看: {os.path.basename(vp)}').classes(
                        'text-xs text-center w-full py-4').style('color: var(--accent)')
                    ui.label('该视频暂无片段').classes('text-gray-400 text-xs text-center w-full')
                else:
                    ui.label('暂无进球结果').classes('text-gray-300 text-xs text-center w-full py-4')
                    ui.label('请先加载视频 → 标定篮筐 → 开始识别').classes('text-gray-400 text-xs text-center w-full')
            export_row.classes(add='hidden')  # 无结果时隐藏导出按钮
            _set_func_collapsed(False)  # 列表为空时展开功能区
            return
        export_row.classes(remove='hidden')  # 有结果时显示导出按钮
        _set_func_collapsed(True)  # 进球列表出来后自动折叠顶部功能区，把空间让给列表
        # 顶部统计行：√ / × / 待标 + 人物分类 + 导出说明
        n_keep = sum(1 for c in clips if c.get("mark") == "keep")
        n_reject = sum(1 for c in clips if c.get("mark") == "reject")
        n_pending = len(clips) - n_keep - n_reject
        person_counts = {}
        for c in clips:
            p = c.get("person")
            if p:
                person_counts[p] = person_counts.get(p, 0) + 1
        n_person = sum(person_counts.values())
        # 人物配色映射（徽章 / 统计行 / 对话框同源，同屏不撞色）
        person_colors = _person_color_map(clips)
        if n_keep:
            export_hint = f'导出集锦：{n_keep} 个 √ 片段'
        elif n_reject:
            export_hint = f'导出集锦：{n_pending} 个未标片段（× 已排除）'
        else:
            export_hint = f'导出集锦：全部 {len(clips)} 个（标记 √ 后只导 √）'
        # 人物筛选下拉选项：'' = 全部 / 仅已分类（存在未分类片段时才有意义）/
        # 各人物（带计数）/ 全局名单其他场次人物（无计数，供整场合并导出选择——
        # 单视频模式下同场其他节的人物也来自这里）
        try:
            _opts = {'': '全部人物'}
            if person_counts and n_person < len(clips):
                _opts[detection.PERSON_FILTER_CLASSIFIED] = f'仅已分类（{n_person}）'
            _opts.update({p: f'{p}（{n}）' for p, n in sorted(person_counts.items())})
            for p in state.load_persons():
                if p and p not in _opts:
                    _opts[p] = p
            _cur = hl_person_select.value
            if _cur not in _opts:
                hl_person_select.set_value('')
            hl_person_select.set_options(_opts, value=hl_person_select.value)
        except Exception:
            pass
        with result_container:
            if vp is not None:
                ui.label(f'当前查看: {os.path.basename(vp)}').classes(
                    'text-xs font-bold w-full pb-1').style('color: var(--accent)')
            with ui.row().classes('w-full items-center gap-2 px-1 pb-2 flex-wrap'):
                ui.label(f'√ {n_keep}').classes('text-xs font-bold').style('color: #22c55e')
                ui.label(f'× {n_reject}').classes('text-xs font-bold').style('color: var(--err)')
                ui.label(f'待标 {n_pending}').classes('text-xs').style('color: var(--text-secondary)')
                if person_counts:
                    for p, n in sorted(person_counts.items()):
                        _c = person_colors.get(p, _PERSON_PALETTE[0])
                        ui.label(f'👤 {p}×{n}').classes(
                            'text-xs font-bold rounded-full px-2 py-0.5').style(
                            f'color: {_c[0]}; background: {_c[1]}; border: 1px solid {_c[0]}')
                ui.label(export_hint).classes('text-xs ml-auto').style('color: var(--text-secondary)')
        for i, clip in enumerate(clips):
            ts = clip["ts"]
            # 预览片段为进球时刻 ±PREVIEW_CLIP_HALF_SEC（与 _generate_preview_clips 同一常量，
            # 旧实现两处各写一个 3，改一处必漏另一处）
            half = detection.PREVIEW_CLIP_HALF_SEC
            start_ts = max(0.0, ts - half)
            end_ts = ts + half
            t_min, t_sec = int(start_ts // 60), start_ts % 60
            end_min, end_sec = int(end_ts // 60), end_ts % 60
            mark = clip.get("mark")
            # 卡片视觉状态：√ 绿框 / × 红框半透明
            card_style = 'margin: 0; background: var(--bg-surface); border: 1px solid var(--border-subtle)'
            if mark == 'keep':
                card_style = ('margin: 0; background: var(--bg-surface); '
                              'border: 1px solid rgba(34, 197, 94, 0.6)')
            elif mark == 'reject':
                card_style = ('margin: 0; background: var(--bg-surface); opacity: 0.55; '
                              'border: 1px solid rgba(239, 68, 68, 0.5)')
            with result_container:
                with ui.card().props('flat').classes('result-card w-full rounded-lg px-3 py-2').style(card_style):
                    # 第一行：时间戳徽章（mono + tabular-nums）+ 右上角人物分类徽章（人物专属色）
                    with ui.row().classes('w-full items-center gap-2 mb-1'):
                        ui.label(f'{t_min}:{t_sec:04.1f} - {end_min}:{end_sec:04.1f}').classes(
                            'text-sm font-bold font-mono').style('color: var(--accent)')
                        person = clip.get("person")
                        _pc = person_colors.get(person) if person else None
                        ui.button((f'👤 {person}' if _pc else '👤 分类'),
                                  on_click=lambda e, idx=i: _on_person_clip(idx)).classes(
                            'ml-auto text-xs rounded-full px-3 py-0.5').props('ripple flat dense no-caps').style(
                            f'color: {_pc[0] if _pc else "var(--text-secondary)"}; '
                            f'border: 1px solid {_pc[0] if _pc else "var(--border-subtle)"}; '
                            f'background: {_pc[1] if _pc else "transparent"}; '
                            'max-width: 55%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap')
                    # 第二行：操作按钮（预览 / √ / × / 导出）
                    with ui.row().classes('w-full gap-1'):
                        ui.button('预览', on_click=lambda e, idx=i: _on_preview_clip(idx)).classes(
                            'flex-1 text-xs rounded-lg py-1').props('ripple flat').style('color: var(--text-secondary); border: 1px solid var(--border-subtle)')
                        _keep_on = mark == 'keep'
                        ui.button('√ 确认', on_click=lambda e, idx=i: _on_mark_clip(idx, 'mark_keep')).classes(
                            'flex-1 text-xs rounded-lg py-1 font-bold').props('ripple flat').style(
                            f'color: {"#22c55e" if _keep_on else "var(--text-secondary)"}; '
                            f'border: 1px solid {"rgba(34, 197, 94, 0.7)" if _keep_on else "var(--border-subtle)"}; '
                            f'background: {"rgba(34, 197, 94, 0.12)" if _keep_on else "transparent"}')
                        _rej_on = mark == 'reject'
                        ui.button('× 误报', on_click=lambda e, idx=i: _on_mark_clip(idx, 'mark_reject')).classes(
                            'flex-1 text-xs rounded-lg py-1 font-bold').props('ripple flat').style(
                            f'color: {"var(--err)" if _rej_on else "var(--text-secondary)"}; '
                            f'border: 1px solid {"rgba(239, 68, 68, 0.7)" if _rej_on else "var(--border-subtle)"}; '
                            f'background: {"rgba(239, 68, 68, 0.12)" if _rej_on else "transparent"}')
                        ui.button('导出', on_click=lambda e, idx=i: _on_export_clip(idx)).classes(
                            'flex-1 text-xs rounded-lg py-1').props('ripple flat').style('color: var(--text-secondary); border: 1px solid var(--border-subtle)')

    def _on_preview_clip(idx):
        # 全局模式下有任务运行时拒绝：检测线程可能正在 clear/extend clips，
        # 与 clip_action 内 len 检查→取值之间竞态（快照模式只读快照，安全）
        if _cards_video["path"] is None and _refuse_if_busy():
            return
        path, status = detection.clip_action("preview", idx, video_path=_cards_video["path"])
        if path and os.path.exists(path):
            result_video_el.set_source(path)
            _show_right_pane('result')
            # 自动播放
            result_video_el.run_method('play')
        _set_status(status, 'info')

    def _on_export_clip(idx):
        if _cards_video["path"] is None and _refuse_if_busy():
            return
        path, status = detection.clip_action("export", idx, video_path=_cards_video["path"])
        if path and os.path.exists(path):
            ui.download(path)
        _set_status(status, 'ok' if path and os.path.exists(path) else 'err')

    def _on_mark_clip(idx, action):
        # 快照模式随时可标（只改快照 dict）；全局模式有任务运行时拒绝（会与检测线程冲突）
        if _cards_video["path"] is None and _refuse_if_busy():
            return
        _, status = detection.clip_action(action, idx, video_path=_cards_video["path"])
        _refresh_result_cards()
        _set_status(status, 'info')

    async def _on_highlights():
        vp = _cards_video["path"]
        # ===== 流水线分支：批量检测运行中，对快照视频生成集锦 =====
        # hl_busy 小锁与取消事件都归 detection.generate_highlights 管理
        # （不占全局任务锁，state 模块级跨页面连接共享，NVENC 与 CUDA 可并行）
        if vp is not None and state.current_task() == 'batch':
            if state.hl_busy["on"]:
                # 生成中再次点击 → 请求取消（与检测/批量按钮的交互一致）
                state.hl_cancel_event.set()
                hl_mini_text.set_text('正在取消集锦...')
                return
            from nicegui import run
            state.hl_cancel_event.clear()
            # 显示集锦迷你进度条（不占右侧进度面板，那边正显示批量进度）
            hl_progress_strip.classes(remove='hidden')
            hl_mini_bar.set_value(0)
            hl_mini_text.set_text(f'集锦: {os.path.basename(vp)}')

            def _hl_progress(pct, msg):
                try:
                    hl_mini_bar.set_value(pct / 100)
                    hl_mini_text.set_text(f'集锦 {pct:.0f}%')
                except Exception:
                    pass

            _set_status(f'正在为 {os.path.basename(vp)} 生成集锦（后台检测不受影响）...', 'busy')
            try:
                path, status = await run.io_bound(
                    detection.generate_highlights, hl_pre_roll.value, hl_post_roll.value,
                    hl_min_gap.value, _hl_progress, vp,
                    person_filter=(hl_person_select.value or None))
            except Exception as _e:
                import traceback
                path, status = None, f"❌ 集锦生成异常: {_e}\n{traceback.format_exc()}"
            finally:
                # hl_busy 由 generate_highlights 内部 finally 释放（锁归任务本体：
                # 页面刷新取消 UI 协程时后台线程仍在跑，UI 提前重置会让新页面
                # 对同一视频再次启动集锦，两线程写同一输出文件）；UI 只收迷你条
                hl_progress_strip.classes(add='hidden')
            if path and os.path.exists(path):
                ui.download(path)
            _set_status(status, 'ok' if path and os.path.exists(path) else 'err')
            return
        token = _try_acquire('highlights')
        if not token:
            return
        # 非流水线集锦使用 cancel_event 作为剪辑取消检查；cancel_event 只在
        # detect/batch 启动处 clear。若用户之前取消过检测/批量，flag 残留会让
        # 本次集锦"第一段即判取消"（幽灵取消）。已独占任务锁、无并发任务，
        # 且非流水线导出本就没有取消 UI，此处 clear 不损失任何功能。
        state.cancel_event.clear()
        from nicegui import run
        # 在右侧预览区显示进度
        _show_right_pane('progress')
        progress_bar.set_value(0)
        progress_text.set_text('正在生成集锦...')
        progress_detail.set_text('')

        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                # 主行短状态 + 详情行完整消息，避免两行重复
                progress_text.set_text('正在生成集锦...')
                progress_detail.set_text(msg)
            except Exception:
                pass

        _set_status('正在生成集锦...', 'busy')
        try:
            path, status = await run.io_bound(
                detection.generate_highlights, hl_pre_roll.value, hl_post_roll.value,
                hl_min_gap.value, _progress_callback, vp, task_token=token,
                person_filter=(hl_person_select.value or None))
        except Exception as _e:
            import traceback
            path, status = None, f"❌ 集锦生成异常: {_e}\n{traceback.format_exc()}"
            state.release_task(token)  # io_bound 未启动/启动即异常时后台 finally 不会执行
        # 正常路径锁由 generate_highlights 内部 finally 释放（锁归任务本体）

        if path and os.path.exists(path):
            ui.download(path)
            highlights_video_el.set_source(path)
            _show_right_pane('highlights')
        else:
            _show_right_pane('preview')
        _set_status(status, 'ok' if path and os.path.exists(path) else 'err')

    async def _on_fullgame_highlights():
        """整场集锦：合并同场全部视频（四节），按当前人物筛选导出。

        批量模式 = 扫描过的文件夹；单视频模式 = 自动发现当前视频所在文件夹
        的同场视频（detection.generate_highlights_fullgame 内回退）。
        """
        if not state.batch_files and not state.video_state.get("path"):
            _set_status('❌ 整场导出需要视频：请先加载视频或扫描文件夹', 'err')
            return
        # 流水线分支：批量检测运行中（与单视频流水线集锦共用 hl_busy 小锁）
        if state.current_task() == 'batch':
            if state.hl_busy["on"]:
                state.hl_cancel_event.set()
                hl_mini_text.set_text('正在取消集锦...')
                return
            from nicegui import run
            state.hl_cancel_event.clear()
            hl_progress_strip.classes(remove='hidden')
            hl_mini_bar.set_value(0)
            hl_mini_text.set_text('整场集锦生成中')

            def _hl_progress(pct, msg):
                try:
                    hl_mini_bar.set_value(pct / 100)
                    hl_mini_text.set_text(f'整场集锦 {pct:.0f}%')
                except Exception:
                    pass

            _set_status('正在生成整场集锦（后台检测不受影响）...', 'busy')
            try:
                path, status = await run.io_bound(
                    detection.generate_highlights_fullgame,
                    (hl_person_select.value or None),
                    hl_pre_roll.value, hl_post_roll.value, hl_min_gap.value,
                    _hl_progress)
            except Exception as _e:
                import traceback
                path, status = None, f"❌ 整场集锦生成异常: {_e}\n{traceback.format_exc()}"
            finally:
                hl_progress_strip.classes(add='hidden')
            if path and os.path.exists(path):
                ui.download(path)
            _set_status(status, 'ok' if path and os.path.exists(path) else 'err')
            return
        token = _try_acquire('highlights')
        if not token:
            return
        # 同 _on_highlights：非流水线整场集锦以 cancel_event 为取消检查，
        # 残留的取消 flag 会造成幽灵取消（第一段即失败）；独占锁后 clear
        state.cancel_event.clear()
        from nicegui import run
        _show_right_pane('progress')
        progress_bar.set_value(0)
        progress_text.set_text('正在生成整场集锦...')
        progress_detail.set_text('')

        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                progress_text.set_text('正在生成整场集锦...')
                progress_detail.set_text(msg)
            except Exception:
                pass

        _set_status('正在生成整场集锦...', 'busy')
        try:
            path, status = await run.io_bound(
                detection.generate_highlights_fullgame,
                (hl_person_select.value or None),
                hl_pre_roll.value, hl_post_roll.value, hl_min_gap.value,
                _progress_callback, token)
        except Exception as _e:
            import traceback
            path, status = None, f"❌ 整场集锦生成异常: {_e}\n{traceback.format_exc()}"
            state.release_task(token)  # io_bound 未启动/启动即异常时后台 finally 不会执行

        if path and os.path.exists(path):
            ui.download(path)
            highlights_video_el.set_source(path)
            _show_right_pane('highlights')
        else:
            _show_right_pane('preview')
        _set_status(status, 'ok' if path and os.path.exists(path) else 'err')

    # 历史记录选中项（按视频路径记录，而非列表索引：
    # 新检测会插入/覆盖记录使索引整体移动，旧索引会指向错误的行）
    _selected_history = {"video": None}

    async def _refresh_history():
        history_list.clear()
        try:
            from nicegui import run
            records = await run.io_bound(state.load_history)
        except OSError as e:
            with history_list:
                ui.label(f'历史记录暂时无法读取（{e}）').classes('text-xs').style('color: var(--err)')
            return
        if not records:
            with history_list:
                ui.label('暂无历史记录').classes('text-gray-400 text-xs')
            return
        for i, r in enumerate(records):
            video = r.get("video", "")
            name = os.path.basename(video) if video else "未知"
            goals = len(r.get("goals", []))
            selected = (_selected_history["video"] is not None
                        and video == _selected_history["video"])
            row_cls = 'history-row selected' if selected else 'history-row'
            with history_list:
                def _make_click(rec_video):
                    async def _on_click():
                        _selected_history["video"] = rec_video
                        await _refresh_history()
                        await _on_load_history(rec_video)
                        exp_hist.set_value(False)  # 加载完成后自动收起历史面板
                    return _on_click
                with ui.row().classes(f'w-full items-center gap-1 p-1 rounded cursor-pointer border {row_cls}').on('click', _make_click(video)):
                    ui.label(f'{i+1}.').classes('text-xs font-bold font-mono').style('color: var(--accent)')
                    ui.label(name).classes('text-xs flex-1 truncate').style('color: var(--text-primary)')
                    ui.label(f'{goals}球').classes('text-xs font-mono').style('color: var(--text-secondary)')

    async def _on_load_history(rec_video=None):
        """按视频路径加载历史记录（非索引：新检测插入会使索引整体位移，点旧行会加载错记录）。"""
        token = _try_acquire('load')
        if not token:
            return
        try:
            records = state.load_history()
        except OSError as e:
            state.release_task(token)  # 后台未启动，UI 侧释放
            _set_status(f'历史记录暂时无法读取（{e}），请稍后重试', 'err')
            return
        rec = next((r for r in records if r.get("video") == rec_video), None)
        if rec is None:
            state.release_task(token)  # 后台未启动，UI 侧释放
            _set_status('历史记录不存在（可能已被覆盖），请刷新列表', 'err')
            await _refresh_history()
            return

        # 显示进度
        _show_right_pane('progress')
        progress_bar.set_value(0)
        progress_text.set_text('正在加载历史记录...')
        progress_detail.set_text('')

        def _progress_callback(pct, msg):
            try:
                progress_bar.set_value(pct / 100)
                # 主行短状态 + 详情行完整消息，避免两行重复
                progress_text.set_text('正在加载历史记录...')
                progress_detail.set_text(msg)
            except Exception:
                pass

        from nicegui import run
        try:
            result = await run.io_bound(detection.on_load_history,
                                         records.index(rec), _progress_callback,
                                         task_token=token)
            frame, info, status = result
        except Exception as _e:
            import traceback
            frame, info, status = None, "", f"❌ 加载历史异常: {_e}\n{traceback.format_exc()}"
            state.release_task(token)  # io_bound 未启动/启动即异常时后台 finally 不会执行
        # 正常路径锁由 on_load_history 内部 finally 释放（锁归任务本体）

        _show_right_pane('preview')
        if frame is not None:
            preview_image.set_source(video_utils.frame_to_base64(frame))
            _sync_frame_selector()
        # 同步路径输入框，显示当前加载的视频
        if state.video_state["path"]:
            path_input.set_value(state.video_state["path"])
            _selected_history["video"] = state.video_state["path"]
        # 加载历史 = 单视频模式：隐藏批量面板（state 层已清 batch 三件套），
        # 并复位快照查看指针——否则此前批量快照查看残留的 _cards_video["path"]
        # 会让列表/√×/导出仍作用在旧快照视频上，历史记录看似加载实则操作错对象
        batch_panel.classes(add='hidden')
        _cards_video["path"] = None
        info_text.set_text(info)
        _set_status(status, 'ok' if frame is not None else 'err')
        _refresh_result_cards()

    # 页面初始化：不自动加载历史列表（空白初始状态），点「刷新」才加载
    _refresh_result_cards()


# ============ 启动 ============

if __name__ == "__main__":
    # 日志落盘（控制台 + cache/logs/app.log 按日轮转）：
    # 不再依赖 start 脚本 tee，直接 python demo_nicegui.py 启动也有日志文件
    # （输出目录由 cutter/state 各自按需 makedirs，无需在此预创建）
    state.setup_logging()

    # YOLO/CUDA 自检已在模块加载时完成（见 _yolo_selfcheck），
    # 失败/降级时页面顶部横幅提示。

    # 端口可用 BBALL_PORT 环境变量覆盖（start.sh / start.bat 同源读取）
    ui.run(host="127.0.0.1", port=int(os.environ.get("BBALL_PORT", "7871")),
           title="进球集锦助手", dark=True, reload=False)
