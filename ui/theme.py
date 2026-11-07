"""页面主题与布局样式（Dark Tech Theme）。

从 demo_nicegui.py 抽出：主题变量、Quasar 覆盖、布局栅格、响应式规则。
字体不再依赖 Google Fonts 外链（P1-7）：优先使用系统已安装的 Geist / 等距字体，
没有则回退到平台原生 UI 字体栈，离线/内网环境也不会因外链失败而推迟首屏。
"""

HEAD_HTML = '''
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
  --text-secondary: #A3A3A3; /* zinc-400 */
  --text-tertiary:  #525252; /* zinc-600 */
  /* 单一 accent: Electric Cyan */
  --accent:          #22D3EE;
  --accent-hover:    #67E8F9;
  --accent-muted:    rgba(34, 211, 238, 0.12);
  /* 语义色（仅状态用，不破坏单 accent 锁定） */
  --ok:    #22C55E;
  --err:   #EF4444;
  --busy:  #22D3EE;
  /* 字体栈：本地优先，不外链（P1-7） */
  --font-sans: 'Geist', 'Segoe UI Variable Text', 'Segoe UI', -apple-system,
               BlinkMacSystemFont, 'PingFang SC', 'Microsoft YaHei', sans-serif;
  --font-mono: 'Geist Mono', 'Cascadia Mono', 'SF Mono', Consolas, Menlo, monospace;
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
.q-field--outlined.q-field--focused .q-field__control {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.15);
}
.q-field--outlined.q-field--focused .q-field__label { color: var(--accent) !important; }

/* ============ 按钮交互态 ============ */
.q-btn { transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1); }
.q-btn:active { transform: scale(0.98); }
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
.history-row { transition: all 0.15s ease; }
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
.nicegui-content, .q-page, .q-layout { position: relative; z-index: 1; }

/* 减少动效偏好 */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}

/* ====== 布局（顶栏 56 / 三栏 / 底栏 46 / 参数抽屉）====== */
.app-shell { position: fixed; inset: 0; display: flex; flex-direction: column; width: 100%; height: 100dvh; overflow: hidden; background: var(--bg-canvas); z-index: 0; }
.app-shell .q-page, .app-shell > .nicegui-content { padding: 0 !important; }
.app-topbar { flex: 0 0 56px; height: 56px; display: flex; align-items: center; gap: 12px; padding: 0 16px; background: var(--bg-surface); border-bottom: 1px solid var(--border-subtle); }
.app-workspace { flex: 1; min-height: 0; display: grid !important; align-items: stretch !important; grid-template-columns: 322px minmax(420px, 1fr) 292px; grid-template-rows: minmax(0, 1fr); }
.app-pane { min-width: 0; min-height: 0; height: 100%; display: flex; flex-direction: column; overflow: hidden; background: var(--bg-canvas); }
.app-pane.center { background: #0d1117; border-left: 1px solid var(--border-subtle); border-right: 1px solid var(--border-subtle); }
.app-pane-head { flex: 0 0 auto; padding: 12px 14px 10px; border-bottom: 1px solid var(--border-subtle); background: var(--bg-surface); }
.clip-row { position: relative; transition: background 0.12s ease; }
.clip-row:hover { background: var(--bg-elevated) !important; }
.clip-row .row-hit { position: absolute; inset: 0; z-index: 1; cursor: pointer; border-radius: 8px; }
.clip-row .row-actions { position: relative; z-index: 2; }
.clip-play { position: absolute; inset: 0; display: grid; place-items: center; font-size: 13px;
  color: #fff; background: rgba(0, 0, 0, 0.38); opacity: 0; transition: opacity 0.12s; pointer-events: none; }
.clip-row:hover .clip-play { opacity: 1; }
.app-footer { flex: 0 0 46px; height: 46px; display: flex; align-items: center; gap: 10px; padding: 0 16px; background: var(--bg-surface); border-top: 1px solid var(--border-subtle); }
.params-drawer { position: fixed; right: 0; top: 56px; bottom: 46px; width: 344px; background: var(--bg-surface); border-left: 1px solid var(--border-strong); transform: translateX(102%); transition: transform 0.22s ease; display: flex; flex-direction: column; z-index: 20; box-shadow: -20px 0 40px rgba(0, 0, 0, 0.4); }
.params-drawer.open { transform: translateX(0); }
.cfg-group { font-size: 11px; color: var(--text-tertiary); letter-spacing: 0.06em; text-transform: uppercase; margin-top: 4px; }
.cfg-group::after { content: ""; display: block; height: 1px; margin-top: 5px; background: var(--border-subtle); }
.cfg-hint { font-size: 10.5px; color: var(--text-tertiary); }
.insp-card { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 11px 12px; }
.meta-pill { font-size: 11.5px; color: var(--text-secondary); padding: 2px 8px; border-radius: 999px; background: var(--bg-elevated); border: 1px solid var(--border-subtle); }
.pending-pill { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; padding: 5px 11px; border-radius: 999px; background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.35); color: #f59e0b; }
.tab-bar { flex: 0 0 42px; height: 42px; display: flex; align-items: center; gap: 2px; padding: 0 12px; background: var(--bg-surface); border-bottom: 1px solid var(--border-subtle); }
.bar-track { height: 5px; border-radius: 3px; background: var(--bg-elevated); overflow: hidden; width: 100%; }
.bar-fill { display: block; height: 100%; border-radius: 3px; }
.kv-row { display: flex; justify-content: space-between; font-size: 12px; padding: 4px 0; color: var(--text-secondary); border-bottom: 1px dashed rgba(255,255,255,0.06); }
.kv-row:last-child { border-bottom: none; }
.kv-row b { color: var(--text-primary); font-weight: 500; font-variant-numeric: tabular-nums; }

/* ====== 统一空态/错误态（P2-9） ====== */
.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 8px; padding: 26px 16px; text-align: center; color: var(--text-tertiary); }
.empty-state .empty-title { font-size: 12.5px; color: var(--text-secondary); font-weight: 600; }
.empty-state .empty-hint { font-size: 11px; line-height: 1.5; max-width: 240px; }
.empty-state.error .empty-title { color: var(--err); }

/* 错误横幅：错误统一走顶栏下方横幅（P1-6），状态栏只留简短进度 */
.app-banner { display: none; align-items: center; gap: 8px; padding: 8px 16px;
  background: rgba(239, 68, 68, 0.10); border-bottom: 1px solid rgba(239, 68, 68, 0.45); }
.app-banner.show { display: flex; }
.app-banner .banner-text { flex: 1; font-size: 12px; color: #FCA5A5; min-width: 0; }
.app-banner .banner-close { color: #FCA5A5; }

/* 窄屏检查器替代入口（P2-8）：默认隐藏，窄屏下把检查器变成滑入抽屉 */
.inspector-toggle { display: none !important; }

@media (max-width: 1180px) {
  .app-workspace { grid-template-columns: 290px minmax(360px, 1fr) 264px; }
}
@media (max-width: 1100px) {
  .app-workspace { grid-template-columns: 300px minmax(0, 1fr); }
  .app-pane.inspector {
    position: fixed; right: 0; top: 56px; bottom: 46px; width: 300px;
    transform: translateX(103%); transition: transform 0.22s ease; z-index: 18;
    border-left: 1px solid var(--border-strong); box-shadow: -16px 0 32px rgba(0, 0, 0, 0.45);
  }
  .app-pane.inspector.open { transform: translateX(0); }
  .app-topbar .meta-pill { display: none; }
  .inspector-toggle { display: inline-flex !important; }
}
@media (max-width: 860px) {
  .app-topbar { gap: 8px; padding: 0 10px; }
  .app-topbar .file-sub { display: none; }
  .app-footer { gap: 6px; padding: 0 10px; }
}
</style>
'''
