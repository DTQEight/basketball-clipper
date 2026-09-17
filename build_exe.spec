# -*- mode: python ; coding: utf-8 -*-
"""篮球进球集锦助手 - PyInstaller 打包配置。

用法:
    env/Scripts/python.exe -m PyInstaller build_exe.spec --noconfirm

产物: dist/basketball-clipper/  (目录模式，内含 basketball-clipper.exe)
体积: ~4.5 GB (含 torch+CUDA 12.1 + ffmpeg + YOLO 权重)
"""
import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve()
if sys.platform == "win32":
    SEP = ";"
else:
    SEP = ":"

# ---- 需收集的包（--collect-all 等价物）----
collect_all_packages = [
    "nicegui",            # Quasar/Vue 静态资源
    "imageio_ffmpeg",     # 自带 ffmpeg.exe
    "ultralytics",        # YOLO + tracker YAML
    "av",                 # PyAV 自带的 ffmpeg 共享库
]

# ---- 额外数据文件 ----
datas = [
    # YOLO 球检测权重
    (str(ROOT / "weights" / "basketball_ft.pt"), "weights"),
]
# 若存在其他权重一并打包
wdir = ROOT / "weights"
if wdir.exists():
    for pt in wdir.glob("*.pt"):
        datas.append((str(pt), "weights"))

# ---- 隐式导入（PyInstaller 静态分析可能漏的）----
hiddenimports = [
    # torch/ultralytics 动态导入
    "torchvision",
    "torchvision.ops",
    "torchvision.models",
    "ultralytics.models.yolo.detect",
    "ultralytics.trackers",
    "ultralytics.utils",
    # nicegui 内部
    "nicegui.components",
    "nicegui.elements",
    # opencv
    "cv2",
    # 数据/科学计算
    "scipy",
    "sklearn",
    # 自建模块
    "services",
    "services.state",
    "services.detection",
    "services.video_utils",
    "cutter",
    "cutter.ffmpeg_cutter",
    "app",
    "tracker",
    "video_io",
]

# ---- 排除项（减小体积）----
excludes = [
    "matplotlib",          # nicegui 可能拉进来，但本项目不用
    "tensorflow",
    "keras",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "sphinx",
    "numpydoc",
    "test",
    "tests",
    "tkinter",
]

a = Analysis(
    ["demo_nicegui.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

# collect-all：对每个包收集 data + binaries + submodules
# 注意：PyInstaller 6.x 的 collect_all 返回的 datas/binaries 已是 3 元组 (dest, src, typecode)
# 直接用 collect_data_files/collect_dynamic_libs 会返回 2 元组，导致 COLLECT 阶段 unpack 失败
from PyInstaller.utils.hooks import collect_all

for pkg in collect_all_packages:
    try:
        collected = collect_all(pkg)
        # PyInstaller 不同版本 collect_all 返回 dict 或 list，兼容两种
        if isinstance(collected, dict):
            a.datas += collected.get("datas", [])
            a.binaries += collected.get("binaries", [])
            a.hiddenimports += collected.get("hiddenimports", [])
        elif isinstance(collected, (list, tuple)) and len(collected) == 3:
            a.datas += collected[0] or []
            a.binaries += collected[1] or []
            a.hiddenimports += collected[2] or []
    except Exception as e:
        print(f"[warn] collect {pkg}: {e}")

# 去重
a.hiddenimports = list(dict.fromkeys(a.hiddenimports))

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="basketball-clipper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # torch 不支持 UPX 压缩，且会拖慢构建
    console=True,        # 保留控制台：便于查看日志/报错
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="basketball-clipper",
)
