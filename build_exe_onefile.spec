# -*- mode: python ; coding: utf-8 -*-
"""篮球进球集锦助手 - PyInstaller 单文件打包配置。

用法:
    env/Scripts/python.exe -m PyInstaller build_exe_onefile.spec --noconfirm

产物: dist/basketball-clipper.exe  (单文件，约 4GB)
特点: 双击即可运行，无需复制整个文件夹
代价: 每次启动需解压到临时目录（约 30-60 秒），之后正常运行
"""
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve()

# ---- 需收集的包 ----
collect_all_packages = [
    "nicegui",
    "imageio_ffmpeg",
    "ultralytics",
    "av",
]

# ---- 额外数据文件 ----
datas = []
wdir = ROOT / "weights"
if wdir.exists():
    for pt in wdir.glob("*.pt"):
        datas.append((str(pt), "weights"))

# ---- 隐式导入 ----
hiddenimports = [
    "torchvision", "torchvision.ops", "torchvision.models",
    "ultralytics.models.yolo.detect", "ultralytics.trackers", "ultralytics.utils",
    "nicegui.components", "nicegui.elements",
    "cv2", "scipy", "sklearn",
    "services", "services.state", "services.detection", "services.video_utils",
    "cutter", "cutter.ffmpeg_cutter",
    "app", "tracker", "video_io",
]

# ---- 排除项 ----
excludes = [
    "matplotlib", "tensorflow", "keras",
    "IPython", "jupyter", "notebook",
    "pytest", "sphinx", "numpydoc",
    "test", "tests", "tkinter",
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

from PyInstaller.utils.hooks import collect_all

for pkg in collect_all_packages:
    try:
        collected = collect_all(pkg)
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

a.hiddenimports = list(dict.fromkeys(a.hiddenimports))


def _normalize_toc(toc_list):
    """确保 TOC 条目都是 3 元组 (dest, src, typecode)。

    PyInstaller 6.x 某些 hook/collect 返回 2 元组 (dest, src)，
    传给 EXE 时会触发 normalize_toc 解包失败。
    """
    fixed = []
    for item in toc_list:
        if isinstance(item, (list, tuple)):
            if len(item) == 3:
                fixed.append(tuple(item))
            elif len(item) == 2:
                # 根据扩展名/来源推断 typecode；二进制给 BINARY，其余 DATA
                dest, src = item
                typecode = "BINARY" if str(src).lower().endswith((".dll", ".pyd", ".so", ".dylib")) else "DATA"
                fixed.append((dest, src, typecode))
    return fixed


a.datas = _normalize_toc(list(a.datas))
a.binaries = _normalize_toc(list(a.binaries))

pyz = PYZ(a.pure)

# onefile 模式：把 binaries + datas 全部塞进 EXE，不需要 COLLECT
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,          # 关键：包含所有动态库
    a.zipfiles,
    a.datas,             # 关键：包含所有数据文件（权重、ffmpeg、nicegui 资源）
    [],
    name="basketball-clipper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
