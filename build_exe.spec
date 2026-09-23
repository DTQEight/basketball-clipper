# -*- mode: python ; coding: utf-8 -*-
"""篮球进球集锦助手 - PyInstaller 打包配置。

用法:
    env/Scripts/python.exe -m PyInstaller build_exe.spec --noconfirm

产物: dist/basketball-clipper/  (目录模式，内含 basketball-clipper.exe)
体积: ~2.9 GB (含 torch+CUDA 12.1 + ffmpeg + YOLO 权重)

注意: 本 spec 的依赖收集与剔除规则必须与 build_exe_onefile.spec 保持一致，
      否则会出现「单文件版能跑、目录版报 WinError 126」这类问题。
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
datas = []
# YOLO 球检测权重（打包 weights 目录下所有 .pt）
wdir = ROOT / "weights"
if wdir.exists():
    for pt in wdir.glob("*.pt"):
        datas.append((str(pt), "weights"))

# 四臂 AI 复核的模型与配置：goal_verifier 用 Path(__file__).parent.parent/"training"
# 定位。不打包进来的话四臂会静默加载失败——exe 版只剩「检测+剪辑」，没有
# AI 自动 √/×。只打包运行必需的几个文件（约 90MB）；frames_b / flow_b /
# features.jsonl 等几 GB 训练中间产物不打（运行不需要）。
tdir = ROOT / "training"
if tdir.exists():
    for _name in ("model_lgbm.txt", "model_b_simclr.pt", "model_flow_simclr.pt",
                  "model_vm_lgbm.txt", "model_temporal_meta.json",
                  "model_meta.json", "model_b_simclr_meta.json",
                  "model_flow_simclr_meta.json"):
        _f = tdir / _name
        if _f.exists():
            datas.append((str(_f), "training"))

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
    "polars",              # 178MB，未直接使用
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


def _normalize_toc(toc_list):
    """确保 TOC 条目都是 3 元组 (dest, src, typecode)。

    PyInstaller 6.x 某些 hook/collect 返回 2 元组 (dest, src)，
    传给 EXE/COLLECT 时会触发 normalize_toc 解包失败。
    """
    fixed = []
    for item in toc_list:
        if isinstance(item, (list, tuple)):
            if len(item) == 3:
                fixed.append(tuple(item))
            elif len(item) == 2:
                dest, src = item
                typecode = "BINARY" if str(src).lower().endswith(
                    (".dll", ".pyd", ".so", ".dylib")) else "DATA"
                fixed.append((dest, src, typecode))
    return fixed


a.datas = _normalize_toc(list(a.datas))
a.binaries = _normalize_toc(list(a.binaries))

# ---- 体积优化：仅剔除经实测确认不影响 YOLO 推理的 CUDA 库 ----
# 以下两个已由单文件版实测验证可安全删除；其余 cudnn/nvrtc/cufft/cusolver
# 系列存在相互依赖，删除会导致 WinError 126 或 CUDNN_STATUS_NOT_INITIALIZED
_EXCLUDE_DLLS = {
    "cudnn_adv64_9.dll",                  # 230MB cuDNN 高级算子(RNN/attention)
    "cusolverMg64_11.dll",                # 73MB  多 GPU 求解器
}
_before = len(a.binaries)
a.binaries = [
    b for b in a.binaries
    if not any(excl in str(b[0]) for excl in _EXCLUDE_DLLS)
]
print(f"[trim] 剔除 {_before - len(a.binaries)} 个 CUDA DLL")

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
