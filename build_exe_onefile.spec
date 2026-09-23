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

# 四臂 AI 复核的模型与配置：goal_verifier 用 Path(__file__).parent.parent/"training"
# 定位（onefile 下即 _MEIPASS/training）。不打包进来的话四臂会静默加载失败——
# exe 版就只剩「检测+剪辑」，没有 AI 自动 √/×（这是上一版 exe 的实际情况）。
# 只打包运行必需的这几个文件（约 90MB）；frames_b / flow_b / features.jsonl 等
# 几 GB 的训练中间产物不打（运行不需要）。
tdir = ROOT / "training"
if tdir.exists():
    for _name in ("model_lgbm.txt", "model_b_simclr.pt", "model_flow_simclr.pt",
                  "model_vm_lgbm.txt", "model_temporal_meta.json",
                  "model_meta.json", "model_b_simclr_meta.json",
                  "model_flow_simclr_meta.json"):
        _f = tdir / _name
        if _f.exists():
            datas.append((str(_f), "training"))

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
    "polars",                           # 178MB，未直接使用
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

# ---- 体积优化：剔除 YOLO 推理不用的 CUDA 库 ----
# 经测试可安全删除（不影响 torch 加载和 YOLO CNN 推理）：
_EXCLUDE_DLLS = {
    "cudnn_adv64_9.dll",                  # 230MB cuDNN 高级算子(RNN/attention)
    "cusolverMg64_11.dll",                # 73MB  多 GPU 求解器
}
_before = len(a.binaries)
a.binaries = [
    b for b in a.binaries
    if not any(excl in str(b[0]) for excl in _EXCLUDE_DLLS)
]
_freed = sum(
    (Path(b[1]).stat().st_size if len(b) >= 2 and Path(b[1]).exists() else 0)
    for b in a.binaries
)
print(f"[trim] 剔除 {_before - len(a.binaries)} 个 CUDA DLL")

pyz = PYZ(a.pure)

# UPX 压缩：压缩非 torch 的 DLL/EXE，torch CUDA 库不压缩（压缩后会损坏）
_upx = r"C:\Users\desktop\AppData\Local\Microsoft\WinGet\Packages\UPX.UPX_Microsoft.Winget.Source_8wekyb3d8bbwe\upx-5.2.1-win64\upx.exe"
_upx_exclude = [
    "torch_cuda", "torch_cpu", "torch_python",
    "cudnn", "cublas", "cufft", "cusparse", "cusolver", "curand",
    "nvrtc", "nvJitLink", "caffe2", "fbgemm",
    "avcodec", "avformat", "avutil", "swscale", "swresample",
]

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
    upx=True,
    upx_exclude=_upx_exclude,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
