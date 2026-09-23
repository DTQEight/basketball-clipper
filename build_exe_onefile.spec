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
# 只打运行时真正加载的球检测权重。temporal_ft.pt / temporal_simclr.pt 是旧一代
# 时序模型的留档，生产代码里没有任何引用（只有 training/ 脚本引用同名的
# oof_*.jsonl 明细），打进去纯让 exe 白胖 86MB。
_WEIGHTS_SKIP = {"temporal_ft.pt", "temporal_simclr.pt"}
if wdir.exists():
    for pt in wdir.glob("*.pt"):
        if pt.name in _WEIGHTS_SKIP:
            continue
        datas.append((str(pt), "weights"))

# 四臂 AI 复核的模型与配置：goal_verifier 用 Path(__file__).parent.parent/"training"
# 定位（onefile 下即 _MEIPASS/training）。不打包进来的话四臂会静默加载失败——
# exe 版就只剩「检测+剪辑」，没有 AI 自动 √/×（这是上一版 exe 的实际情况）。
# 只打包运行必需的这几个文件（约 90MB）；frames_b / flow_b / features.jsonl 等
# 几 GB 的训练中间产物不打（运行不需要）。
# 注意后半段是**脚本**而不是权重：goal_verifier 用 importlib 按文件路径
# （_MEIPASS/training/xxx.py）加载它们，缺一个就整条臂加载失败——只打模型文件
# 不够。A 臂靠 extract_features，B/Flow 靠 train_temporal + extract_frames_b
# （train_temporal 顶层 `from training.extract_frames_b import ...`），
# VM 臂靠 extract_videomae。这几个文件合计约 55KB。
tdir = ROOT / "training"
if tdir.exists():
    for _name in ("model_lgbm.txt", "model_b_simclr.pt", "model_flow_simclr.pt",
                  "model_vm_lgbm.txt", "model_temporal_meta.json",
                  "model_meta.json", "model_b_simclr_meta.json",
                  "model_flow_simclr_meta.json",
                  "extract_features.py", "extract_frames_b.py",
                  "train_temporal.py", "extract_videomae.py"):
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

# ---- 体积优化：剔除推理确实不用的 CUDA 库 ----
# ⚠ cudnn_adv64_9.dll **不能删**（311MB）。四臂里的 B/Flow 臂用 torch.nn.GRU
#   （bigru 头），cuDNN 的 RNN 实现落在 cudnn_adv 里；缺它时 exe 会在加载 B/Flow
#   臂的瞬间硬崩（实机日志：Could not locate cudnn_adv64_9.dll /
#   Invalid handle. Cannot load symbol cudnnCreateRNNDescriptor，进程以
#   0xC0000409 退出，try/except 拦不住）。旧版"实测可删"只在 YOLO（纯卷积）
#   路径下成立——那时 exe 里根本没有四臂。
_EXCLUDE_DLLS = {
    "cusolverMg64_11.dll",                # 145MB 多 GPU 求解器
}
# 注：其余 CUDA 库经本轮实测**都不可删**（torch 的加载链相互依赖，缺一即
# WinError 126 或 CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED）：
#   cublas64_12    ← 依赖 cublasLt64_12
#   cusolver64_11  ← 依赖 cusparse64_12
#   caffe2_nvrtc   ← 依赖 nvrtc64_120_0
#   cudnn 的卷积路径 ← 依赖 cudnn_engines_precompiled64_9（504MB）
# 逐个改名实测均失败，故保持原样（详见上轮打包记录）。
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
