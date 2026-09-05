"""ffmpeg 剪辑模块：根据进球时间戳切片并拼接成集锦。

- 临时目录 / 输出目录统一走 BBALL_CACHE_ROOT（跨平台）
- subprocess 加 creationflags=0x08000000 绕开 Windows 沙箱限制
- concat 优先 -c copy 流拷贝（各片段由同一命令模板切出，参数天然一致），
  流拷贝失败时回退整体 re-encode 兜底
- GPU 硬编：优先使用 h264_nvenc（NVIDIA GPU 加速），
  名单探测 + 真实编码探测双保险，运行时失败自动整体回退 libx264
"""
import logging
import os
import shutil
import subprocess
import tempfile

# 优先用 imageio-ffmpeg 自带版本（含 libx264 + nvenc），conda 版无 libx264
try:
    import imageio_ffmpeg
    _DEFAULT_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _DEFAULT_FFMPEG = "ffmpeg"

# Windows 沙箱规避标志（CREATE_NO_WINDOW）
_SBOX = 0x08000000 if os.name == "nt" else 0

# 缓存根目录 / NVENC 会话信号量：复用 services.state 的单一实现
# （各自维护一份环境变量回退逻辑会漂移；信号量与预览片段线程池共用驱动配额）
from services.state import CACHE_ROOT as _CACHE_ROOT, nvenc_semaphore


_log = logging.getLogger("cutter")


def _stderr_tail(e, limit=2000):
    """从 CalledProcessError 提取 stderr 尾部（根因定位的关键信息）。"""
    try:
        err = getattr(e, "stderr", None)
        if err:
            text = err.decode("utf-8", "replace") if isinstance(err, bytes) else str(err)
            return text[-limit:].strip()
    except Exception:
        pass
    return ""


def _detect_nvenc(ffmpeg):
    """检测 h264_nvenc 是否真实可用（两级探测）。

    1. 名单探测：-encoders 列表里有该编码器（快，但只证明编译支持）
    2. 运行时探测：用 lavfi 黑帧源真实编码 3 帧 —— 无 N 卡 / 驱动过旧 /
       NVENC 会话超限时此步失败
    旧实现只做第 1 步：-encoders 列表有 ≠ 硬件可用（如 GTX 1650 TU117
    核心无 NVENC 单元，ffmpeg 构建里却有该编码器名），导致每个片段
    全部失败且无回退。
    """
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                           capture_output=True, text=True, creationflags=_SBOX, timeout=10)
        if "h264_nvenc" not in (r.stdout or ""):
            return False
    except Exception:
        return False
    try:
        # 真实编码探测也要持 NVENC 信号量：流水线模式下预览线程池可能已占满
        # 2 路会话配额，无锁探测会 OpenEncodeSession 失败 → 误判"NVENC 不可用"
        # → 整次集锦静默降级软编。探测会话仅 ~0.1s，等锁代价远小于误降级
        with nvenc_semaphore:
            r = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
                 "-frames:v", "3", "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, text=True, creationflags=_SBOX, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def build_encode_args(ffmpeg, quality="hq", use_nvenc=None):
    """构建视频编码参数，优先用 NVENC 硬编，回退 libx264 软编。

    quality: "hq" 高质量（集锦，cq=20）/ "preview" 预览（cq=26）
    use_nvenc: True/False 直接指定编码器，None 则内部探测（只应在单处调用时用）
    返回: (编码器参数列表，含 codec/preset/rc 质量参数)
    （公开 API：services/detection.py 生成预览片段也用同一套参数）
    """
    if use_nvenc is None:
        use_nvenc = _detect_nvenc(ffmpeg)
    if use_nvenc:
        # NVENC 硬编：-preset p4（平衡速度与质量）-rc vbr -cq 控制质量
        # GTX 1650 有 1 个 NVENC 编码器，可大幅加速
        cq = "20" if quality == "hq" else "26"
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                "-cq", cq, "-b:v", "0", "-spatial_aq", "1"]
    else:
        # 软编回退
        crf = "18" if quality == "hq" else "26"
        return ["-c:v", "libx264", "-preset", "fast", "-crf", crf]


def _cleanup_tmp(clip_files, list_path, tmp_dir=None):
    """清理临时切片文件、拼接列表和本次运行的独立临时目录。"""
    for p in clip_files:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.remove(list_path)
    except OSError:
        pass
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def cut_clips(video_path, timestamps, pre_roll: int = 5, post_roll: int = 5,
              min_gap: int = 8, output_path=None, ffmpeg_path: str = "",
              progress_callback=None, cancel_check=None):
    """根据进球时间戳剪辑集锦（GPU 硬编加速）。

    timestamps: 进球时刻列表（秒，浮点）
    min_gap: 两个片段间隔小于此值则合并，避免连续得分重复切。
    output_path: 输出文件路径，None 则默认到 cache/demo_output/{源视频名}-highlights.mp4
    ffmpeg_path: ffmpeg 可执行文件路径，留空则用 imageio-ffmpeg 自带版本。
    progress_callback: 可选进度回调 (pct, msg)，0-100。
    cancel_check: 可选取消检查函数（返回 True 时中止并清理临时文件）。
    返回: 输出文件路径 / None（无进球、取消或全部失败）
    """

    def _report(pct, msg):
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    def _cancelled():
        return bool(cancel_check and cancel_check())

    if not timestamps:
        _log.info("没有进球时间戳，跳过剪辑")
        return None

    ffmpeg = ffmpeg_path or _DEFAULT_FFMPEG
    timestamps = sorted([float(t) for t in timestamps])

    # 合并过近的片段
    segments = []
    for ts in timestamps:
        start = max(0.0, ts - pre_roll)
        end = ts + post_roll
        if segments and start - segments[-1][1] < min_gap:
            segments[-1][1] = end
        else:
            segments.append([start, end])

    # 临时目录：cache/clips 下每次运行独立子目录（tempfile.mkdtemp），
    # 旧实现固定命名 clip_000.mp4，第二个实例/流水线集锦并发时会互相覆盖、误删对方文件
    _clips_root = os.path.join(_CACHE_ROOT, "clips")
    os.makedirs(_clips_root, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="hl-", dir=_clips_root)
    clip_files = []

    # 输出路径默认到 cache/demo_output/，文件名带源视频名避免覆盖
    if output_path is None:
        out_dir = os.path.join(_CACHE_ROOT, "demo_output")
        _vname = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(out_dir, f"{_vname}-highlights.mp4")
    # 显式传入的 output_path（如按人物导出）同样保证目录存在
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        # 检测编码器一次（名单 + 真实编码双探测），避免每个片段重复检测
        use_nvenc = _detect_nvenc(ffmpeg)
        encode_tag = "NVENC 硬编" if use_nvenc else "libx264 软编"
        _log.info(f"[剪辑] 使用 {encode_tag} | 共 {len(segments)} 段")
        _report(10, f'初始化编码器（{len(segments)} 段，{encode_tag}）...')

        # 切片：用 -ss 在输入前做快速 seek，-t 控制时长
        # 集锦保持原画质：NVENC cq=20 / libx264 crf=18，不缩放
        # 容错：单个片段失败不中断整体，跳过该片段继续其余切片；
        #       NVENC 运行时失败（驱动/会话配额/显存）→ 整体切换软编重切该段
        # ⚠️ use_nvenc 复用已探测结果，避免 build_encode_args 内部再开子进程
        encode_args = build_encode_args(ffmpeg, quality="hq", use_nvenc=use_nvenc)
        failed_segments = 0
        for i, (start, end) in enumerate(segments):
            if _cancelled():
                _log.info("[剪辑] 已取消，清理临时文件")
                return None
            _report(15 + 70 * i / len(segments), f'剪切片段 {i+1}/{len(segments)}...')
            clip_path = os.path.join(tmp_dir, f"clip_{i:03d}.mp4")
            duration = end - start

            def _build_cmd(e_args):
                return [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-ss", f"{start:.3f}", "-i", video_path,
                    "-t", f"{duration:.3f}",
                ] + e_args + [
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart",
                    clip_path,
                ]

            try:
                # NVENC 会话配额（消费卡限 2 路）：与预览片段线程池共用信号量排队
                if use_nvenc:
                    with nvenc_semaphore:
                        subprocess.run(_build_cmd(encode_args), check=True,
                                       capture_output=True, text=True,
                                       creationflags=_SBOX, timeout=300)
                else:
                    subprocess.run(_build_cmd(encode_args), check=True,
                                   capture_output=True, text=True,
                                   creationflags=_SBOX, timeout=300)
            except subprocess.CalledProcessError as e:
                if use_nvenc:
                    # 运行时探测漏网的 NVENC 失败（如中途驱动重置/并发会话超限）：
                    # 整体切换软编，重切当前段，后续段全部走软编
                    _log.warning(f"[剪辑] NVENC 切片失败，整体回退 libx264 软编"
                                 f"(start={start:.1f}s): {_stderr_tail(e) or e}")
                    use_nvenc = False
                    encode_tag = "libx264 软编(NVENC 回退)"
                    encode_args = build_encode_args(ffmpeg, quality="hq", use_nvenc=False)
                    try:
                        subprocess.run(_build_cmd(encode_args), check=True,
                                       capture_output=True, text=True,
                                       creationflags=_SBOX, timeout=300)
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e2:
                        failed_segments += 1
                        _log.warning(f"[WARN] 片段 {i+1}/{len(segments)} 软编仍失败，跳过"
                                     f"(start={start:.1f}s): {_stderr_tail(e2) or e2}")
                        continue
                else:
                    failed_segments += 1
                    _log.warning(f"[WARN] 片段 {i+1}/{len(segments)} 切片失败，跳过"
                                 f"(start={start:.1f}s): {_stderr_tail(e) or e}")
                    continue
            except subprocess.TimeoutExpired:
                if use_nvenc:
                    # NVENC 挂起超时同样触发整体软编回退：
                    # 只跳过当前段的话后续每段仍各等满 300s
                    _log.warning(f"[剪辑] NVENC 切片超时，整体回退 libx264 软编"
                                 f"(start={start:.1f}s)")
                    use_nvenc = False
                    encode_tag = "libx264 软编(NVENC 回退)"
                    encode_args = build_encode_args(ffmpeg, quality="hq", use_nvenc=False)
                    try:
                        subprocess.run(_build_cmd(encode_args), check=True,
                                       capture_output=True, text=True,
                                       creationflags=_SBOX, timeout=300)
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e2:
                        failed_segments += 1
                        _log.warning(f"[WARN] 片段 {i+1}/{len(segments)} 软编仍失败，跳过"
                                     f"(start={start:.1f}s): {_stderr_tail(e2) or e2}")
                        continue
                else:
                    failed_segments += 1
                    _log.warning(f"[WARN] 片段 {i+1}/{len(segments)} 切片超时（>300s），跳过 (start={start:.1f}s)")
                    continue
            except FileNotFoundError as e:
                # PATH 无 ffmpeg（imageio_ffmpeg 导入失败时回退 "ffmpeg"）
                _log.error(f"[剪辑] 找不到 ffmpeg 可执行文件: {ffmpeg}")
                raise
            # 验证产物有效（非空文件）
            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                clip_files.append(clip_path)
            else:
                failed_segments += 1
                _log.warning(f"[WARN] 片段 {i+1}/{len(segments)} 生成空文件，跳过 (start={start:.1f}s)")

        # 全部失败才返回 None，避免拼接空列表
        if not clip_files:
            _log.warning(f"[剪辑] 全部 {len(segments)} 段切片失败，未生成集锦")
            return None
        if _cancelled():
            _log.info("[剪辑] 已取消（切片完成、拼接前），清理临时文件")
            return None

        # 拼接列表文件
        list_path = os.path.join(tmp_dir, "concat.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in clip_files:
                # concat demuxer 要求路径用正斜杠；单引号需转义
                # （文件名含 ' 时未转义会破坏 concat 语法，两路拼接全失败）
                p_norm = os.path.abspath(p).replace("\\", "/")
                p_norm = p_norm.replace("'", "'\\''")
                f.write(f"file '{p_norm}'\n")

        # 拼接：各片段由同一命令模板 + 同一组 encode_args 切出，编码参数天然一致，
        # 优先 -c copy 流拷贝（秒级完成、零质量损失）；失败时回退整体重编码兜底。
        # 旧实现无条件 re-encode，拼接时长与集锦总时长成正比（十几分钟集锦需多花
        # 数分钟），且二次编码带来代际质量损失，与"接近无损"目标相悖。
        _report(90, '正在拼接集锦视频（流拷贝）...')
        concat_copy_cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        concat_encode_cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", list_path,
        ] + encode_args + [
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]

        def _run_concat_encode():
            # NVENC 会话配额（消费卡限 2 路）：与切片/预览片段线程池共用信号量，
            # 避免并发时第 3 路会话 OpenEncodeSession 失败导致兜底拼接失败
            if use_nvenc:
                with nvenc_semaphore:
                    subprocess.run(concat_encode_cmd, check=True, capture_output=True,
                                   text=True, creationflags=_SBOX, timeout=1800)
            else:
                subprocess.run(concat_encode_cmd, check=True, capture_output=True,
                               text=True, creationflags=_SBOX, timeout=1800)

        def _remove_incomplete_output():
            # ffmpeg -y 失败中断会在 output_path 留下半截文件（无 moov/索引损坏），
            # 播放器无法打开却占用磁盘：流拷贝与重编码均失败时主动清理，
            # 避免 demo_output 目录残留无效视频被误当成品
            try:
                os.remove(output_path)
            except OSError:
                pass

        try:
            # 流拷贝不做编解码，600s 超时已非常宽裕
            subprocess.run(concat_copy_cmd, check=True, capture_output=True, text=True,
                           creationflags=_SBOX, timeout=600)
        except subprocess.TimeoutExpired:
            _log.warning("[剪辑] 流拷贝拼接超时（>600s），回退重编码拼接...")
            try:
                _run_concat_encode()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                _log.warning(f"[剪辑] 拼接失败: {_stderr_tail(e) or e}")
                _remove_incomplete_output()
                return None
        except subprocess.CalledProcessError as e:
            # 流拷贝失败（个别片段参数异常，如源视频中途变分辨率）→ 回退重编码
            _log.warning(f"[剪辑] 流拷贝拼接失败，回退重编码拼接: {_stderr_tail(e) or e}")
            try:
                _run_concat_encode()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e2:
                _log.warning(f"[剪辑] 拼接失败: {_stderr_tail(e2) or e2}")
                _remove_incomplete_output()
                return None
        _report(98, '清理临时文件...')
    finally:
        # 整体兜底清理：成功/失败/取消/未预期异常（旧实现只在三条已知路径清理，
        # FileNotFoundError/磁盘满等异常会跳过全部清理，hl-* 目录永久残留）
        _cleanup_tmp(clip_files, os.path.join(tmp_dir, "concat.txt"), tmp_dir)

    total_dur = sum(e - s for s, e in segments)
    skip_info = f" | 跳过 {failed_segments} 段失败" if failed_segments > 0 else ""
    _log.info(f"集锦已生成: {output_path}（{encode_tag} | 共 {len(clip_files)}/{len(segments)} 段{skip_info} | 时长 {total_dur:.0f}s）")
    return output_path
