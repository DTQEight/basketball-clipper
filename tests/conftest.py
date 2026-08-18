"""pytest 公共配置：项目根注入 sys.path + 缓存目录隔离。

必须在任何 services/app 导入**之前**设置 BBALL_CACHE_ROOT：
services.state 模块级会 makedirs 项目 cache/tmp、改写 TMP/TEMP/TMPDIR
环境变量、并执行 _purge_old_clips()（可能删除用户 demo_output 下 7 天前
的旧片段）。不隔离的话，跑一次测试就污染真实缓存目录/环境变量/删用户文件。
"""
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 会话级缓存根：优先复用外部指定的 BBALL_TEST_CACHE，否则建临时目录
_TEST_CACHE = os.environ.get("BBALL_TEST_CACHE")
if not _TEST_CACHE:
    _TEST_CACHE = tempfile.mkdtemp(prefix="bball-test-cache-")
os.environ["BBALL_CACHE_ROOT"] = _TEST_CACHE
