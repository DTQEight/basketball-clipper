"""pytest 公共配置：把项目根目录加入 sys.path，使扁平模块可直接导入。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
