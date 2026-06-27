"""
小t Agent 终端入口

安装后可直接在终端输入 `xiaot` 启动。
支持参数：--web（Web界面）、--gui（桌面界面）
"""
import os
import sys

# 把包目录加到 sys.path，让 flat imports 工作
_entry_dir = os.path.dirname(os.path.abspath(__file__))
if _entry_dir not in sys.path:
    sys.path.insert(0, _entry_dir)

from main import main


if __name__ == "__main__":
    main()
