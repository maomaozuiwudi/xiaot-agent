"""
小红书内容工坊 AI Agent v2.0 — 主入口

支持三种启动模式：
  python main.py           → CLI 终端对话模式
  python main.py --web     → Web 服务模式
  python main.py --gui     → Desktop GUI 模式
"""
import sys
import os

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    """主入口"""
    args = sys.argv[1:]

    # 加载配置（尽早，让其他模块能用 config_loader.get()）
    from config_loader import load_config
    try:
        load_config()
    except FileNotFoundError:
        print("=" * 50)
        print("  ⚠️  未找到 config.yaml")
        print()
        print("  请复制 config.yaml 模板并填入 API Key：")
        print("    1. cp config.yaml config.local.yaml")
        print("    2. 编辑 config.local.yaml 填入 API Key")
        print("    3. 运行: python main.py")
        print("=" * 50)
        sys.exit(1)

    # 解析命令行参数
    if "--web" in args:
        _start_web()
    elif "--gui" in args:
        _start_gui()
    else:
        _start_cli()


def _start_cli():
    """启动 CLI 终端对话模式"""
    from interfaces.cli import main as cli_main
    cli_main()


def _start_web():
    """启动 Web 服务模式"""
    print("[Web] 启动 Web 服务...")
    # Web 模式待实现
    # from interfaces.web.server import run
    # run()
    print("[Web] 暂未实现，期待贡献！")
    print("[Fallback] 切换到 CLI 模式...")
    _start_cli()


def _start_gui():
    """启动 Desktop GUI 模式"""
    print("[GUI] 启动桌面客户端...")
    # GUI 模式待实现
    # from interfaces.gui import run
    # run()
    print("[GUI] 暂未实现，期待贡献！")
    print("[Fallback] 切换到 CLI 模式...")
    _start_cli()


if __name__ == "__main__":
    main()
