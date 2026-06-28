"""
小t Agent 终端入口

安装后可直接在终端输入 `xiaot` 启动。
支持参数：--web（Web界面）、--gui（桌面界面）
"""
import os
import sys
import subprocess

# 把包目录加到 sys.path，让 flat imports 工作
_entry_dir = os.path.dirname(os.path.abspath(__file__))
if _entry_dir not in sys.path:
    sys.path.insert(0, _entry_dir)


def _first_run_check():
    """首次运行检查：自动补全缺失的依赖"""
    if os.environ.get("XIAOT_SKIP_CHECK"):
        return

    # 检查 playwright 浏览器是否已安装
    try:
        from playwright.sync_api import sync_playwright

        try:
            with sync_playwright() as p:
                p.chromium.launch(headless=True).close()
        except Exception:
            # 浏览器未安装，自动安装
            print("🌐 首次启动，正在安装浏览器引擎...")
            subprocess.check_call(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("✅ 浏览器引擎安装完成")
    except ImportError:
        print("⚠️  playwright 未安装，部分功能不可用")
        print("   运行: playwright install chromium")

    # 检查 config.yaml
    config_path = os.path.join(_entry_dir, "config.yaml")
    example_path = os.path.join(_entry_dir, "config.example.yaml")
    if not os.path.exists(config_path) and os.path.exists(example_path):
        import shutil
        shutil.copy2(example_path, config_path)
        print("📝 已生成 config.yaml，请编辑填入 API Key")


def main():
    _first_run_check()
    from main import main as _main
    _main()


if __name__ == "__main__":
    main()
