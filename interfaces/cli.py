"""
CLI v4 — 多用户 + 隐私选择 + 纯净聊天
"""
import sys
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.align import Align
from rich import box

from config_loader import load_config, get
from brain.agent import AgentBrain
from knowledge.memory import UserManager
from skills.manager import SkillManager

console = Console()


CAT_ART = """
     ╭──────────────────────╮
     │    ╭───╮  ╭───╮      │
     │    │ ╮ │  │ ╮ │      │
     │     ╰─╯    ╰─╯       │
     │   ╭────────────╮     │
     │   ╰────────────╯     │
     │    ╲  ╱╲  ╱          │
     │     ╲╱  ╲╱           │
     │  ╭──────────╮        │
     │  │ ████████ │        │
     │  ╰──────────╯        │
     │   ████████████       │
     ╰──────────────────────╯
"""


def print_logo():
    console.clear()
    logo = Text(CAT_ART)
    logo.stylize("cyan")
    console.print(Align.center(logo))
    console.print(Align.center("[bold cyan]工具猫 AI[/bold cyan]"))
    console.print()


PROVIDERS = {
    "1": {"name": "DeepSeek",       "base_url": "https://api.deepseek.com",          "model": "deepseek-chat"},
    "2": {"name": "豆包/火山引擎",   "base_url": "https://ark.cn-beijing.volces.com/api/v3", "model": "ep-xxxxxx"},
    "3": {"name": "OpenAI (GPT)",   "base_url": "https://api.openai.com",            "model": "gpt-4o-mini"},
    "4": {"name": "Kimi (月之暗面)", "base_url": "https://api.moonshot.cn/v1",       "model": "kimi-k2.5"},
    "5": {"name": "通义千问",        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-max"},
    "6": {"name": "其他（手动输入）", "base_url": "", "model": ""},
}


def choose_provider() -> dict:
    """选择模型提供商"""
    console.print(Panel(
        "[bold]选择你要用的模型：[/bold]\n\n"
        "  1️⃣  DeepSeek\n"
        "  2️⃣  豆包 / 火山引擎\n"
        "  3️⃣  OpenAI (GPT)\n"
        "  4️⃣  Kimi (月之暗面)\n"
        "  5️⃣  通义千问\n"
        "  6️⃣  其他（手动输入）",
        border_style="cyan", box=box.ROUNDED, padding=(1, 2),
    ))
    choice = Prompt.ask("[bold]请选择[/bold]", choices=["1", "2", "3", "4", "5", "6"], default="1")

    provider = dict(PROVIDERS[choice])
    if choice == "2":
        console.print("\n[dim]豆包需要填入推理端点 ID (ep-xxxxx 格式)[/dim]")
        ep = Prompt.ask("[bold]Endpoint ID[/bold]").strip()
        if ep:
            provider["model"] = ep
    elif choice == "6":
        provider["base_url"] = Prompt.ask("[bold]API 地址[/bold]").strip()
        provider["model"] = Prompt.ask("[bold]模型名[/bold]").strip()

    return provider


def key_login() -> tuple:
    """Key 输入 + 隐私选择，返回 (api_key, privacy_mode, provider_info)"""
    print_logo()

    # 1. 选模型
    provider = choose_provider()
    console.print()

    # 2. 输 Key
    console.print(f"[dim]输入你的 [bold]{provider['name']}[/bold] API Key[/dim]\n")
    api_key = Prompt.ask("[bold cyan]🔑 API Key[/bold cyan]").strip()
    while not api_key:
        console.print("[red]Key 不能为空[/red]")
        api_key = Prompt.ask("[bold cyan]🔑 API Key[/bold cyan]").strip()

    # 3. 隐私选择（仅首次）
    user_mgr = UserManager()
    welcome_msg = user_mgr.login(api_key)

    if user_mgr.is_new_user:
        console.print()
        console.print(Panel(
            "[bold]选择隐私模式：[/bold]\n\n"
            "[cyan]🌐  共享模式（推荐）[/cyan]\n"
            "  你的使用习惯会匿名汇入共享库\n"
            "  同时享受社区积累的优化经验\n"
            "  越多人用，AI 越聪明\n\n"
            "[yellow]🔒  私有模式[/yellow]\n"
            "  所有数据不上传不共享\n"
            "  但也不能从社区受益\n"
            "  白纸一张开始",
            border_style="cyan", box=box.ROUNDED, padding=(1, 2),
        ))
        mode = Prompt.ask("[bold]选择模式[/bold]", choices=["sync", "local"], default="sync")
        user_mgr.set_privacy(mode)
        console.print()

    console.print(f"[green]{welcome_msg}[/green]")
    console.print()

    return api_key, user_mgr.privacy_mode, provider


def show_help():
    console.print()
    console.print(Panel(
        "[bold cyan]跟我这样说：[/bold cyan]\n\n"
        "  • 「帮我做6个穿搭视频，35秒」\n"
        "  • 「分析一下这些素材」\n"
        "  • 「换个风格，改成小清新」\n"
        "  • 「保存当前BGM偏好」\n\n"
        "[dim]━━━━━━━━━━━━━━━━━━━━[/dim]\n\n"
        "[bold]系统命令：[/bold]\n"
        "  /help     — 显示帮助\n"
        "  /quit     — 退出\n"
        "  /reset    — 重置对话\n"
        "  /privacy  — 切换隐私模式\n"
        "  /prefs    — 查看你的偏好\n"
        "  /trends   — 查看社区趋势\n"
        "  /model    — 查看当前模型",
        border_style="cyan", box=box.ROUNDED, padding=(1, 2),
    ))
    console.print()


def show_ai_response(text: str):
    if not text:
        return
    console.print()
    console.print(Panel(
        Markdown(text),
        border_style="cyan", box=box.SIMPLE, padding=(0, 1),
    ))


def main():
    try:
        load_config()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    # ── 启动：选模型 → Key 登录 → 隐私选择 ──
    api_key, privacy_mode, provider = key_login()

    # 把 Key 和 Provider 写入配置
    import yaml
    cfg_path = Path.cwd() / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["llm"]["api_key"] = api_key
    cfg["llm"]["base_url"] = provider["base_url"]
    cfg["llm"]["model"] = provider["model"]
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # 重新加载配置
    load_config()

    # 构建用户上下文
    user_mgr = UserManager()
    user_mgr.login(api_key, privacy_mode)

    # ── 初始化 ──
    print_logo()

    mode_tag = "🌐" if user_mgr.is_shared_mode() else "🔒"
    console.print(Align.center(f"[bold]{mode_tag} 工具猫 AI[/bold]  [dim]输入 /help 看命令  |  /quit 退出[/dim]"))
    console.print()

    try:
        brain = AgentBrain(user_mgr)
    except Exception as e:
        console.print(f"[red]❌ 初始化失败: {e}[/red]")
        sys.exit(1)

    try:
        skill_mgr = SkillManager()
        skill_mgr.load_all()
    except Exception:
        pass

    # ── 对话循环 ──
    while brain.running:
        try:
            user_input = Prompt.ask("[bold cyan]🐱[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]👋 下次见~[/cyan]")
            break

        if not user_input:
            continue

        # ── 命令 ──
        if user_input.startswith("/"):
            cmd = user_input[1:].lower()

            if cmd in ("quit", "exit"):
                console.print("\n[cyan]👋 下次见~[/cyan]")
                break

            elif cmd == "help":
                show_help()
                continue

            elif cmd == "reset":
                brain.reset()
                console.print("\n[dim]重置完成[/dim]")
                continue

            elif cmd == "privacy":
                current = "🌐 共享模式" if user_mgr.is_shared_mode() else "🔒 私有模式"
                console.print(f"\n[dim]当前: {current}[/dim]")
                new_mode = Prompt.ask(
                    "切换为", choices=["sync", "local"],
                    default="sync" if user_mgr.privacy_mode == "local" else "local"
                )
                if new_mode != user_mgr.privacy_mode:
                    user_mgr.set_privacy(new_mode)
                    brain.memory.is_shared = user_mgr.is_shared_mode()
                    console.print(f"[green]已切换至 {'🌐 共享' if new_mode == 'sync' else '🔒 私有'}模式[/green]")
                continue

            elif cmd == "prefs":
                prefs = brain.memory.get_all_prefs()
                if prefs:
                    console.print()
                    for cat, data in prefs.items():
                        items = []
                        for k, v in data.items():
                            val = v.get("value", v) if isinstance(v, dict) else v
                            ts = datetime.fromtimestamp(v.get("timestamp", 0)).strftime("%m-%d %H:%M") if isinstance(v, dict) and v.get("timestamp") else ""
                            items.append(f"  [dim]{k}[/dim]: {val}  [dim]{ts}[/dim]")
                        if items:
                            console.print(Panel("\n".join(items), title=cat, border_style="green", box=box.SIMPLE))
                else:
                    console.print("\n[dim]还没有偏好记录~ 多聊聊让我更懂你[/dim]")
                continue

            elif cmd == "trends":
                if user_mgr.is_shared_mode():
                    trends = brain.memory.get_shared_trends("prefs")
                    if trends:
                        console.print("\n[bold]📊 社区趋势[/bold]")
                        for t in trends:
                            console.print(f"  [{t['key']}] {t['value']}  [dim]({t['count']}人)[/dim]")
                    else:
                        console.print("\n[dim]数据太少，再多点人用就能看到趋势了[/dim]")
                else:
                    console.print("\n[dim]当前是私有模式，看不到社区趋势[/dim]")
                continue

            elif cmd == "model":
                console.print(f"\n[cyan]🔧 {get('llm.model', 'deepseek-chat')}[/cyan]")
                continue

            else:
                console.print(f"\n[dim]未知命令 /{cmd}，输入 /help 查看[/dim]")
                continue

        # ── 对话 ──
        console.print()
        with console.status("[cyan] 思考中...[/cyan]", spinner="dots"):
            try:
                response_text = ""
                for result in brain.chat(user_input):
                    if result["type"] == "text":
                        response_text = result["content"]
                    elif result["type"] == "error":
                        response_text = f"❌ {result['content']}"
                    elif result["type"] == "confirm_needed":
                        tc = result["content"]
                        confirm = Confirm.ask(f"  [yellow]执行「{tc['name']}」?[/yellow]")
                        if confirm:
                            r = brain.tool_registry.execute(tc["name"], tc["arguments"])
            except Exception as e:
                response_text = f"❌ 出错了: {e}"
                import traceback
                console.print(f"[dim]{traceback.format_exc()[:300]}[/dim]")

        show_ai_response(response_text)

    brain.memory.clear_session()
