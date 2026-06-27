"""
CLI v5 — 多用户 + 隐私选择 + 纯净聊天 + 双Key登录
"""
import sys
import os
import json
import base64
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.markup import escape
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
    "1": {"name": "DeepSeek",       "base_url": "https://api.deepseek.com",          "model": "deepseek-chat",          "type": "openai_compat"},
    "2": {"name": "豆包/火山引擎",   "base_url": "https://ark.cn-beijing.volces.com/api/v3", "model": "ep-xxxxxx",         "type": "openai_compat"},
    "3": {"name": "OpenAI (GPT)",   "base_url": "https://api.openai.com",            "model": "gpt-4o-mini",           "type": "openai_compat"},
    "4": {"name": "Kimi (月之暗面)", "base_url": "https://api.moonshot.cn/v1",       "model": "kimi-k2.5",             "type": "openai_compat"},
    "5": {"name": "通义千问",        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-max",             "type": "openai_compat"},
    "6": {"name": "Claude (Anthropic)", "base_url": "https://api.anthropic.com",     "model": "claude-sonnet-4-20250514", "type": "anthropic"},
    "7": {"name": "其他（手动输入）", "base_url": "", "model": "",                   "type": "openai_compat"},
}

# ── 登录态持久化 ──
_KEY_FILE = Path.home() / ".xhs_agent_auth.json"


def _save_auth(api_key: str, provider: dict, vision_key: str = "", vision_provider: dict = None,
               rnote_api_key: str = ""):
    """保存登录态到本地（含看图 Key + Rnote Key）"""
    try:
        data = {
            "api_key": api_key,
            "provider_name": provider["name"],
            "base_url": provider["base_url"],
            "model": provider["model"],
            "vision_key": vision_key,
            "vision_provider_name": vision_provider["name"] if vision_provider else "",
            "vision_base_url": vision_provider["base_url"] if vision_provider else "",
            "vision_model": vision_provider["model"] if vision_provider else "",
            "rnote_api_key": rnote_api_key,
            "timestamp": datetime.now().isoformat(),
        }
        _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _KEY_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _load_auth() -> tuple:
    """读取保存的登录态，返回 (api_key, provider, vision_key, vision_provider, rnote_api_key) 或全 None"""
    try:
        if _KEY_FILE.exists():
            data = json.loads(_KEY_FILE.read_text(encoding="utf-8"))
            key = data.get("api_key", "")
            prov = {
                "name": data.get("provider_name", "DeepSeek"),
                "base_url": data.get("base_url", "https://api.deepseek.com"),
                "model": data.get("model", "deepseek-chat"),
            }
            vision_key = data.get("vision_key", "")
            vision_prov = None
            if vision_key and data.get("vision_provider_name"):
                vision_prov = {
                    "name": data.get("vision_provider_name", "Kimi (月之暗面)"),
                    "base_url": data.get("vision_base_url", "https://api.moonshot.cn/v1"),
                    "model": data.get("vision_model", "kimi-k2.5"),
                }
            rnote_key = data.get("rnote_api_key", "")
            if key:
                return key, prov, vision_key, vision_prov, rnote_key
    except Exception:
        pass
    return None, None, None, None, None


def choose_provider(saved_key=None, saved_prov=None) -> dict:
    """选择模型提供商（支持继续上次登录、返回重选）"""
    while True:
        # 构建选项面板
        lines = ["[bold]选择你要用的模型：[/bold]\n"]
        # 如果有登录态，选项 0 放在最前面
        if saved_key and saved_prov:
            lines.append(f"  [bold cyan]0️⃣  继续上次登录（{saved_prov['name']} · {saved_prov['model']}）[/bold cyan]\n")
        lines.extend([
            "  1️⃣  DeepSeek\n",
            "  2️⃣  豆包 / 火山引擎\n",
            "  3️⃣  OpenAI (GPT)\n",
            "  4️⃣  Kimi (月之暗面)\n",
            "  5️⃣  通义千问\n",
            "  6️⃣  Claude (Anthropic)\n",
            "  7️⃣  其他（手动输入）\n",
        ])
        if saved_key and saved_prov:
            lines.append("  [dim]0 = 继续上次登录[/dim]")
        else:
            lines.append("  [dim]首次使用请选择 1-7[/dim]")

        console.print(Panel(
            "".join(lines),
            border_style="cyan", box=box.ROUNDED, padding=(1, 2),
        ))

        # 根据是否有登录态决定可选范围
        if saved_key and saved_prov:
            choices = ["0", "1", "2", "3", "4", "5", "6", "7"]
            default = "0"
        else:
            choices = ["1", "2", "3", "4", "5", "6", "7"]
            default = "1"

        choice = Prompt.ask("[bold]请选择[/bold]", choices=choices, default=default)

        # 继续上次登录
        if choice == "0" and saved_key and saved_prov:
            return dict(saved_prov), "continue"

        provider = dict(PROVIDERS[choice])
        if choice == "2":
            console.print("\n[dim]豆包需要填入推理端点 ID (ep-xxxxx 格式)[/dim]")
            ep = Prompt.ask("[bold]Endpoint ID[/bold]").strip()
            if ep:
                provider["model"] = ep
        elif choice == "7":
            provider["base_url"] = Prompt.ask("[bold]API 地址[/bold]").strip()
            provider["model"] = Prompt.ask("[bold]模型名[/bold]").strip()

        # 确认
        console.print()
        console.print(f"[dim]  模型: {provider['name']}  |  地址: {provider['base_url']}  |  模型: {provider['model']}[/dim]")
        if Confirm.ask("[bold]确认使用这个模型？[/bold]", default=True):
            return provider, "fresh"
        console.print()


def _choose_vision_provider() -> tuple:
    """选择视觉模型（同主模型选择界面），返回 (vision_key, vision_provider)"""
    console.print()
    console.print(Panel(
        "[bold]📷 配置看图模块[/bold]\n\n"
        "可选择单独的视觉模型来分析图片素材。\n"
        "如果不单独配置，则使用主模型承担看图任务（需主模型支持多模态）。",
        border_style="yellow", box=box.SIMPLE, padding=(1, 2),
    ))

    while True:
        lines = [
            "[bold]选择看图用的视觉模型：[/bold]\n",
            "  [bold cyan]0️⃣  使用主模型（不单独配置）[/bold cyan]\n",
            "  1️⃣  DeepSeek\n",
            "  2️⃣  豆包 / 火山引擎\n",
            "  3️⃣  OpenAI (GPT)\n",
            "  4️⃣  Kimi (月之暗面)\n",
            "  5️⃣  通义千问\n",
            "  6️⃣  Claude (Anthropic)\n",
            "  7️⃣  其他（手动输入）\n",
            "  [dim]0 = 跟主模型用同一个，不额外配置[/dim]",
        ]
        console.print(Panel(
            "".join(lines),
            border_style="yellow", box=box.ROUNDED, padding=(1, 2),
        ))

        choice = Prompt.ask("[bold]请选择[/bold]", choices=["0", "1", "2", "3", "4", "5", "6", "7"], default="0")

        # 使用主模型
        if choice == "0":
            return "", None

        vision_prov = dict(PROVIDERS[choice])
        if choice == "2":
            console.print("\n[dim]豆包需要填入推理端点 ID (ep-xxxxx 格式)[/dim]")
            ep = Prompt.ask("[bold]Endpoint ID[/bold]").strip()
            if ep:
                vision_prov["model"] = ep
        elif choice == "7":
            vision_prov["base_url"] = Prompt.ask("[bold]API 地址[/bold]").strip()
            vision_prov["model"] = Prompt.ask("[bold]模型名[/bold]").strip()

        console.print()
        console.print(f"[dim]  模型: {vision_prov['name']}  |  地址: {vision_prov['base_url']}  |  模型: {vision_prov['model']}[/dim]")
        if not Confirm.ask("[bold]确认使用这个视觉模型？[/bold]", default=True):
            console.print()
            continue

        console.print(f"\n[dim]输入 [bold]{vision_prov['name']}[/bold] 的 API Key（看图专用）[/dim]")
        vision_key = Prompt.ask("[bold cyan]🔑 视觉 API Key[/bold cyan]").strip()
        while not vision_key:
            console.print("[red]Key 不能为空（不想配置请选 0 返回）[/red]")
            vision_key = Prompt.ask("[bold cyan]🔑 视觉 API Key[/bold cyan]").strip()

        return vision_key, vision_prov


def key_login() -> tuple:
    """Key 输入 + 隐私选择，返回 (api_key, privacy_mode, provider_info, rnote_api_key)"""
    print_logo()

    # 检查是否有保存的登录态
    saved_key, saved_prov, saved_vision_key, saved_vision_prov, saved_rnote_key = _load_auth()

    # 选择模型（含继续上次登录）
    provider, mode = choose_provider(saved_key, saved_prov)

    if mode == "continue":
        # 继续上次登录
        api_key = saved_key
        vision_key = saved_vision_key or ""
        vision_prov = saved_vision_prov
        console.print()
        vision_line = ""
        if vision_key and vision_prov:
            vision_line = f"\n  [dim]看图: {vision_prov['name']} · {vision_prov['model']}[/dim]"
        console.print(Panel(
            f"[green]✅ 继续上次登录[/green]\n"
            f"  [dim]主模型: {provider['name']} · {provider['model']}[/dim]\n"
            f"  [dim]Key: {api_key[:8]}...{api_key[-4:]}[/dim]"
            + vision_line,
            border_style="green", box=box.SIMPLE, padding=(1, 2),
        ))
    else:
        # 全新登录：输主 Key
        console.print()
        console.print(f"[dim]输入你的 [bold]{provider['name']}[/bold] API Key[/dim]\n")
        api_key = Prompt.ask("[bold cyan]🔑 API Key[/bold cyan]").strip()
        while not api_key:
            console.print("[red]Key 不能为空[/red]")
            api_key = Prompt.ask("[bold cyan]🔑 API Key[/bold cyan]").strip()

        # 选择视觉模型（看图用）
        vision_key, vision_prov = _choose_vision_provider()

        # 保存登录态
        _save_auth(api_key, provider, vision_key, vision_prov)

    # ── Rnote API Key（小红书搜索，可选，留空仅用常规搜索） ──
    rnote_api_key = ""
    if mode == "continue" and saved_rnote_key:
        rnote_api_key = saved_rnote_key
        console.print(f"\n[dim]已保存 Rnote Key: {rnote_api_key[:8]}...{rnote_api_key[-4:]}[/dim]")
        if Confirm.ask("[bold]更换 Rnote Key？[/bold]", default=False):
            rnote_api_key = Prompt.ask("[bold cyan]🔑 Rnote API Key[/bold cyan] (留空跳过)").strip()
    else:
        console.print("\n[dim]Rnote API Key 用于搜索小红书笔记（可选，留空仅用常规搜索）[/dim]")
        rnote_api_key = Prompt.ask("[bold cyan]🔑 Rnote API Key (可选)[/bold cyan]").strip()

    # 重新保存（含 Rnote Key）
    if mode == "continue" or rnote_api_key != saved_rnote_key:
        _save_auth(api_key, provider, vision_key, vision_prov, rnote_api_key)

    # 设为环境变量，供搜索模块使用
    if rnote_api_key:
        os.environ["RNOTE_API_KEY"] = rnote_api_key

    # 隐私选择
    console.print()
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

    # 把 vision_key 和 vision 模型信息带回给 main()
    provider["vision_key"] = vision_key
    if vision_prov:
        provider["vision_name"] = vision_prov["name"]
        provider["vision_base_url"] = vision_prov["base_url"]
        provider["vision_model"] = vision_prov["model"]
    return api_key, user_mgr.privacy_mode, provider, rnote_api_key


# ── 界面渲染 ──

class ChatUI:
    """聊天界面渲染器：纯静态 Panel 渲染，不依赖 Live"""

    def __init__(self, mode_tag: str, main_model: str, vision_model: str = "",
                 total_prompt: int = 0, total_completion: int = 0,
                 context_window: int = 1000000):
        self.mode_tag = mode_tag
        self.main_model = main_model
        self.vision_model = vision_model
        self.total_prompt_tokens = total_prompt
        self.total_completion_tokens = total_completion
        self.context_window = context_window
        self.conv_history: list[tuple[str, str]] = []
        self._status = "idle"  # idle | thinking | input

        # 待优化的面板（可选的，用于 thinking 状态时显示的 step 消息）
        self.thinking_step = ""
        self.thinking_step_visible = False

    def set_status(self, status: str):
        self._status = status

    def add_message(self, role: str, content: str):
        self.conv_history.append((role, content))

    def _fmt_number(self, n: int) -> str:
        return f"{n:,}" if n >= 1000 else str(n)

    def _fmt_context(self, used: int, total: int) -> str:
        def _f(v: int) -> str:
            if v >= 1_000_000:
                return f"{v / 1_000_000:.1f}M"
            elif v >= 1_000:
                return f"{v / 1_000:.1f}K"
            return str(v)
        return f"{_f(used)}/{_f(total)}"

    def render_header(self) -> Panel:
        line = f"[bold][cyan]🐱 工具猫 AI[/cyan][/bold]   [dim]{escape(self.mode_tag)}[/dim]"
        if self.vision_model:
            line += f"   [dim]📷 {escape(self.vision_model)}[/dim]"
        return Panel(line, border_style="cyan", box=box.ROUNDED, padding=(0, 2))

    def render_body(self) -> Panel:
        if self.conv_history:
            lines = []
            for role, text in self.conv_history[-6:]:
                safe = escape(text)
                if role == "user":
                    lines.append(f"[bold][cyan]🐱[/cyan][/bold] {safe}")
                elif role == "ai" and text:
                    lines.append(safe)
                lines.append("")
            content = "\n".join(lines).strip()
        else:
            content = "[dim]跟我说：\n  •「帮我做6个穿搭视频，35秒」\n  •「分析一下这些素材」\n  •「换个风格，改成小清新」\n  • /help 看完整命令[/dim]"
        return Panel(content, title="💬 对话", border_style="cyan", box=box.ROUNDED, padding=(0, 1))

    def render_think_bar(self) -> Panel:
        """thinking 状态时显示的等待框"""
        text = "[yellow]⏳ 思考中...[/yellow]"
        if self.thinking_step_visible and self.thinking_step:
            text += f"  [dim]{escape(self.thinking_step)}[/dim]"
        return Panel(text, border_style="yellow", box=box.ROUNDED, padding=(0, 2))

    def render_status_bar(self) -> Panel:
        parts = [f"[bold]{escape(self.main_model)}[/bold]"]
        if self.vision_model:
            parts.append(f"[dim]{escape(self.vision_model)}[/dim]")
        total_used = self.total_prompt_tokens + self.total_completion_tokens
        parts.append(f"📤 {self._fmt_number(self.total_completion_tokens)}  📥 {self._fmt_number(self.total_prompt_tokens)}")
        parts.append(f"上下文 {self._fmt_context(total_used, self.context_window)}")
        return Panel(
            "  │  ".join(parts),
            border_style="grey58", box=box.SIMPLE, padding=(0, 2),
        )

    def render_all(self):
        """打印完整界面到终端"""
        console.clear()
        console.print(self.render_header())
        console.print(self.render_body())
        console.print()
        if self._status == "thinking":
            console.print(self.render_think_bar())
        console.print(self.render_status_bar())
        console.print()

    def get_input(self) -> str:
        """显示带边框输入框，光标在框内"""
        w = console.width - 4
        console.print(f"[bold green]╭{'─' * (w - 2)}╮[/bold green]")
        text = console.input("[bold green]│ 🐱 [/bold green]").strip()
        console.print(f"[bold green]╰{'─' * (w - 2)}╯[/bold green]")
        return text


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


def _handle_file_input(text: str, brain: AgentBrain) -> tuple[bool, str]:
    """检测用户输入是否为文件路径，将路径文字传给 AI，由 AI 自主调用 vision_analyze 工具"""
    import re

    def _msys_to_win(p: str) -> str:
        m = re.match(r'^/([a-zA-Z])/(.*)', p)
        if m:
            return f"{m.group(1).upper()}:/{m.group(2)}"
        return p

    parts = [p.strip() for p in re.split(r'[,\s]+', text.strip()) if p.strip()]
    file_paths = []

    for part in parts:
        win_path = _msys_to_win(part)
        p = Path(win_path)
        if p.exists():
            file_paths.append(str(p.resolve()))
        elif os.path.exists(win_path):
            file_paths.append(os.path.abspath(win_path))

    if not file_paths:
        return False, ""

    # 只识别媒体文件
    media_exts = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.mov', '.avi', '.mkv'}
    media_files = [fp for fp in file_paths if Path(fp).suffix.lower() in media_exts]

    if not media_files:
        return False, ""

    names = [Path(p).name for p in media_files]
    paths_str = '\n'.join(media_files)

    # 用户原始输入文本（用于 RAG 检索 + AI 理解意图）
    original_text = text.strip()

    console.print(f"\n[green]✅ 检测到素材: {', '.join(names)}[/green]")
    console.print(f"[dim]  AI 将搜索参考→剪辑→分析→合成[/dim]")

    text = (
        f"{original_text}\n\n"
        f"【用户拖入了素材文件】\n"
        f"文件: {', '.join(names)}\n"
        f"完整路径:\n{paths_str}\n"
        f"先搜索穿搭类爆款参考（调 search_web 或 search_xhs），"
        f"然后根据需要调 clip_videos 剪辑视频、"
        f"调 vision_analyze 分析画面内容。"
    )
    return True, text


def main():
    try:
        load_config()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    # ── 启动：选模型 → Key 登录 → 隐私选择 ──
    api_key, privacy_mode, provider, rnote_api_key = key_login()

    # 把 Key 和 Provider 写入配置
    import yaml
    cfg_path = Path.cwd() / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["llm"]["api_key"] = api_key
    cfg["llm"]["base_url"] = provider["base_url"]
    cfg["llm"]["model"] = provider["model"]
    cfg["llm"]["provider"] = provider.get("type", "openai_compat")
    # 看图 Key
    vision_key = provider.get("vision_key", "")
    if vision_key:
        cfg.setdefault("llm", {}).setdefault("vision", {})
        cfg["llm"]["vision"]["api_key"] = vision_key
        cfg["llm"]["vision"]["base_url"] = provider.get("vision_base_url", "https://api.moonshot.cn/v1")
        cfg["llm"]["vision"]["model"] = provider.get("vision_model", "kimi-k2.5")
    else:
        cfg["llm"].pop("vision", None)
    # Rnote API Key（小红书搜索，可选）
    if rnote_api_key:
        cfg["rnote"] = {"api_key": rnote_api_key}
    elif "rnote" in cfg:
        cfg.pop("rnote", None)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # 重新加载配置
    load_config()

    # 构建用户上下文
    user_mgr = UserManager()
    user_mgr.login(api_key, privacy_mode)

    # ── 初始化 ──
    mode_tag = "🌐" if user_mgr.is_shared_mode() else "🔒"
    # 构造主模型显示名：提供商标识 · 模型ID
    main_model_display = f"{provider['name']} · {provider['model']}"
    vision_model_display = ""
    if vision_key:
        vision_model_display = provider.get("vision_name", provider.get("vision_model", "视觉模型"))

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

    # ── 聊天界面 ──
    ui = ChatUI(
        mode_tag, main_model_display, vision_model_display,
        context_window=get("llm.context_window", 1000000),
    )

    while brain.running:
        # ══════════════ 显示界面 ══════════════
        ui.render_all()

        # ══════════════ 获取输入 ══════════════
        try:
            user_input = ui.get_input()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]⚠️ Ctrl+C 已捕获，输入 /quit 退出[/yellow]")
            continue

        if not user_input:
            continue

        # ══════════════ 拖拽文件检测（先于命令处理）══════
        handled, analysis_result = _handle_file_input(user_input, brain)
        if handled:
            if analysis_result:
                user_input = analysis_result  # str 或 list（多模态消息）
            else:
                _pause()
                continue

        # ══════════════ 命令处理（MSYS 文件路径 /e/xxx 会以 / 开头，先豁免）══════
        _is_file_path = False
        if isinstance(user_input, str) and user_input.startswith("/"):
            import re as _re
            # 拆分空格分隔的多个路径，逐个检测是否存在
            _parts = user_input.split()
            _all_paths = True
            for _part in _parts:
                _m = _re.match(r'^/([a-zA-Z])/(.*)', _part)
                if _m:
                    _win = f"{_m.group(1).upper()}:/{_m.group(2)}"
                    if Path(_win).exists():
                        continue
                _all_paths = False
                break
            _is_file_path = _all_paths and len(_parts) > 0

        if not _is_file_path and isinstance(user_input, str) and user_input.startswith("/"):
            cmd = user_input[1:].lower()

            if cmd in ("quit", "exit"):
                console.print("\n[cyan]👋 下次见~[/cyan]")
                break

            elif cmd == "help":
                show_help()
                _pause()
                continue

            elif cmd == "reset":
                brain.reset()
                ui.conv_history.clear()
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
                _pause()
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
                _pause()
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
                _pause()
                continue

            elif cmd == "model":
                console.print(f"\n[cyan]🔧 {get('llm.model', 'deepseek-chat')}[/cyan]")
                _pause()
                continue

            else:
                console.print(f"\n[dim]未知命令 /{cmd}，输入 /help 查看[/dim]")
                _pause()
                continue

        # ══════════════ 对话处理 ══════════════
        # 多模态输入时提取文本用于 UI 显示和关键词检测
        if isinstance(user_input, list):
            ui_display_text = "📎 [拖入了素材文件，含画面帧]"
            text_for_keywords = ""
            for part in user_input:
                if part.get("type") == "text":
                    text_for_keywords = part.get("text", "")
                    break
        else:
            ui_display_text = user_input
            text_for_keywords = user_input

        ui.add_message("user", ui_display_text)
        ui.set_status("thinking")
        # 显示 thinking 状态
        ui.render_all()

        response_text = ""
        try:
            for result in brain.chat(user_input):
                t = result["type"]
                c = result["content"]
                if t == "text":
                    response_text = c
                elif t == "error":
                    response_text = f"❌ {c}"
                elif t == "step":
                    console.print(f"[dim]{c}[/dim]")
                elif t == "tool_call":
                    console.print(f"[dim]🔧 {c}[/dim]")
                    ui.render_all()
                elif t == "confirm_needed":
                    tc = c
                    if Confirm.ask(f"  [yellow]执行「{tc['name']}」?[/yellow]"):
                        r = brain.tool_registry.execute(tc["name"], tc["arguments"])
                        # 补全 assistant tool_call 消息 + tool result，继续对话
                        from brain.tools import format_tool_result
                        result_msg = format_tool_result(tc["name"], r, tool_call_id=tc.get("id", ""))
                        # tc 格式: {"id": "call_xxx", "name": "compose_video", "arguments": {...}, "raw_arguments": "..."}
                        tc_for_llm = {
                            "id": tc.get("id", tc["name"]),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc.get("raw_arguments", json.dumps(tc.get("arguments", {}), ensure_ascii=False)),
                            }
                        }
                        brain.memory.add_message("assistant", None, tool_calls=[tc_for_llm])
                        brain.memory.add_message("tool", result_msg)
                        for result2 in brain.chat(""):
                            t2, c2 = result2["type"], result2["content"]
                            if t2 == "text":
                                response_text = c2
                            elif t2 == "error":
                                response_text = f"❌ {c2}"
                            elif t2 == "step":
                                console.print(f"[dim]{c2}[/dim]")
                            elif t2 == "tool_call":
                                console.print(f"[dim]🔧 {c2}[/dim]")
        except KeyboardInterrupt:
            response_text = "[yellow]⏹ 已取消[/yellow]"
            console.print(f"\n[yellow]⏹ 请求已取消[/yellow]")

        ui.add_message("ai", response_text or "[red]无回复[/red]")
        ui.set_status("idle")

        # ── 用户确认产出 → 写入 RAG ──
        # 当用户说"可以了"/"确认"/"保存"等关键词时，将 AI 产出保存到知识库
        _confirm_keywords = ["可以了", "确认", "没问题", "就这样", "保存", "好的", "好", "行", "挺好", "不错"]
        if response_text and any(kw in text_for_keywords.lower() for kw in _confirm_keywords):
            brain.memory.save_user_knowledge(
                content=f"用户需求: {text_for_keywords}\n\nAI产出: {response_text}",
                tags=["user_output", "confirmed"]
            )
            console.print("[dim]📚 已保存本次产出到知识库[/dim]")

        # 同步 token 用量
        ui.total_prompt_tokens = brain.total_prompt_tokens
        ui.total_completion_tokens = brain.total_completion_tokens

    brain.memory.clear_session()


def _pause() -> None:
    """暂停等待用户按 Enter"""
    console.print("\n[dim]按 Enter 继续...[/dim]")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
