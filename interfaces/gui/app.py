"""
🐱 小t Agent — Desktop GUI 客户端
基于 CustomTkinter，深蓝暗黑主题，流式对话
"""
import sys
import os
import queue
import threading
from pathlib import Path

# ── 确保能导入项目模块 ──
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import customtkinter as ctk
from config_loader import load_config, get as cfg_get
from brain.agent import AgentBrain
from knowledge.memory import UserManager

# ── 配色 ──
COLOR_BG_DARK = "#1a1a2e"
COLOR_BG_MID = "#16213e"
COLOR_BG_LIGHT = "#0f3460"
COLOR_ACCENT = "#e94560"
COLOR_TEXT = "#eaeaea"
COLOR_TEXT_DIM = "#8899aa"
COLOR_USER_BUBBLE = "#0f3460"
COLOR_AI_BUBBLE = "#1a2744"
COLOR_INPUT_BG = "#0d1b2a"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class SettingsDialog(ctk.CTkToplevel):
    """设置弹窗：模型选择 + API Key 输入"""

    def __init__(self, parent, current_model="", current_api_key="", on_save=None):
        super().__init__(parent)
        self.on_save = on_save
        self.title("⚙️ 设置")
        self.geometry("480x320")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        # 居中
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 480, 320
        self.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        self.configure(fg_color=COLOR_BG_DARK)

        ctk.CTkLabel(self, text="🔧 模型设置", font=("Microsoft YaHei", 16, "bold"),
                     text_color=COLOR_TEXT).pack(pady=(20, 10))

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(padx=30, fill="both", expand=True)

        # 模型选择
        ctk.CTkLabel(frame, text="模型提供商", anchor="w",
                     font=("Microsoft YaHei", 13), text_color=COLOR_TEXT_DIM).pack(fill="x", pady=(5, 2))
        self.model_var = ctk.StringVar(value=current_model or "deepseek-chat")
        model_menu = ctk.CTkOptionMenu(
            frame, variable=self.model_var,
            values=[
                "deepseek-chat", "deepseek-reasoner",
                "gpt-4o", "gpt-4o-mini",
                "kimi-k2.5", "qwen-max",
                "claude-sonnet-4-20250514",
            ],
            fg_color=COLOR_BG_LIGHT, button_color=COLOR_ACCENT,
            button_hover_color="#c73e54",
            dropdown_fg_color=COLOR_BG_MID,
            text_color=COLOR_TEXT,
            font=("Microsoft YaHei", 12),
        )
        model_menu.pack(fill="x", pady=(0, 10))

        # API Key 输入
        ctk.CTkLabel(frame, text="API Key", anchor="w",
                     font=("Microsoft YaHei", 13), text_color=COLOR_TEXT_DIM).pack(fill="x", pady=(5, 2))
        self.key_entry = ctk.CTkEntry(
            frame, placeholder_text="sk-...",
            show="*",
            fg_color=COLOR_INPUT_BG, text_color=COLOR_TEXT,
            font=("Consolas", 12),
        )
        self.key_entry.pack(fill="x", pady=(0, 5))
        if current_api_key:
            self.key_entry.insert(0, current_api_key)

        # Base URL
        ctk.CTkLabel(frame, text="Base URL（可选）", anchor="w",
                     font=("Microsoft YaHei", 13), text_color=COLOR_TEXT_DIM).pack(fill="x", pady=(5, 2))
        self.url_entry = ctk.CTkEntry(
            frame, placeholder_text="https://api.deepseek.com",
            fg_color=COLOR_INPUT_BG, text_color=COLOR_TEXT,
            font=("Consolas", 11),
        )
        self.url_entry.pack(fill="x", pady=(0, 5))
        self.url_entry.insert(0, cfg_get("llm.base_url", "https://api.deepseek.com"))

        # 保存按钮
        save_btn = ctk.CTkButton(
            frame, text="💾 保存",
            command=self._on_save,
            fg_color=COLOR_ACCENT, hover_color="#c73e54",
            text_color="white", font=("Microsoft YaHei", 14, "bold"),
        )
        save_btn.pack(pady=(15, 0))

    def _on_save(self):
        model = self.model_var.get()
        api_key = self.key_entry.get().strip()
        base_url = self.url_entry.get().strip()
        if self.on_save:
            self.on_save(model, api_key, base_url)
        self.destroy()


class ChatBubble(ctk.CTkFrame):
    """单条消息气泡"""

    def __init__(self, master, text, is_user=False, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.is_user = is_user

        # 内部容器（控制宽度和对齐）
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="x", padx=10, pady=3)

        # 头像标签
        avatar = "🐱" if not is_user else "👤"
        role_text = ctk.CTkLabel(
            inner, text=f"{avatar}  {'你' if is_user else '小t'}",
            font=("Microsoft YaHei", 11, "bold"),
            text_color=COLOR_ACCENT if is_user else "#5dade2",
            anchor="e" if is_user else "w",
        )
        role_text.pack(fill="x", padx=(5, 5), pady=(2, 0))

        # 气泡
        bubble_frame = ctk.CTkFrame(
            inner,
            fg_color=COLOR_USER_BUBBLE if is_user else COLOR_AI_BUBBLE,
            corner_radius=12,
        )
        bubble_frame.pack(fill="x", pady=(0, 2))

        self._label = ctk.CTkLabel(
            bubble_frame, text=text,
            font=("Microsoft YaHei", 13),
            text_color=COLOR_TEXT,
            justify="left",
            wraplength=520,
            anchor="w",
            padx=12, pady=8,
        )
        self._label.pack(fill="x")

    def update_text(self, text):
        """流式更新气泡内容"""
        self._label.configure(text=text)


class ChatArea(ctk.CTkScrollableFrame):
    """可滚动的聊天区域"""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=COLOR_BG_DARK, **kwargs)
        self._bubbles = []
        self._current_ai_bubble = None

        # 欢迎消息
        self._add_welcome()

    def _add_welcome(self):
        welcome = (
            "🐱 你好！我是小t，你的小红书内容创作助手！\n\n"
            "我可以帮你：\n"
            "  • 🎬 剪辑视频素材\n"
            "  • ✍️ 生成文案\n"
            "  • 🔍 搜索小红书爆款参考\n"
            "  • 🖼️ 分析图片/视频内容\n"
            "  • 🎤 语音合成\n\n"
            "有什么需要帮忙的吗？直接发消息给我~"
        )
        self.add_bubble(welcome, is_user=False)

    def add_bubble(self, text, is_user=False):
        """添加新消息气泡"""
        bubble = ChatBubble(self, text, is_user=is_user)
        bubble.pack(fill="x", pady=2,
                    anchor="e" if is_user else "w")
        self._bubbles.append(bubble)

        if not is_user:
            self._current_ai_bubble = bubble
        else:
            self._current_ai_bubble = None

        # 滚动到底部
        self.after(20, self._scroll_to_bottom)
        return bubble

    def update_last_ai(self, text):
        """更新最后一条 AI 气泡（流式追加）"""
        if self._current_ai_bubble is None:
            self.add_bubble("", is_user=False)
        if self._current_ai_bubble:
            self._current_ai_bubble.update_text(text)
            self.after(20, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        try:
            self._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def clear(self):
        """清空聊天记录"""
        for b in self._bubbles:
            b.destroy()
        self._bubbles.clear()
        self._current_ai_bubble = None
        self._add_welcome()


class App(ctk.CTk):
    """主窗口"""

    def __init__(self):
        super().__init__()

        # ── 窗口属性 ──
        self.title("🐱 小t Agent")
        self.geometry("800x600")
        self.minsize(640, 480)
        self.configure(fg_color=COLOR_BG_DARK)

        # ── 状态 ──
        self._brain = None
        self._stream_queue = queue.Queue()
        self._streaming = False
        self._current_response = ""  # 当前流式回复累积

        # 从 config 读取
        self._api_key = cfg_get("llm.api_key", "")
        self._model = cfg_get("llm.model", "deepseek-chat")
        self._base_url = cfg_get("llm.base_url", "https://api.deepseek.com")

        # ── 初始化 Agent ──
        self._init_agent()

        # ── UI 构建 ──
        self._build_ui()

        # ── 键盘绑定 ──
        self.bind("<Control-Return>", lambda e: self._send_message())

        # ── 流式轮询 ──
        self.after(100, self._poll_stream)

        # 关闭时的清理
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── 初始化 ──

    def _init_agent(self):
        """初始化或重新初始化 AgentBrain"""
        try:
            load_config()
        except FileNotFoundError:
            pass

        try:
            user_mgr = UserManager()
            if self._api_key:
                user_mgr.login(self._api_key)
            self._brain = AgentBrain(user_mgr)
        except Exception as e:
            print(f"[GUI] 初始化 Agent 失败: {e}")
            self._brain = None

    # ── UI 构建 ──

    def _build_ui(self):
        """构建完整界面"""
        # 顶部工具栏
        self._build_toolbar()

        # 聊天区域（主内容）
        self.chat_area = ChatArea(self, scrollbar_button_color=COLOR_BG_LIGHT)
        self.chat_area.pack(fill="both", expand=True, padx=0, pady=0)

        # 底部输入区
        self._build_input_area()

    def _build_toolbar(self):
        """工具栏：[新建对话] [设置]"""
        toolbar = ctk.CTkFrame(self, fg_color=COLOR_BG_MID, height=44)
        toolbar.pack(fill="x", padx=0, pady=0)
        toolbar.pack_propagate(False)

        # 左侧标题
        title = ctk.CTkLabel(
            toolbar, text="🐱 小t — 小红书内容工坊",
            font=("Microsoft YaHei", 15, "bold"),
            text_color=COLOR_TEXT,
        )
        title.pack(side="left", padx=(12, 0), pady=0)

        # 右侧按钮
        btn_frame = ctk.CTkFrame(toolbar, fg_color="transparent")
        btn_frame.pack(side="right", padx=(0, 8))

        new_btn = ctk.CTkButton(
            btn_frame, text="🆕 新建对话",
            command=self._new_chat,
            fg_color="transparent", hover_color=COLOR_BG_LIGHT,
            text_color=COLOR_TEXT, font=("Microsoft YaHei", 12),
            width=100, height=30,
            border_width=1, border_color=COLOR_BG_LIGHT,
        )
        new_btn.pack(side="left", padx=4)

        settings_btn = ctk.CTkButton(
            btn_frame, text="⚙️ 设置",
            command=self._open_settings,
            fg_color="transparent", hover_color=COLOR_BG_LIGHT,
            text_color=COLOR_TEXT, font=("Microsoft YaHei", 12),
            width=80, height=30,
            border_width=1, border_color=COLOR_BG_LIGHT,
        )
        settings_btn.pack(side="left", padx=4)

    def _build_input_area(self):
        """底部输入框 + 发送按钮"""
        bottom = ctk.CTkFrame(self, fg_color=COLOR_BG_MID, height=80)
        bottom.pack(fill="x", padx=0, pady=0)
        bottom.pack_propagate(False)

        inner = ctk.CTkFrame(bottom, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=12, pady=8)

        # 输入框
        self.input_text = ctk.CTkTextbox(
            inner,
            fg_color=COLOR_INPUT_BG, text_color=COLOR_TEXT,
            font=("Microsoft YaHei", 13),
            border_width=0,
            corner_radius=10,
            height=50,
            wrap="word",
        )
        self.input_text.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.input_text.bind("<Control-Return>", lambda e: self._send_message())

        # 发送按钮
        self.send_btn = ctk.CTkButton(
            inner, text="📤 发送",
            command=self._send_message,
            fg_color=COLOR_ACCENT, hover_color="#c73e54",
            text_color="white", font=("Microsoft YaHei", 14, "bold"),
            width=80, height=50,
            corner_radius=10,
        )
        self.send_btn.pack(side="right")

        # 提示标签
        hint = ctk.CTkLabel(
            bottom, text="Ctrl+Enter 发送",
            font=("Microsoft YaHei", 10),
            text_color=COLOR_TEXT_DIM,
        )
        hint.pack(after=inner, side="bottom", pady=(0, 3))

    # ── 操作 ──

    def _new_chat(self):
        """新建对话"""
        if self._streaming:
            return
        if self._brain:
            self._brain.reset()
        self.chat_area.clear()
        self._current_response = ""

    def _open_settings(self):
        """打开设置弹窗"""
        SettingsDialog(
            self,
            current_model=self._model,
            current_api_key=self._api_key,
            on_save=self._apply_settings,
        )

    def _apply_settings(self, model, api_key, base_url):
        """应用设置并重启 Agent"""
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

        # 写入 config.yaml
        try:
            import yaml
            cfg_path = Path(_PROJECT_ROOT) / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg.setdefault("llm", {})
                cfg["llm"]["model"] = model
                if api_key:
                    cfg["llm"]["api_key"] = api_key
                if base_url:
                    cfg["llm"]["base_url"] = base_url
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True,
                              default_flow_style=False, sort_keys=False)
                # 重新加载配置
                load_config()
        except Exception as e:
            print(f"[GUI] 写入配置失败: {e}")

        # 刷新 Agent
        self._init_agent()
        self.chat_area.add_bubble(f"🔄 已切换模型: {model}", is_user=False)

    def _send_message(self):
        """发送消息"""
        if self._streaming:
            return

        text = self.input_text.get("0.0", "end").strip()
        if not text:
            return

        # 清空输入框
        self.input_text.delete("0.0", "end")

        # 显示用户消息
        self.chat_area.add_bubble(text, is_user=True)

        # 开始流式生成
        self._start_stream(text)

    def _start_stream(self, text):
        """在后台线程中启动流式对话"""
        if not self._brain:
            self.chat_area.add_bubble("⚠️ Agent 未初始化，请检查设置中的 API Key", is_user=False)
            return

        self._streaming = True
        self._current_response = ""
        self.send_btn.configure(state="disabled", text="⏳...")

        # 先显示 AI 气泡占位
        self.chat_area.add_bubble("", is_user=False)

        def _run():
            try:
                for chunk in self._brain.chat_stream(text):
                    self._stream_queue.put(("token", chunk))
                self._stream_queue.put(("done", None))
            except Exception as e:
                self._stream_queue.put(("error", str(e)))

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _poll_stream(self):
        """轮询流式队列（由主线程 after 调度）"""
        try:
            while True:
                msg_type, data = self._stream_queue.get_nowait()

                if msg_type == "token":
                    self._current_response += data
                    self.chat_area.update_last_ai(self._current_response)

                elif msg_type == "error":
                    error_msg = f"❌ {data}"
                    self.chat_area.update_last_ai(error_msg)
                    self._streaming = False
                    self._reset_send_btn()

                elif msg_type == "done":
                    self._streaming = False
                    self._reset_send_btn()
                    # 保存到对话历史
                    if self._current_response.strip() and self._brain:
                        self._brain.memory.add_message(
                            "assistant", self._current_response
                        )

        except queue.Empty:
            pass

        # 继续轮询
        self.after(50, self._poll_stream)

    def _reset_send_btn(self):
        """恢复发送按钮状态"""
        self.send_btn.configure(state="normal", text="📤 发送")

    def _on_close(self):
        """窗口关闭清理"""
        self._streaming = False
        if self._brain:
            try:
                self._brain.memory.clear_session()
            except Exception:
                pass
        self.destroy()


def run():
    """启动 GUI 入口"""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    run()
