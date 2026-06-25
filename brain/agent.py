"""
AI 大脑 v2 — 多用户上下文 + 共享记忆
"""
import time
import json
from typing import Optional, Generator

from config_loader import get
from knowledge.rag import RAGEngine
from knowledge.memory import UserManager, MemoryManager
from brain.providers import get_provider
from brain.tools import get_registry, parse_tool_calls, format_tool_result
from guard.critic import Critic, CritiqueResult
from guard.hallucination import HallucinationGuard


class AgentBrain:
    """AI 大脑 — 对话 + 推理 + 工具调度"""

    def __init__(self, user_mgr: UserManager = None):
        # 用户上下文
        self.user_mgr = user_mgr or UserManager()
        self.memory = MemoryManager(self.user_mgr)
        self.rag = RAGEngine()

        # 共享 RAG（如果开启共享模式）
        if self.user_mgr.is_shared_mode():
            self.shared_rag = RAGEngine(
                sources_dir=get("user.shared_dir", "data/shared/") + "/rag/"
            )
        else:
            self.shared_rag = None

        self.tool_registry = get_registry()
        self.critic = Critic()
        self.hallucination_guard = HallucinationGuard()

        # 用用户的 Key 初始化 Provider
        api_key = self.user_mgr.api_key
        vision_key = get("llm.vision.api_key", "")
        self.provider = get_provider({
            "provider": get("llm.provider", "openai_compat"),
            "model": get("llm.model", "deepseek-chat"),
            "api_key": api_key,
            "base_url": get("llm.base_url", "https://api.deepseek.com"),
            "temperature": get("llm.temperature", 0.3),
            "max_tokens": get("llm.max_tokens", 8192),
            "context_window": get("llm.context_window", 1000000),
        })

        self.system_prompt = self._build_system_prompt()
        self.running = True
        self.current_skill = None

    def _build_system_prompt(self) -> str:
        mode_desc = "你正在共享模式下运行。用户的使用习惯会匿名汇入知识库，也会从社区积累的知识中受益。" \
            if self.user_mgr.is_shared_mode() else \
            "你正在私有模式下运行。所有数据仅限本地，不共享不联网。"

        prompt = f"""你是小红书内容工坊 AI 助手，一个专业的小红书内容创作智能体。

{mode_desc}

## 核心原则
1. **诚实优先**：不知道就说不知道，没有的数据不要编造
2. **主动批评**：用户要求不合理时，先指出问题再问是否继续
3. **确认制**：每个关键步骤（文案、合成）必须先问用户
4. **知识优先级**：上下文 > 参考库(RAG) > 网络检索

## 工作流
收到"做视频""出片"需求时：
1. 素材检查 → 2. 可行性评估 → 3. 骨架剪辑 → 4. 视觉分析 → 5. 文案生成(需确认) → 6. 配音/BGM → 7. 视频合成(需确认) → 8. 输出

## 工具使用规则
- 一次最多调用 1 个工具，等结果回来再决定下一步
- 工具结果是事实，不要重新解释
- 需要确认的步骤停下来等用户

## 输出
- 用中文，每个步骤用 emoji 开头
- 标注置信度：📖有数据 | 💡有参考 | ⚠️AI推测"""
        return self.hallucination_guard.force_honest(prompt)

    # ── 核心对话 ──

    def chat(self, user_input: str) -> Generator[dict, None, None]:
        if not self.running:
            yield {"type": "error", "content": "AI 已停止"}
            return

        self.memory.add_message("user", user_input)
        yield {"type": "step", "content": "🔍 分析需求中..."}

        critique = self._pre_check(user_input)
        if not critique.passes:
            yield {"type": "step", "content": self._format_critique(critique)}

        yield {"type": "step", "content": "📚 检索知识库..."}
        context = self._augment_context(user_input)
        messages = self._build_messages(context)

        max_rounds = 5
        for round_idx in range(max_rounds):
            try:
                response = self.provider.chat(messages, tools=self.tool_registry.get_tools_for_llm())
            except Exception as e:
                yield {"type": "error", "content": f"调用失败: {e}"}
                return

            assistant_msg = {
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": response.get("tool_calls", []),
            }
            messages.append(assistant_msg)
            self.memory.add_message("assistant", response.get("content", ""))

            tool_calls = response.get("tool_calls", [])
            if tool_calls:
                for tc in parse_tool_calls(response):
                    yield {"type": "tool_call", "content": f"🔧 调用: {tc['name']}", "metadata": tc}
                    tool_info = self.tool_registry.get_tool_info(tc["name"])
                    if tool_info and tool_info.get("requires_confirm"):
                        yield {"type": "confirm_needed", "content": tc}
                        return

                    result = self.tool_registry.execute(tc["name"], tc["arguments"])
                    result_msg = format_tool_result(tc["name"], result)
                    messages.append(result_msg)
                    self.memory.record_feedback("tool_usage", {
                        "tool": tc["name"], "args": tc["arguments"],
                        "success": result.get("success", False),
                    })
                continue

            content = response.get("content", "")
            if content:
                if get("hallucination_guard.confidence_marking", True):
                    anchors = self._collect_anchors()
                    content = self.hallucination_guard.annotate_source(content, anchors)
                yield {"type": "text", "content": content}
            break

        yield {"type": "done", "content": ""}

    def chat_stream(self, user_input: str) -> Generator[str, None, None]:
        """流式对话"""
        if not self.running:
            return

        self.memory.add_message("user", user_input)
        context = self._augment_context(user_input)
        messages = self._build_messages(context)

        max_rounds = 5
        for round_idx in range(max_rounds):
            try:
                stream = self.provider.chat_stream(messages, tools=self.tool_registry.get_tools_for_llm())
            except Exception as e:
                yield f"\n[错误] 调用失败: {e}"
                return

            collected_content = ""
            collected_tool_calls = []
            for chunk in stream:
                if chunk["type"] == "text":
                    collected_content += chunk["content"]
                    yield chunk["content"]
                elif chunk["type"] == "tool_call":
                    collected_tool_calls.append(chunk["content"])

            if collected_tool_calls:
                self.memory.add_message("assistant", collected_content)
                for tc_info in collected_tool_calls:
                    tc = {"name": tc_info.get("name", ""), "arguments": tc_info.get("arguments", {})}
                    yield f"\n[工具] 调用: {tc['name']}\n"
                    result = self.tool_registry.execute(tc["name"], tc["arguments"])
                    result_msg = format_tool_result(tc["name"], result)
                    messages.append(result_msg)
                    self.memory.record_feedback("tool_usage", {
                        "tool": tc["name"], "args": tc["arguments"],
                        "success": result.get("success", False),
                    })
                continue

            if collected_content:
                self.memory.add_message("assistant", collected_content)
            break

    # ── 内部 ──

    def _pre_check(self, user_input: str) -> CritiqueResult:
        if not get("critic.enabled", True):
            return CritiqueResult(is_feasible=True)
        request_type = "copywriting"
        if any(kw in user_input for kw in ["视频", "剪辑", "出片", "合成"]):
            request_type = "clip"
        elif any(kw in user_input for kw in ["照片", "图片", "封面", "卡片"]):
            request_type = "composition"
        return self.critic.review(request_type, {"input": user_input})

    def _augment_context(self, user_input: str) -> str:
        parts = []
        # 个人 RAG
        if get("knowledge.rag.auto_index", True):
            rag_results = self.rag.query(user_input, top_k=2)
            if rag_results:
                refs = [f"[来源: {r.title} ({r.category})]\n{r.content}" for r in rag_results]
                parts.append("【参考库资料】\n" + "\n\n".join(refs))

        # 共享 RAG（共享模式）
        if self.shared_rag:
            shared_results = self.shared_rag.query(user_input, top_k=2)
            if shared_results:
                refs = [f"[社区共享: {r.title}]\n{r.content}" for r in shared_results]
                parts.append("【社区共享经验】\n" + "\n\n".join(refs))

        # 共享趋势（共享模式）
        if self.user_mgr.is_shared_mode():
            trends = self.memory.get_shared_trends("prefs", top_k=3)
            if trends:
                trend_lines = []
                for t in trends:
                    trend_lines.append(f"  {t['key']}: {t['value']} ({t['count']}人)")
                parts.append("【社区趋势】\n" + "\n".join(trend_lines))

        # 个人偏好
        prefs = self.memory.get_all_prefs()
        if prefs:
            pref_lines = []
            for cat, data in prefs.items():
                items = [f"  {k}: {v.get('value', v) if isinstance(v, dict) else v}"
                        for k, v in data.items()]
                if items:
                    pref_lines.append(f"{cat}:\n" + "\n".join(items))
            if pref_lines:
                parts.append("【你的偏好】\n" + "\n".join(pref_lines))

        return "\n\n".join(parts)

    def _build_messages(self, extra_context: str = "") -> list:
        messages = [{"role": "system", "content": self.system_prompt}]
        if extra_context:
            messages.append({"role": "system", "content": f"## 上下文信息\n{extra_context}"})
        messages.extend(self.memory.get_context())
        return messages

    def _collect_anchors(self) -> dict:
        anchors = {}
        for msg in reversed(self.memory.short_term):
            if msg["role"] == "tool":
                try:
                    data = json.loads(msg["content"]) if isinstance(msg["content"], str) else msg["content"]
                    anchors["tool_result"] = data
                except Exception:
                    pass
                break
        return anchors

    def _format_critique(self, critique: CritiqueResult) -> str:
        lines = ["⚠️ 需求评估"]
        if critique.warnings:
            lines.extend(["\n警告："] + [f"  • {w}" for w in critique.warnings])
        if critique.risks:
            lines.extend(["\n风险："] + [f"  • {r}" for r in critique.risks])
        if critique.suggestions:
            lines.extend(["\n建议："] + [f"  • {s}" for s in critique.suggestions])
        return "\n".join(lines)

    def reset(self):
        self.memory.clear_session()

    def set_skill(self, skill_name: str):
        self.current_skill = skill_name

    def stop(self):
        self.running = False
