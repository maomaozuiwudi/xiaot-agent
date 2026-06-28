"""
AI 大脑 v2 — 多用户上下文 + 共享记忆
"""
import time
import json
import logging
from typing import Optional, Generator, Union

from config_loader import get


logger = logging.getLogger(__name__)
from knowledge.rag import RAGEngine
from knowledge.memory import UserManager, MemoryManager
from evolution import AestheticEvolution, ClipRuleEvolution
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
        self.rag = RAGEngine(
            shared_repo=get("knowledge.shared.repo", ""),
            shared_enabled=get("knowledge.shared.enabled", False),
        )

        # 进化引擎（越用越懂）
        self.evolution = AestheticEvolution()
        self.clip_evolution = ClipRuleEvolution()

        # 把 RAG 注入进化引擎和记忆引擎
        self.evolution.set_rag_engine(self.rag)
        self.clip_evolution.set_rag_engine(self.rag)
        self.memory.set_rag_engine(self.rag)

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

        # ── MCP 客户端（可选，有配置才初始化） ──
        self.mcp_manager = None
        mcp_servers = get("mcp_servers", {})
        if mcp_servers:
            try:
                from brain.mcp_client import MCPManager, make_mcp_handler
                self.mcp_manager = MCPManager(mcp_servers)
                self.mcp_manager.connect_all()
                if self.mcp_manager.is_connected():
                    mcp_tools = self.mcp_manager.get_all_tools()
                    for tool in mcp_tools:
                        srv_name = tool["server_name"]
                        org_name = tool["name"]
                        full_name = f"mcp_{srv_name}_{org_name}"
                        # 生成一份简短的参数描述
                        props = tool.get("inputSchema", {}).get("properties", {})
                        prop_lines = ", ".join(
                            f"{k}: {v.get('type', 'any')}"
                            for k, v in props.items()
                        )
                        param_desc = f"参数: {prop_lines}" if prop_lines else "无参数"
                        self.tool_registry.register(
                            name=full_name,
                            description=tool.get("description", "") or f"MCP 工具: {full_name}. {param_desc}",
                            handler=make_mcp_handler(self.mcp_manager, srv_name, org_name),
                            parameters_schema=tool.get("inputSchema", {"type": "object", "properties": {}}),
                            requires_confirm=False,
                        )
                    logger.info(
                        "已注册 %d 个 MCP 工具（来自 %d 个 server）",
                        len(mcp_tools), len(self.mcp_manager.connected_servers()),
                    )
            except Exception as e:
                logger.warning("MCP 客户端初始化失败: %s", e)

        # 懒加载小红书搜索工具
        try:
            from brain.tools import _ensure_xhs_tools
            _ensure_xhs_tools()
            # 传递 RAG 引擎给 XHS 搜索工具，使搜索结果自动注入知识库
            import importlib
            xhs_mod = importlib.import_module(
                "skills.xhs-content-factory.tools.xhs_search"
            )
            xhs_mod.set_rag_engine(self.rag)
        except ImportError:
            pass

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

        # Token 用量追踪
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.context_window = get("llm.context_window", 1000000)

    def _build_system_prompt(self) -> str:
        mode_desc = "你正在共享模式下运行。用户的使用习惯会匿名汇入知识库，也会从社区积累的知识中受益。" \
            if self.user_mgr.is_shared_mode() else \
            "你正在私有模式下运行。所有数据仅限本地，不共享不联网。"

        prompt = f"""你是小红书内容工坊 AI 助手，一个专业的小红书内容创作智能体。

{mode_desc}

## 核心原则
1. **诚实优先**：不知道就说不知道，没有的数据不要编造
2. **主动批评**：用户要求不合理时，先指出问题再问是否继续
3. **剪辑直接执行**：收到视频素材主动剪辑，不用问用户
4. **知识优先级**（按权重从高到低）：
   - 🥇 **当前对话上下文（1M 窗口）** — 用户刚说/刚给的素材，最优先
   - 🥈 **用户本地偏好/习惯** — 越用越懂，你的核心了解
   - 🥉 **爆款数据库 / RAG 参考库** — 标记为 xhs_search/viral 的内容及喂入的规则/风格
   - 🎯 **网络搜索** — 实时信息，仅作微调补充

## 执行原则
收到视频素材/做视频需求时，根据情况自主决定调用哪些工具：
- 需要参考 → 调 search_web / search_xhs 搜爆款参考
- 有视频素材 → 调 clip_videos 骨架智能剪辑
- 需要分析画面 → 调 vision_analyze 分析内容
- 需要文案 → 调 generate_copy 生成文案，展示给用户确认
- 合视频 → 调 compose_video（默认竖屏 1080×1920），展示给用户确认
- 配音 → 调 synthesize_tts

不用按固定顺序，根据当前情况判断下一步做什么。一次最多调1个工具，等结果回来再决定。

## 工具使用规则
- 可用工具：vision_analyze（分析图片/视频画面内容）、clip_videos（视频剪辑）、compose_video（视频合成）、generate_copy（生成文案）、synthesize_tts（语音合成）、xhs_search（小红书搜索）、search_web（网络搜索）
- vision_analyze 可分析图片和视频的画面细节（人物、服装、颜色、场景、动作等）。当用户拖入/下发素材文件时，你必须第一时间主动调用 vision_analyze 工具来分析画面内容，**不要反问用户"内容是什么"**。调用时传入文件完整路径作为 paths 参数。
- clip_videos 可以直接执行，不需要询问用户。收到视频路径后立即调用，传入 paths（视频路径列表）和 duration（每段目标时长秒数）。clip_videos 返回结果中的 clip_paths 就是剪辑后的视频路径，直接用于后续合成。
- compose_video 的 clips 参数传入 clip_videos 返回的 clip_paths 列表，不要自己构造路径。
- generate_copy 的 storyboard 参数：传入每镜的画面描述+目标时长（秒），DeepSeek 会根据总时长控制配音文案字数（正常语速约 3-4 字/秒），确保文案长度刚好匹配视频时长。
- 一次最多调用 1 个工具，等结果回来再决定下一步
- 工具结果是事实，不要重新解释
- 文案和合成需要停下来等用户确认，剪辑和视觉分析不需要

## 输出
- 用中文，每个步骤用 emoji 开头
- 标注置信度：📖有数据 | 💡有参考 | ⚠️AI推测"""
        return self.hallucination_guard.force_honest(prompt)

    # ── 核心对话 ──

    def chat(self, user_input: Union[str, list[dict]]) -> Generator[dict, None, None]:
        if not self.running:
            yield {"type": "error", "content": "AI 已停止"}
            return

        # 提取文本部分（多模态消息中仅文本用于检索/审查，完整消息体原样传给 LLM）
        if isinstance(user_input, list):
            user_text = "".join(
                part.get("text", "") for part in user_input if part.get("type") == "text"
            )
        else:
            user_text = user_input

        self.memory.add_message("user", user_input)
        yield {"type": "step", "content": "🔍 分析需求中..."}

        critique = self._pre_check(user_text)
        if not critique.passes:
            yield {"type": "step", "content": self._format_critique(critique)}

        yield {"type": "step", "content": "📚 检索知识库..."}
        context = self._augment_context(user_text)
        messages = self._build_messages(context)

        max_rounds = 5
        for round_idx in range(max_rounds):
            try:
                response = self.provider.chat(messages, tools=self.tool_registry.get_tools_for_llm())
            except Exception as e:
                yield {"type": "error", "content": f"调用失败: {e}"}
                return

            # 累加 token 用量
            usage = response.get("usage")
            if usage:
                self.total_prompt_tokens += usage.get("prompt", 0)
                self.total_completion_tokens += usage.get("completion", 0)

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
                    result_msg = format_tool_result(tc["name"], result, tool_call_id=tc["id"])
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

    def chat_stream(self, user_input: Union[str, list[dict]]) -> Generator[str, None, None]:
        """流式对话"""
        if not self.running:
            return

        # 提取文本部分（多模态消息）
        if isinstance(user_input, list):
            user_text = "".join(
                part.get("text", "") for part in user_input if part.get("type") == "text"
            )
        else:
            user_text = user_input

        self.memory.add_message("user", user_input)
        context = self._augment_context(user_text)
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
        # 简短问候/开始指令跳过检查
        _greetings = {"开始", "开始吧", "你好", "嗨", "hi", "hello", "🐱", "🐱 开始", "🐱 开始吧"}
        clean = user_input.strip().lower()
        if clean in _greetings or len(clean) <= 4:
            return CritiqueResult(is_feasible=True)
        request_type = "copywriting"
        if any(kw in user_input for kw in ["视频", "剪辑", "出片", "合成"]):
            request_type = "clip"
            # 用户输入中可能已包含视频路径或时长信息
            import re as _re
            has_paths = bool(_re.search(r'([a-zA-Z]:[/\\]|/e/|\.mp4|\.mov|\.avi)', user_input))
            has_duration = bool(_re.search(r'\d+\s*秒', user_input))
            params = {"input": user_input}
            if has_paths:
                params["materials"] = user_input
            if has_duration:
                dur_match = _re.search(r'(\d+)\s*秒', user_input)
                if dur_match:
                    params["duration"] = float(dur_match.group(1))
            return self.critic.review(request_type, params)
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
