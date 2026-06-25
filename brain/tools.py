"""
内容工坊工具注册与执行系统
================================
ToolRegistry: 管理所有可被 AI 调用的工具
提供 OpenAI-compatible Function Calling 定义
"""

import json
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ======================================================================
# 内置工具桩（stub handlers）
# 真实实现在 skill 模块中注入覆盖
# ======================================================================

def _stub_clip_videos(args: dict) -> tuple[str, dict]:
    """剪辑视频 — 桩"""
    paths = args.get("paths", [])
    duration = args.get("duration", 15.0)
    logger.info("[Tool:clip_videos] paths=%s, duration=%s", paths, duration)
    return (
        f"[STUB] 已接收剪辑请求：{len(paths)} 个视频，每段 {duration}s",
        {"status": "ok", "tool": "clip_videos", "paths": paths, "duration": duration},
    )


def _stub_vision_analyze(args: dict) -> tuple[str, dict]:
    """Kimi 视觉分析 — 桩"""
    paths = args.get("paths", [])
    logger.info("[Tool:vision_analyze] paths=%s", paths)
    return (
        f"[STUB] 已接收视觉分析请求：{len(paths)} 张图片",
        {"status": "ok", "tool": "vision_analyze", "paths": paths},
    )


def _stub_generate_copy(args: dict) -> tuple[str, dict]:
    """生成文案 — 桩"""
    topic = args.get("topic", "")
    context = args.get("context", "")
    visual_context = args.get("visual_context", "")
    logger.info("[Tool:generate_copy] topic=%s", topic)
    return (
        f"[STUB] 已为「{topic}」生成文案（长度：{len(context) + len(visual_context)} 字符上下文）",
        {
            "status": "ok",
            "tool": "generate_copy",
            "topic": topic,
            "copy_preview": f"这是关于「{topic}」的小红书种草文案……",
        },
    )


def _stub_compose_video(args: dict) -> tuple[str, dict]:
    """最终视频合成 — 桩"""
    clips = args.get("clips", [])
    settings = args.get("settings", {})
    logger.info("[Tool:compose_video] clips=%s, settings=%s", len(clips), settings)
    return (
        f"[STUB] 已合成视频：{len(clips)} 个片段，分辨率 {settings.get('resolution', '1080p')}",
        {"status": "ok", "tool": "compose_video", "clip_count": len(clips), "settings": settings},
    )


def _stub_generate_card(args: dict) -> tuple[str, dict]:
    """生成推广卡片 — 桩"""
    text = args.get("text", "")
    style = args.get("style", {})
    logger.info("[Tool:generate_card] text_length=%s, style=%s", len(text), style)
    return (
        f"[STUB] 已生成卡片海报：文案 {len(text)} 字，风格 {style.get('theme', 'default')}",
        {"status": "ok", "tool": "generate_card", "text_length": len(text), "style": style},
    )


def _stub_search_web(args: dict) -> tuple[str, dict]:
    """网络搜索 — 桩"""
    query = args.get("query", "")
    logger.info("[Tool:search_web] query=%s", query)
    return (
        f"[STUB] 搜索「{query}」的结果（模拟数据）",
        {
            "status": "ok",
            "tool": "search_web",
            "query": query,
            "results": [{"title": f"关于「{query}」的搜索结果 1", "url": "https://example.com/1"}],
        },
    )


def _stub_synthesize_tts(args: dict) -> tuple[str, dict]:
    """文本转语音 — 桩"""
    text = args.get("text", "")
    voice = args.get("voice", "default")
    logger.info("[Tool:synthesize_tts] text_length=%s, voice=%s", len(text), voice)
    return (
        f"[STUB] 已将 {len(text)} 字文本转为语音（音色：{voice}）",
        {"status": "ok", "tool": "synthesize_tts", "text_length": len(text), "voice": voice},
    )


# 内置工具注册元数据
_BUILTIN_TOOLS: list[dict] = [
    {
        "name": "clip_videos",
        "description": "剪辑/拼接多个视频片段，可设置每段时长。支持裁剪、合并、调速等基础操作",
        "handler": _stub_clip_videos,
        "requires_confirm": False,
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "视频文件路径列表",
                },
                "duration": {
                    "type": "number",
                    "description": "每段视频的目标时长（秒），默认 15",
                    "default": 15.0,
                },
            },
            "required": ["paths"],
        },
    },
    {
        "name": "vision_analyze",
        "description": "调用 Kimi 视觉模型分析图片内容，可用于分析素材图片、截图等",
        "handler": _stub_vision_analyze,
        "requires_confirm": False,
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "待分析图片文件路径列表",
                },
            },
            "required": ["paths"],
        },
    },
    {
        "name": "generate_copy",
        "description": "根据主题与上下文生成小红书种草文案、标题、标签等文本内容",
        "handler": _stub_generate_copy,
        "requires_confirm": False,
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "文案主题或产品名称",
                },
                "context": {
                    "type": "string",
                    "description": "补充背景信息，如卖点、使用场景",
                },
                "visual_context": {
                    "type": "string",
                    "description": "视觉素材描述，用于文案与画面的配合",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "compose_video",
        "description": "将多个已剪辑的视频片段合成为最终视频，可配置分辨率、背景音乐、转场等",
        "handler": _stub_compose_video,
        "requires_confirm": True,
        "parameters": {
            "type": "object",
            "properties": {
                "clips": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "已剪辑好的视频片段路径列表",
                },
                "settings": {
                    "type": "object",
                    "description": "合成设置：resolution, bgm, transition, subtitle 等",
                    "properties": {
                        "resolution": {
                            "type": "string",
                            "description": "输出分辨率，如 1080p, 4k",
                        },
                        "bgm": {
                            "type": "string",
                            "description": "背景音乐文件路径",
                        },
                        "transition": {
                            "type": "string",
                            "description": "转场效果名称",
                        },
                        "subtitle": {
                            "type": "string",
                            "description": "字幕样式或 SRT 路径",
                        },
                    },
                },
            },
            "required": ["clips"],
        },
    },
    {
        "name": "generate_card",
        "description": "生成推广卡片/封面图，支持自定义文案和视觉风格",
        "handler": _stub_generate_card,
        "requires_confirm": True,
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "卡片上要展示的文案",
                },
                "style": {
                    "type": "object",
                    "description": "卡片视觉风格配置",
                    "properties": {
                        "theme": {
                            "type": "string",
                            "description": "主题色/风格，如 '清新', '复古', '科技'",
                        },
                        "font_size": {
                            "type": "integer",
                            "description": "正文字号",
                        },
                        "layout": {
                            "type": "string",
                            "description": "布局方式：'center', 'top', 'split'",
                        },
                    },
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "search_web",
        "description": "通过 SearXNG 搜索引擎检索网络信息，获取实时资讯、素材灵感或热点话题",
        "handler": _stub_search_web,
        "requires_confirm": False,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或自然语言问句",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "synthesize_tts",
        "description": "文本转语音（TTS），将文案转换为语音旁白，用于视频配音",
        "handler": _stub_synthesize_tts,
        "requires_confirm": False,
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要转为语音的文本内容",
                },
                "voice": {
                    "type": "string",
                    "description": "音色选择：'default', 'female_1', 'male_1', 'cute' 等",
                    "default": "default",
                },
            },
            "required": ["text"],
        },
    },
]


# ======================================================================
# ToolRegistry
# ======================================================================

class ToolRegistry:
    """
    工具注册表 —— 管理系统内所有可被 LLM 调用的工具。

    用法::

        registry = ToolRegistry()
        registry.register("my_tool", "描述", handler_func, params_schema)
        tools_def = registry.get_tools_for_llm()
        result = registry.execute("my_tool", {"arg1": "val1"})
    """

    def __init__(self):
        # _tools: dict[name -> ToolInfo]
        self._tools: dict[str, dict] = {}
        self._load_builtins()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        handler: Callable[[dict], tuple[str, dict]],
        parameters_schema: dict,
        *,
        requires_confirm: bool = False,
    ) -> None:
        """
        注册一个工具。

        Args:
            name: 工具名称（需唯一，用于 Function Calling）
            description: 工具描述，LLM 据此判断何时调用
            handler: 可调用对象，签名 ``fn(args: dict) -> (result: str, metadata: dict)``
            parameters_schema: OpenAI Function Calling 风格的 JSON Schema
            requires_confirm: 若为 True，AI 应询问用户确认后再执行
        """
        if name in self._tools:
            logger.warning("工具 '%s' 已被覆盖注册", name)

        self._tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "parameters_schema": parameters_schema,
            "requires_confirm": requires_confirm,
        }
        logger.debug("工具已注册: %s", name)

    def unregister(self, name: str) -> bool:
        """注销一个工具，返回是否成功。"""
        if name in self._tools:
            del self._tools[name]
            logger.debug("工具已注销: %s", name)
            return True
        return False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict]:
        """
        返回所有已注册工具的名称与描述。

        Returns:
            [{"name": "...", "description": "...", "requires_confirm": bool}, ...]
        """
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "requires_confirm": t["requires_confirm"],
            }
            for t in self._tools.values()
        ]

    def get_tool_info(self, name: str) -> Optional[dict]:
        """获取单个工具的完整信息，不存在时返回 None。"""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """检查工具是否已注册。"""
        return name in self._tools

    # ------------------------------------------------------------------
    # OpenAI-compatible tool definitions (for Function Calling API)
    # ------------------------------------------------------------------

    def get_tools_for_llm(self) -> list[dict]:
        """
        返回 OpenAI-compatible 工具定义列表，可直接传入 ``tools=`` 参数。

        每条格式::

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}
                }
            }
        """
        result = []
        for t in self._tools.values():
            result.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters_schema"],
                },
            })
        return result

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, arguments: dict) -> dict:
        """
        执行已注册的工具。

        Args:
            tool_name: 工具名称
            arguments: 参数字典

        Returns::

            {
                "tool": tool_name,
                "success": True/False,
                "result": "人类可读的结果文本",
                "metadata": { ... },
                "requires_confirm": True/False,
            }

        工具 handler 抛出异常时，success=False，异常信息写入 metadata。
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return {
                "tool": tool_name,
                "success": False,
                "result": f"错误：未找到工具 '{tool_name}'",
                "metadata": {"error": f"Unknown tool: {tool_name}"},
                "requires_confirm": False,
            }

        handler = tool["handler"]
        try:
            result_text, metadata = handler(arguments)
            return {
                "tool": tool_name,
                "success": True,
                "result": result_text,
                "metadata": metadata,
                "requires_confirm": tool["requires_confirm"],
            }
        except Exception as e:
            logger.exception("工具 '%s' 执行异常", tool_name)
            return {
                "tool": tool_name,
                "success": False,
                "result": f"工具 '{tool_name}' 执行失败：{e}",
                "metadata": {"error": str(e), "exception_type": type(e).__name__},
                "requires_confirm": tool["requires_confirm"],
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_builtins(self) -> None:
        """加载内置工具定义。"""
        for t_def in _BUILTIN_TOOLS:
            self.register(
                name=t_def["name"],
                description=t_def["description"],
                handler=t_def["handler"],
                parameters_schema=t_def["parameters"],
                requires_confirm=t_def.get("requires_confirm", False),
            )


# ======================================================================
# 单例
# ======================================================================

_default_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """获取全局默认的 ToolRegistry 单例。"""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


# ======================================================================
# 辅助函数
# ======================================================================

def parse_tool_calls(response: dict) -> list[dict]:
    """
    从 LLM 响应中提取所有工具调用（function calling）指令。

    Args:
        response: ``chat()`` 方法返回的响应字典，格式::

            {
                "role": "assistant",
                "content": "...",
                "tool_calls": [
                    {
                        "id": "call_xxx",
                        "type": "function",
                        "function": {"name": "...", "arguments": "..."}
                    }
                ]
            }

    Returns:
        解析后的工具调用列表，每条结构::

            {
                "id": "call_xxx",
                "name": "工具名称",
                "arguments": {"arg1": "val1", ...},   # 已 parse 为 dict
                "raw_arguments": '{"arg1": "val1"}',   # 原始 JSON 字符串
            }

        如果 ``response`` 中没有 ``tool_calls``，返回空列表 ``[]``。
    """
    tool_calls_raw = response.get("tool_calls", [])
    if not tool_calls_raw:
        return []

    parsed = []
    for tc in tool_calls_raw:
        if tc.get("type") != "function":
            continue

        func = tc.get("function", {})
        name = func.get("name", "")
        raw_args = func.get("arguments", "{}")

        # 尝试解析 JSON 参数字符串
        try:
            args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            logger.warning("工具 '%s' 的参数 JSON 解析失败，使用原始字符串: %s", name, raw_args)
            args_dict = {"_raw": raw_args}

        parsed.append({
            "id": tc.get("id", ""),
            "name": name,
            "arguments": args_dict,
            "raw_arguments": raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False),
        })

    return parsed


def format_tool_result(tool_name: str, result: dict) -> dict:
    """
    将工具执行结果格式化为 LLM 可消费的 ``tool`` role message。

    Args:
        tool_name: 工具名称
        result: ``execute()`` 返回的结果字典

    Returns::

        {
            "role": "tool",
            "tool_call_id": tool_name,   # 简化处理，生产环境应传真实 call_id
            "content": json.dumps(result, ensure_ascii=False)
        }

    也可通过 ``content_format="text"`` 控制返回纯文本格式。
    """
    return {
        "role": "tool",
        "tool_call_id": tool_name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def execute_tool_calls(
    registry: ToolRegistry,
    tool_calls: list[dict],
) -> list[dict]:
    """
    便捷函数：批量解析并执行 LLM 下发的工具调用。

    Args:
        registry: ToolRegistry 实例
        tool_calls: ``parse_tool_calls()`` 返回的列表

    Returns:
        每条工具调用对应的 ``tool`` role message 列表，
        可直接追加到 ``messages`` 中返回给 LLM。
    """
    results = []
    for call in tool_calls:
        tool_name = call["name"]
        args = call["arguments"]
        call_id = call.get("id", tool_name)

        exec_result = registry.execute(tool_name, args)
        results.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(exec_result, ensure_ascii=False),
        })
    return results
