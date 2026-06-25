"""
LLM Provider 抽象基类
所有模型适配器必须继承此基类
"""
from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    """LLM Provider 基类"""

    def __init__(self, config: dict):
        self.config = config
        self.model = config.get("model", "deepseek-chat")
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.deepseek.com")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 8192)
        self.context_window = config.get("context_window", 1000000)

    @abstractmethod
    def chat(self, messages: list, tools: Optional[list] = None,
             temperature: Optional[float] = None) -> dict:
        """
        对话接口
        返回: {"role": "assistant", "content": "...", "tool_calls": [...]}
        tool_calls 格式: [{"id": "...", "type": "function",
                          "function": {"name": "...", "arguments": "..."}}]
        """
        ...

    @abstractmethod
    def chat_stream(self, messages: list, tools: Optional[list] = None,
                    temperature: Optional[float] = None):
        """
        流式对话接口，yield 每个 token 片段
        每个片段: {"type": "text"|"tool_call", "content": "..."}
        """
        ...
        yield  # pragma: no cover

    @property
    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """是否支持 Function Calling"""
        ...

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """是否支持流式输出"""
        ...

    def format_tools_for_prompt(self, tools: list) -> str:
        """
        降级方案：对于不支持 tool calling 的模型，
        将工具定义转为纯文本描述嵌入 system prompt
        """
        lines = ["\n【可用工具】"]
        for t in tools:
            name = t["function"]["name"]
            desc = t["function"]["description"]
            params = t["function"]["parameters"]
            lines.append(f"\n## {name}")
            lines.append(f"描述: {desc}")
            if "properties" in params:
                for pname, pinfo in params["properties"].items():
                    required = "(必填)" if pname in params.get("required", []) else "(可选)"
                    lines.append(f"  - {pname} {required}: {pinfo.get('description', '')}")
        return "\n".join(lines)
