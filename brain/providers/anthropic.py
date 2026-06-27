"""
Anthropic/Claude Provider
使用 Anthropic Messages API（claude-sonnet-4-20250514 等）
"""
import json
import logging
import os
from typing import Optional, Generator

import anthropic
from anthropic import APIError, APITimeoutError, RateLimitError, AuthenticationError

from brain.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic/Claude LLM Provider"""

    def __init__(self, config: dict):
        super().__init__(config)

        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        base_url = self.base_url or "https://api.anthropic.com"

        self._api_key = api_key
        self._base_url = base_url
        self._client = anthropic.Anthropic(
            api_key=api_key or "no-api-key-provided",
            base_url=base_url,
        )

        if not api_key:
            logger.warning(
                "AnthropicProvider: No API key provided. "
                "Set ANTHROPIC_API_KEY env var or pass api_key in config."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def supports_tool_calling(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Public API - non-streaming chat
    # ------------------------------------------------------------------

    def chat(self, messages: list, tools: Optional[list] = None,
             temperature: Optional[float] = None) -> dict:
        """
        非流式对话。

        返回格式（统一）:
        {
            "role": "assistant",
            "content": "...",
            "tool_calls": [
                {
                    "id": "toolu_xxx",
                    "type": "function",
                    "function": {"name": "...", "arguments": "{...}"}
                }
            ],
            "usage": {"prompt": int, "completion": int, "total": int}
        }
        """
        system, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None

        kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        try:
            response = self._client.messages.create(**kwargs)
        except AuthenticationError as e:
            logger.error(f"Anthropic API 认证失败: {e}")
            raise
        except RateLimitError as e:
            logger.error(f"Anthropic API 速率限制: {e}")
            raise
        except APITimeoutError as e:
            logger.error(f"Anthropic API 请求超时: {e}")
            raise
        except APIError as e:
            logger.error(f"Anthropic API 错误: status={e.status_code}, body={e.body}")
            raise
        except Exception as e:
            logger.error(f"Anthropic 未知错误: {e}")
            raise

        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Public API - streaming chat
    # ------------------------------------------------------------------

    def chat_stream(self, messages: list, tools: Optional[list] = None,
                    temperature: Optional[float] = None) -> Generator[dict, None, None]:
        """
        流式对话，每次 yield 一个片段。

        片段格式:
        - 文本: {"type": "text", "content": "..."}
        - 工具调用: {"type": "tool_call", "content": tool_call_dict}
          tool_call_dict = {"id": "...", "type": "function",
                            "function": {"name": "...", "arguments": "..."}}
        """
        system, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None

        kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        kwargs["stream"] = True

        try:
            with self._client.messages.stream(**kwargs) as stream:
                # 处理流式事件
                pending_tool_calls: list[dict] = []
                for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            pending_tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": "",
                                },
                            })
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield {"type": "text", "content": delta.text}
                        elif delta.type == "input_json_delta":
                            if pending_tool_calls:
                                pending_tool_calls[-1]["function"]["arguments"] += delta.partial_json
                    elif event.type == "message_delta":
                        # 流结束，输出累积的工具调用
                        for tc in pending_tool_calls:
                            yield {"type": "tool_call", "content": tc}
                        pending_tool_calls.clear()
        except Exception as e:
            logger.error(f"Anthropic 流式请求错误: {e}")
            raise

    # ------------------------------------------------------------------
    # Internal helpers - message conversion
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list) -> tuple:
        """
        将统一消息格式转换为 Anthropic 格式。

        Anthropic 格式:
          - system: 独立字符串参数（提取所有 system 消息合并）
          - messages: list of {"role": "user"|"assistant",
                                "content": [content_block, ...]}

        统一格式 → Anthropic 格式转换规则:
          - system → 提取到 system 参数
          - user (str content) → user, content as text block
          - assistant (str content) → assistant, content as text block
          - assistant (str content + tool_calls) → assistant, text + tool_use blocks
          - tool → user, content as tool_result block
        """
        system_parts = []
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "")

            # --- system → 提取到独立参数 ---
            if role == "system":
                system_parts.append(str(msg.get("content", "")))
                continue

            # --- tool → 转为 user role 的 tool_result block ---
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                content = str(msg.get("content", ""))
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content,
                        }
                    ],
                })
                continue

            # --- user → 转为 content blocks ---
            if role == "user":
                raw_content = msg.get("content", "")
                if isinstance(raw_content, str):
                    anthropic_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": raw_content}],
                    })
                elif isinstance(raw_content, list):
                    # 已经是 content blocks 格式，直接透传
                    anthropic_messages.append({
                        "role": "user",
                        "content": raw_content,
                    })
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": str(raw_content)}],
                    })
                continue

            # --- assistant → 可能包含文本 + tool_calls ---
            if role == "assistant":
                content_blocks = []
                text_content = msg.get("content", "") or ""
                if text_content:
                    content_blocks.append({"type": "text", "text": text_content})

                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    raw_args = func.get("arguments", "{}")
                    # arguments 可能是 JSON 字符串或 dict
                    if isinstance(raw_args, str):
                        try:
                            args_dict = json.loads(raw_args)
                        except json.JSONDecodeError:
                            args_dict = {"_raw": raw_args}
                    else:
                        args_dict = raw_args

                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": name,
                        "input": args_dict,
                    })

                if content_blocks:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content_blocks,
                    })
                else:
                    # 空响应，给一个空文本块
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": ""}],
                    })
                continue

            # --- 其他角色（兜底）→ 原样传给 user ---
            anthropic_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": str(msg.get("content", ""))}],
            })

        system = "\n\n".join(system_parts) if system_parts else None
        return system, anthropic_messages

    @staticmethod
    def _convert_tools(tools: list) -> list:
        """将统一工具格式转为 Anthropic 工具格式"""
        converted = []
        for t in tools:
            func = t.get("function", t)
            converted.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            })
        return converted

    # ------------------------------------------------------------------
    # Internal helpers - response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response) -> dict:
        """解析 Anthropic Messages API 响应为统一格式"""
        result = {
            "role": "assistant",
            "content": "",
        }

        content_text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                })

        result["content"] = "".join(content_text_parts)
        if tool_calls:
            result["tool_calls"] = tool_calls

        # 提取 token 用量
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
            result["usage"] = {
                "prompt": input_tokens,
                "completion": output_tokens,
                "total": input_tokens + output_tokens,
            }

        return result
