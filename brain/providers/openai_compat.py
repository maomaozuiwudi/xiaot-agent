"""
OpenAI 兼容协议 Provider
支持所有 OpenAI 兼容 API: DeepSeek, Kimi, Qwen, ZhiPu, GLM, Moonshot 等
"""
import json
import logging
import os
from typing import Optional, Generator

import openai
from openai import APIError, APITimeoutError, RateLimitError, AuthenticationError

from brain.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatProvider(BaseLLMProvider):
    """OpenAI 兼容协议 LLM Provider"""

    def __init__(self, config: dict):
        super().__init__(config)

        # 优先使用配置中的 api_key，否则回退到环境变量
        api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        # 如果没有 base_url，默认使用 DeepSeek
        base_url = self.base_url or "https://api.deepseek.com"

        # openai v2 client 不允许 api_key 为空字符串，
        # 但我们希望延迟报错到实际调用时，因此用占位符
        self._api_key = api_key
        self._base_url = base_url
        self._client = openai.OpenAI(
            api_key=api_key or "no-api-key-provided",
            base_url=base_url,
        )

        self._provider_name = config.get("provider", "openai_compat")

        if not api_key:
            logger.warning(
                "OpenAICompatProvider: No API key provided. "
                "Set DEEPSEEK_API_KEY env var or pass api_key in config."
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
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list, tools: Optional[list] = None,
             temperature: Optional[float] = None) -> dict:
        """
        非流式对话。

        返回格式:
        {
            "role": "assistant",
            "content": "模型回复文本",
            "tool_calls": [
                {
                    "id": "call_xxx",
                    "type": "function",
                    "function": {"name": "...", "arguments": "..."}
                }
            ]
        }
        """
        kwargs = self._build_kwargs(messages, tools, temperature)
        try:
            response = self._client.chat.completions.create(**kwargs)
        except AuthenticationError as e:
            logger.error(f"API 认证失败: {e}")
            raise
        except RateLimitError as e:
            logger.error(f"API 速率限制: {e}")
            raise
        except APITimeoutError as e:
            logger.error(f"API 请求超时: {e}")
            raise
        except APIError as e:
            logger.error(f"API 错误: status={e.status_code}, body={e.body}")
            raise
        except Exception as e:
            logger.error(f"未知错误: {e}")
            raise

        return self._parse_response(response)

    def chat_stream(self, messages: list, tools: Optional[list] = None,
                    temperature: Optional[float] = None) -> Generator[dict, None, None]:
        """
        流式对话，每次 yield 一个片段。

        片段格式:
        - 文本: {"type": "text", "content": "..."}
        - 工具调用: {"type": "tool_call", "content": json.dumps(tool_call_dict)}
          tool_call_dict = {"id": "...", "type": "function",
                            "function": {"name": "...", "arguments": "..."}}
        """
        kwargs = self._build_kwargs(messages, tools, temperature)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": False}

        try:
            stream = self._client.chat.completions.create(**kwargs)
        except AuthenticationError as e:
            logger.error(f"流式 API 认证失败: {e}")
            raise
        except RateLimitError as e:
            logger.error(f"流式 API 速率限制: {e}")
            raise
        except APITimeoutError as e:
            logger.error(f"流式 API 请求超时: {e}")
            raise
        except APIError as e:
            logger.error(f"流式 API 错误: status={e.status_code}, body={e.body}")
            raise
        except Exception as e:
            logger.error(f"流式未知错误: {e}")
            raise

        # 累积工具调用状态（流式 function calling 需要合并多个 delta）
        tool_call_buffer: dict[int, dict] = {}

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            finish_reason = chunk.choices[0].finish_reason

            # --- 文本 token ---
            if delta.content:
                yield {"type": "text", "content": delta.content}

            # --- 工具调用 delta（流式 function calling） ---
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    index = tc_delta.index
                    if index not in tool_call_buffer:
                        tool_call_buffer[index] = {
                            "id": tc_delta.id or "",
                            "type": "function",
                            "function": {
                                "name": tc_delta.function.name or "",
                                "arguments": tc_delta.function.arguments or "",
                            },
                        }
                    else:
                        buf = tool_call_buffer[index]
                        if tc_delta.id:
                            buf["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                buf["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                buf["function"]["arguments"] += tc_delta.function.arguments

            # --- 结束信号 ---
            if finish_reason is not None:
                if tool_call_buffer:
                    # 将所有累积的工具调用依次 yield
                    for tc in tool_call_buffer.values():
                        tc_json = json.dumps(tc, ensure_ascii=False)
                        yield {"type": "tool_call", "content": tc_json}
                    tool_call_buffer.clear()
                break

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_kwargs(self, messages: list, tools: Optional[list] = None,
                      temperature: Optional[float] = None) -> dict:
        """构建 OpenAI API 请求参数字典"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return kwargs

    def _parse_response(self, response) -> dict:
        """解析 OpenAI API 的完整响应（非流式）"""
        choice = response.choices[0]
        message = choice.message

        result = {
            "role": "assistant",
            "content": message.content or "",
        }

        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
            result["tool_calls"] = tool_calls

        # 提取 token 用量
        if hasattr(response, "usage") and response.usage:
            result["usage"] = {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            }

        return result
