"""
LLM Provider 注册表
根据 config 自动加载对应的 provider
"""
from brain.providers.base import BaseLLMProvider


_providers = {}


def register_provider(name: str, provider_cls):
    """注册 provider 类"""
    _providers[name] = provider_cls


def get_provider(config: dict = None) -> BaseLLMProvider:
    """根据配置获取 LLM Provider 实例"""
    if config is None:
        from config_loader import get as cfg_get
        config = {
            "provider": cfg_get("llm.provider", "openai_compat"),
            "model": cfg_get("llm.model", "deepseek-chat"),
            "api_key": cfg_get("llm.api_key", ""),
            "base_url": cfg_get("llm.base_url", "https://api.deepseek.com"),
            "temperature": cfg_get("llm.temperature", 0.3),
            "max_tokens": cfg_get("llm.max_tokens", 8192),
            "context_window": cfg_get("llm.context_window", 1000000),
        }
    provider_name = config.get("provider", "openai_compat")
    if provider_name not in _providers:
        # 尝试动态导入
        import importlib
        try:
            mod = importlib.import_module(f"brain.providers.{provider_name}")
            # 模块内应该调用了 register_provider
        except ImportError:
            raise ValueError(f"未知的 LLM Provider: {provider_name}，"
                             f"可用: {list(_providers.keys())}")

    cls = _providers.get(provider_name)
    if cls is None:
        raise ValueError(f"Provider {provider_name} 未注册")
    return cls(config)


# 注册 OpenAI 兼容协议（默认）
from brain.providers.openai_compat import OpenAICompatProvider
register_provider("openai_compat", OpenAICompatProvider)
