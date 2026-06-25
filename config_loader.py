"""
配置加载器 — 读取 config.yaml，支持环境变量注入
"""
import os
import re
from pathlib import Path
from typing import Any

_CONFIG = None
_CONFIG_PATH = None


def _resolve_env_vars(value):
    """递归解析 ${VAR_NAME} 环境变量"""
    if isinstance(value, str):
        def _replace(m):
            var = m.group(1)
            return os.environ.get(var, "")
        return re.sub(r'\$\{(\w+)\}', _replace, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_config(config_path=None):
    """加载配置文件"""
    global _CONFIG, _CONFIG_PATH

    if config_path is None:
        # 自动查找
        candidates = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
            Path.home() / ".xhs_agent" / "config.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = str(c)
                break
        if config_path is None:
            raise FileNotFoundError("未找到 config.yaml，请在项目根目录创建")

    _CONFIG_PATH = Path(config_path)
    import yaml
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _CONFIG = _resolve_env_vars(raw) if raw else {}
    return _CONFIG


def get(key: str, default=None) -> Any:
    """按点号路径取值，如 get('llm.model')"""
    if _CONFIG is None:
        load_config()
    parts = key.split(".")
    val = _CONFIG
    for p in parts:
        if isinstance(val, dict) and p in val:
            val = val[p]
        else:
            return default
    return val


def get_config() -> dict:
    if _CONFIG is None:
        load_config()
    return _CONFIG


def resolve_path(relative_path: str) -> str:
    """相对路径 → 绝对路径（基于配置的 work_dir）"""
    work_dir = get("output.work_dir", str(Path.cwd()))
    return str(Path(work_dir) / relative_path)
