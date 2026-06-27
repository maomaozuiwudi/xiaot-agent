"""Key余额查询模块 — 从各配置文件读取 Key 并查询余额"""

import json
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


def _is_truncated(key: str) -> bool:
    """检查 Key 是否被安全系统截断（包含 ... 或仅显示头尾字符）"""
    if not key:
        return True
    key = key.strip()
    if len(key) < 10:
        return True
    if "..." in key:
        return True
    return False


def _read_key(path: Path) -> Optional[str]:
    """从文件读取 Key，返回 None 表示未找到"""
    try:
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            if key and not _is_truncated(key):
                return key
        return None
    except OSError:
        return None


def _read_config_key(config_path: Path, key_path: str) -> Optional[str]:
    """从 YAML 配置读取指定路径的 key"""
    try:
        if config_path.exists():
            import yaml
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            parts = key_path.split(".")
            val = cfg
            for p in parts:
                if isinstance(val, dict) and p in val:
                    val = val[p]
                else:
                    return None
            if isinstance(val, str) and val and not _is_truncated(val):
                return val.strip()
    except Exception:
        pass
    return None


def _http_get_json(url: str, headers: dict, timeout: int = 5) -> Optional[dict]:
    """HTTP GET 请求并返回 JSON"""
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (URLError, OSError, json.JSONDecodeError) as e:
        print(f"[Balances] HTTP 请求失败 {url}: {e}")
        return None


def _fmt_balance(val) -> str:
    """格式化余额为 ¥xx.xx，处理数字/字符串类型"""
    try:
        return f"¥{float(val):.2f}"
    except (TypeError, ValueError):
        return f"¥{val}"


def query_deepseek(key: str) -> dict:
    """查询 DeepSeek 余额"""
    data = _http_get_json(
        "https://api.deepseek.com/user/balance",
        {"Authorization": f"Bearer {key}"},
    )
    if data and "balance_infos" in data:
        infos = data["balance_infos"]
        if infos:
            total = infos[0].get("total_balance", 0)
            return {"balance": _fmt_balance(total), "status": "✅"}
    return {"balance": "查询失败", "status": "❌"}


def query_kimi(key: str) -> dict:
    """查询 Kimi/Moonshot 余额"""
    data = _http_get_json(
        "https://api.moonshot.cn/v1/billing/balance",
        {"Authorization": f"Bearer {key}"},
    )
    if data and "available_balance" in data:
        bal = data["available_balance"]
        return {"balance": _fmt_balance(bal), "status": "✅"}
    return {"balance": "查询失败", "status": "❌"}


def query_siliconflow(key: str) -> dict:
    """查询 SiliconFlow 余额"""
    data = _http_get_json(
        "https://api.siliconflow.com/v1/user/balance",
        {"Authorization": f"Bearer {key}"},
    )
    if data and "balance" in data:
        bal = data["balance"]
        return {"balance": _fmt_balance(bal), "status": "✅"}
    return {"balance": "查询失败", "status": "❌"}


def query_rnote(key: str) -> dict:
    """Rnote 不支持余额查询"""
    return {"balance": "按量计费", "status": "⚡"}


def query_volcengine(key: str) -> dict:
    """火山引擎/豆包不支持公开余额API"""
    return {"balance": "需网页查看", "status": "⚠️"}


def get_all_balances() -> dict:
    """查询所有平台的余额"""
    proj_root = Path(__file__).parent.parent.parent.resolve()
    config_path = proj_root / "config.yaml"

    # DeepSeek — 从项目 config.yaml llm.api_key
    ds_key = _read_config_key(config_path, "llm.api_key")
    # Kimi — config.yaml llm.vision.api_key
    km_key = _read_config_key(config_path, "llm.vision.api_key")
    # Rnote — config.yaml rnote.api_key
    rn_key = _read_config_key(config_path, "rnote.api_key")
    # SiliconFlow — ~/voice_clone/.sf_key
    sf_path = Path.home() / "voice_clone" / ".sf_key"
    sf_key = _read_key(sf_path)
    # 火山引擎 — ~/voice_clone/.volc_voice_key
    vc_path = Path.home() / "voice_clone" / ".volc_voice_key"
    vc_key = _read_key(vc_path)

    balances = {}

    if ds_key:
        balances["DeepSeek"] = query_deepseek(ds_key)
    else:
        balances["DeepSeek"] = {"balance": "未配置", "status": "⚪"}

    if km_key:
        balances["Kimi"] = query_kimi(km_key)
    else:
        balances["Kimi"] = {"balance": "未配置", "status": "⚪"}

    if sf_key:
        balances["SiliconFlow"] = query_siliconflow(sf_key)
    else:
        balances["SiliconFlow"] = {"balance": "未配置", "status": "⚪"}

    # Rnote — always show billing info
    balances["Rnote"] = query_rnote(rn_key)

    if vc_key:
        balances["火山引擎"] = query_volcengine(vc_key)
    else:
        balances["火山引擎"] = {"balance": "未配置", "status": "⚪"}

    return balances
