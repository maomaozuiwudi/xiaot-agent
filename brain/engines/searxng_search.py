"""SearXNG 搜索引擎"""
import json
import urllib.request
import urllib.parse
import ssl
from pathlib import Path


def _load_searxng_url():
    """从 Agent config.yaml 读取 SearXNG URL"""
    try:
        import sys
        import os

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from config_loader import get

        return (
            get("api_keys.searxng.base_url", "http://172.27.202.242:8080")
            or "http://172.27.202.242:8080"
        )
    except Exception:
        return "http://172.27.202.242:8080"


def search(query, max_results=5, language="zh"):
    """通过 SearXNG 搜索网络"""
    base_url = _load_searxng_url()
    params = {"q": query, "format": "json", "language": language, "pageno": 1}
    url = base_url + "/search?" + urllib.parse.urlencode(params)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, context=ctx, timeout=10)
        data = json.loads(resp.read())
        results = data.get("results", [])[:max_results]
        return {
            "query": query,
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                }
                for r in results
            ],
            "total": len(results),
        }
    except Exception as e:
        return {"query": query, "results": [], "total": 0, "error": str(e)}
