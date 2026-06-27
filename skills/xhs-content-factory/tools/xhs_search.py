"""
小红书搜索工具 — Rnote API + 公开搜索（SearXNG）
================================================
XHS Playwright 爬虫已删除（反爬太严）。
- 有 Rnote Key → Rnote 搜索一次 + 公开搜索一次，合并返回
- 无 Rnote Key → 只公开搜索一次

外部接口: search_xhs_notes, search_xhs_suggest, analyze_xhs_keywords
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

# ── 常量 ──
HEADERS_TEMPLATE = {
    'Accept': 'application/json, text/plain, */*',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                  ' (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
}

# ── Rnote API ──
RNOTE_BASE = "https://rnote.dev/api/v2/crawler"
_RNOTE_API_KEY = os.environ.get("RNOTE_API_KEY", "")
_SORT_MAP = {
    "综合": "general",
    "最新": "time_descending",
    "最多点赞": "popularity_descending",
}

# ── SearXNG 配置（从 config_loader 读取） ──
_SEARXNG_URL = None


def _get_searxng_url() -> str:
    """获取 SearXNG 地址（懒加载）"""
    global _SEARXNG_URL
    if _SEARXNG_URL:
        return _SEARXNG_URL
    try:
        from config_loader import get
        _SEARXNG_URL = get("search.base_url", "http://172.27.202.242:8080")
    except ImportError:
        _SEARXNG_URL = "http://172.27.202.242:8080"
    return _SEARXNG_URL


# ── Rnote 小红书搜索（单次） ─────────────────────────────────

def _rnote_search_notes(keyword: str, max_results: int = 20,
                        sort_by: str = "综合") -> dict:
    """
    调 Rnote API 搜索小红书笔记。
    余额不足（402）静默跳过。
    返回: {"success": bool, "notes": [...], "count": N, "error": ""}
    """
    global _RNOTE_API_KEY
    if not _RNOTE_API_KEY:
        try:
            from config_loader import get
            _RNOTE_API_KEY = get("rnote.api_key", "")
        except ImportError:
            pass
    if not _RNOTE_API_KEY:
        return {"success": False, "notes": [], "count": 0,
                "error": ""}

    sort = _SORT_MAP.get(sort_by, "general")
    url = (f"{RNOTE_BASE}/search/notes"
           f"?keyword={urllib.parse.quote(keyword)}"
           f"&page=1&sort={sort}")
    req = urllib.request.Request(url, headers={
        "X-API-Key": _RNOTE_API_KEY,
        "User-Agent": HEADERS_TEMPLATE["User-Agent"],
        "Accept": "application/json",
    })
    try:
        resp = urllib.request.urlopen(req, timeout=20)
        data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 402:
            return {"success": False, "notes": [], "count": 0,
                    "error": ""}
        err_body = e.read().decode("utf-8", errors="replace")[:200]
        return {"success": False, "notes": [], "count": 0,
                "error": f"Rnote HTTP {e.code}"}
    except Exception as e:
        return {"success": False, "notes": [], "count": 0,
                "error": f"Rnote 请求失败: {str(e)[:100]}"}

    raw_items = data.get("data", {}).get("data", {}).get("items", [])
    if not raw_items:
        return {"success": True, "notes": [], "count": 0, "error": ""}

    notes = []
    for item in raw_items:
        note_data = item.get("note", {}) or item
        tags = [t.strip() for t in note_data.get("desc", "").split("#") if t.strip()][:8]
        notes.append({
            "title": note_data.get("title", ""),
            "description": note_data.get("desc", "")[:200],
            "likes": note_data.get("liked_count", 0),
            "comments": note_data.get("comments_count", 0),
            "collects": note_data.get("collected_count", 0),
            "shares": note_data.get("shared_count", 0),
            "author": note_data.get("user", {}).get("nickname", ""),
            "note_id": note_data.get("id", ""),
            "tags": tags,
            "is_video": note_data.get("type") == "video",
            "cover_url": (note_data.get("image_list") or [{}])[0].get("url", ""),
            "source_keyword": keyword,
            "source": "rnote",
        })

    return {"success": True, "notes": notes[:max_results],
            "count": len(notes), "error": ""}


# ── 公开搜索（SearXNG，单次） ──────────────────────────────

def _web_search(query: str, max_results: int = 5) -> dict:
    """
    调 SearXNG 搜索一次。失败也不抛。
    返回: {"success": bool, "results": [...], "error": ""}
    """
    base = _get_searxng_url()
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "language": "zh-CN",
        "categories": "general",
        "pageno": 1,
    })
    url = f"{base}/search?{params}"
    req = urllib.request.Request(url, headers={
        "User-Agent": HEADERS_TEMPLATE["User-Agent"],
        "Accept": "application/json",
    })
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results", [])
        return {
            "success": True,
            "results": [{
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:300],
                "source": r.get("engine", "web"),
            } for r in results[:max_results]],
            "count": len(results),
            "error": "",
        }
    except Exception as e:
        return {"success": False, "results": [], "count": 0,
                "error": f"公开搜索失败: {str(e)[:100]}"}


# ── RAG 格式化 ─────────────────────────────────────────────

def _format_notes_for_rag(notes: list, keyword: str) -> str:
    """笔记列表 → markdown，供 RAG 注入"""
    lines = []
    lines.append(f"# {keyword} 小红书搜索笔记\n")
    lines.append(f"> 来源：XHS Search (xhs_search.py)")
    lines.append(f"> 采集时间：{time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> 用途：内容工坊 RAG 参考库 — 搜索结果知识注入\n")
    lines.append("---\n")

    for rank, note in enumerate(notes, 1):
        title = note.get("title") or "(无标题)"
        likes = note.get("likes", 0) or 0
        comments = note.get("comments", 0) or 0
        collects = note.get("collects", 0) or 0
        author = note.get("author", "未知")
        note_id = note.get("note_id", "")
        ntype = "视频" if note.get("is_video") else "图文"
        source = note.get("source", "rnote")
        lines.append(f"### {rank}. {title} [{source}]")
        lines.append(f"- 类型：{ntype} | 👍{likes:,} 💬{comments} ⭐{collects:,}")
        lines.append(f"- 作者：{author}")
        if note_id:
            lines.append(f"- 链接：https://xiaohongshu.com/explore/{note_id}")
        lines.append("")

    lines.append("---\n")
    lines.append("## 📊 数据概览\n")
    if notes:
        likes_list = [n.get("likes", 0) or 0 for n in notes if n.get("likes")]
        if likes_list:
            lines.append(f"| 指标 | 数据 |")
            lines.append(f"|:----|:-----|")
            lines.append(f"| 最高赞 | {max(likes_list):,} |")
            lines.append(f"| 最低赞 | {min(likes_list):,} |")
            lines.append(f"| 平均赞 | {sum(likes_list)//len(likes_list):,} |")
            lines.append(f"| 收录 | {len(notes)} 条 |")

    return "\n".join(lines)


# ── 便捷调用（供 agent 工具系统） ──────────────────────────

_rag_engine = None


def set_rag_engine(engine):
    """设置 RAG 引擎实例"""
    global _rag_engine
    _rag_engine = engine


def search_xhs_notes(keyword: str, max_results: int = 20,
                     sort_by: str = "综合", **kwargs) -> dict:
    """
    搜索小红书笔记 — 合并 Rnote 搜索 + 公开搜索。
    - 有 Rnote Key：Rnote 一次 + 公开搜索一次
    - 无 Rnote Key：只公开搜索一次
    两边各算一次，出错不重试。
    """
    combined_notes = []
    rnote_info = ""
    web_info = ""

    # 1) Rnote 搜索（有 Key 才调）
    rnote_result = _rnote_search_notes(keyword, max_results=max_results, sort_by=sort_by)
    if rnote_result["notes"]:
        combined_notes.extend(rnote_result["notes"])
        rnote_info = f"Rnote {rnote_result['count']}条"
    elif rnote_result["error"]:
        rnote_info = f"Rnote: {rnote_result['error']}"
    else:
        rnote_info = "Rnote: 无结果/无Key"

    # 2) 公开搜索（SearXNG，每次都调一次）
    web_result = _web_search(f"小红书 {keyword}", max_results=5)
    if web_result["results"]:
        web_info = f"公开搜索 {web_result['count']}条"
    elif web_result["error"]:
        web_info = f"公开搜索: {web_result['error']}"
    else:
        web_info = "公开搜索: 无结果"

    # 合并结果
    summary = f"[{rnote_info}] + [{web_info}]"

    # 自动注入 RAG
    if combined_notes and _rag_engine is not None:
        try:
            formatted = _format_notes_for_rag(combined_notes, keyword)
            _rag_engine.ingest(
                title=keyword,
                content=formatted,
                category="xhs_search",
                tags=[keyword],
                source="xhs_search",
            )
        except Exception:
            pass

    return {
        "success": True,
        "notes": combined_notes,
        "count": len(combined_notes),
        "summary": summary,
        "rnote": rnote_info,
        "web": web_info,
        "error": "",
    }


def search_xhs_suggest(keyword: str) -> list:
    """搜索联想词（edith API 需浏览器 Cookie，已废弃，返回空）"""
    return []


def analyze_xhs_keywords(seed_keywords: list) -> dict:
    """
    关键词挖掘 — 基于 Rnote 搜索（单个关键词只搜一次）
    """
    all_terms = {}
    for seed in seed_keywords:
        result = _rnote_search_notes(seed, max_results=5)
        if result["success"] and result["notes"]:
            for note in result["notes"]:
                keyword_key = note.get("title", "")[:20]
                if keyword_key and keyword_key not in all_terms:
                    all_terms[keyword_key] = {
                        "keyword": keyword_key,
                        "source": seed,
                        "likes": note.get("likes", 0),
                    }

    sorted_terms = sorted(all_terms.values(), key=lambda x: -x["likes"])
    suggestions = [t["keyword"] for t in sorted_terms[:10]]

    return {
        "success": True,
        "keywords": sorted_terms,
        "suggestions": suggestions,
        "count": len(sorted_terms),
        "error": "",
    }
