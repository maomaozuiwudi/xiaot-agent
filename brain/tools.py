"""
内容工坊工具注册与执行系统
================================
ToolRegistry: 管理所有可被 AI 调用的工具
提供 OpenAI-compatible Function Calling 定义
"""

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

from config_loader import get

logger = logging.getLogger(__name__)


# ======================================================================
# 内置工具 handlers
# ======================================================================

def _clip_videos(args: dict) -> tuple[str, dict]:
    """裁剪/拼接多个视频片段，用 MediaPipe 骨架检测自动选取最佳片段"""
    paths = args.get("paths", [])
    # 兼容 MSYS 路径 /e/xxx → E:/xxx
    import re as _re
    _converted = []
    for p in paths:
        _m = _re.match(r'^/([a-zA-Z])/', p)
        if _m:
            p = f"{_m.group(1).upper()}:/{p[3:]}"
        _converted.append(p)
    paths = _converted
    duration = args.get("duration", 15.0)

    if not paths:
        return ("未提供视频路径", {"status": "error", "error": "未提供视频路径"})

    from brain.engines.clip_engine import ClipVideo

    output_dir = "output/temp_clips/"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    clip = ClipVideo()

    for i, vpath in enumerate(paths):
        out_path = os.path.join(output_dir, f"clip_{i}.mp4")
        try:
            actual_path, bbox = clip.clip_video(vpath, out_path, target_duration=duration)
            results.append({"index": i, "path": actual_path, "bbox_size": bbox})
        except Exception as e:
            results.append({"index": i, "path": vpath, "error": str(e)})

    summary = f"已剪辑 {len([r for r in results if 'error' not in r])}/{len(paths)} 个视频"
    clip_paths = [r["path"] for r in results if "error" not in r]
    path_list = "\n".join(f"  clip_{i}: {p}" for i, p in enumerate(clip_paths))
    result_text = f"{summary}\n剪辑后的片段路径:\n{path_list}"
    return (result_text, {"status": "ok", "clips": results, "clip_paths": clip_paths})


def _vision_analyze(args: dict) -> tuple[str, dict]:
    """调用 Kimi K2.5 视觉模型分析图片内容（人物、服装、颜色、场景、动作等）"""
    paths = args.get("paths", [])
    logger.info("[Tool:vision_analyze] paths=%s", paths)

    if not paths:
        return (
            "未提供图片路径",
            {"status": "error", "tool": "vision_analyze", "error": "未提供图片路径"},
        )

    # 读取配置：优先用视觉模型配置，未配置时降级用主模型
    vision_key = get("llm.vision.api_key", "")
    vision_base_url = get("llm.vision.base_url", "")
    vision_model = get("llm.vision.model", "")

    # 没独立配看图 Key → 降级用主模型
    if not vision_key:
        vision_key = get("llm.api_key", "")
        vision_base_url = get("llm.base_url", "https://api.deepseek.com")
        vision_model = get("llm.model", "deepseek-chat")
    else:
        # 有视觉配置就用视觉的，缺省 fallback
        vision_base_url = vision_base_url or "https://api.moonshot.cn/v1"
        vision_model = vision_model or "kimi-k2.5"

    # 主模型 Key 也没有，才报错
    if not vision_key:
        return (
            "看图功能未配置，请在登录时配置视觉模型 API Key",
            {"status": "error", "tool": "vision_analyze", "error": "视觉模型 API Key 未配置"},
        )

    # 构建多模态消息内容
    content_parts = [
        {"type": "text", "text": "请详细描述这些图片的内容，包括人物、服装、颜色、场景、动作等细节"}
    ]

    for path_str in paths:
        try:
            # 转换 MSYS 路径 /e/xxx → E:/xxx
            _msys_m = __import__('re').match(r'^/([a-zA-Z])/', path_str)
            if _msys_m:
                path_str = f"{_msys_m.group(1).upper()}:/{path_str[3:]}"
            p = Path(path_str).resolve()
            if not p.exists():
                logger.warning("图片文件不存在: %s", p)
                content_parts.append({"type": "text", "text": f"[文件不存在: {path_str}]"})
                continue

            suffix = p.suffix.lower()
            if suffix in (".jpg", ".jpeg"):
                mime = "image/jpeg"
                # base64 编码图片
                with open(p, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                data_url = f"data:{mime};base64,{b64_data}"
                content_parts.append({"type": "image_url", "image_url": {"url": data_url}})

            elif suffix == ".png":
                mime = "image/png"
                with open(p, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                data_url = f"data:{mime};base64,{b64_data}"
                content_parts.append({"type": "image_url", "image_url": {"url": data_url}})

            elif suffix == ".webp":
                mime = "image/webp"
                with open(p, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                data_url = f"data:{mime};base64,{b64_data}"
                content_parts.append({"type": "image_url", "image_url": {"url": data_url}})

            elif suffix == ".mp4":
                # 视频文件：ffmpeg 抽一帧，转为 jpg 发给 Kimi 分析
                import subprocess, tempfile
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    frame_path = tmp.name
                ret = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(p), "-ss", "00:00:01",
                     "-vframes", "1", "-q:v", "2", frame_path],
                    capture_output=True, timeout=30,
                )
                if ret.returncode == 0 and Path(frame_path).exists():
                    with open(frame_path, "rb") as f:
                        b64_data = base64.b64encode(f.read()).decode("utf-8")
                    data_url = f"data:image/jpeg;base64,{b64_data}"
                    content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                    Path(frame_path).unlink(missing_ok=True)
                else:
                    content_parts.append({"type": "text", "text": f"[视频抽帧失败: {path_str}]"})

            else:
                logger.warning("不支持的格式: %s", suffix)
                content_parts.append({"type": "text", "text": f"[不支持的格式: {path_str}]"})

        except Exception as e:
            logger.exception("读取图片失败: %s", path_str)
            content_parts.append({"type": "text", "text": f"[读取图片失败: {path_str} - {e}]"})

    # 如果没有成功加载任何图片，返回错误
    image_count = sum(1 for c in content_parts if c["type"] == "image_url")
    if image_count == 0:
        return (
            "没有可分析的图片",
            {"status": "error", "tool": "vision_analyze", "error": "没有可分析的图片"},
        )

    # 调用 Kimi API
    try:
        import openai

        client = openai.OpenAI(api_key=vision_key, base_url=vision_base_url, timeout=90)
        response = client.chat.completions.create(
            model=vision_model,
            messages=[{"role": "user", "content": content_parts}],
            max_tokens=2048,
        )
        description = response.choices[0].message.content or ""
    except Exception as e:
        logger.exception("Kimi API 调用失败")
        return (
            f"图片分析失败：{e}",
            {"status": "error", "tool": "vision_analyze", "error": str(e)},
        )

    return (
        description,
        {
            "status": "ok",
            "tool": "vision_analyze",
            "paths": paths,
            "image_count": image_count,
            "description": description,
        },
    )


def _generate_copy(args: dict) -> tuple[str, dict]:
    """根据主题和分镜信息生成小红书种草文案"""
    topic = args.get("topic", "")
    context = args.get("context", "")
    visual_context = args.get("visual_context", "")
    storyboard = args.get("storyboard", "")
    logger.info("[Tool:generate_copy] topic=%s", topic)

    if not topic:
        return ("请提供文案主题", {"status": "error", "error": "缺少主题"})

    from brain.engines.copy_engine import generate_xhs_copy

    result = generate_xhs_copy(topic, context, visual_context, storyboard)

    preview = (
        f"【{result.get('title','')}】\n\n"
        f"{result.get('body','')[:200]}...\n\n"
        f"{' '.join(result.get('tags',[]))}"
    )
    return (preview, {"status": "ok", "copy": result})


def _compose_video(args: dict) -> tuple[str, dict]:
    """将多个已剪辑的视频片段合成为最终视频"""
    clips = args.get("clips", [])
    settings = args.get("settings", {})
    logger.info("[Tool:compose_video] clips=%s, settings=%s", len(clips), settings)

    if not clips:
        return ("未提供视频片段", {"status": "error", "error": "缺少clips"})

    from brain.engines.composer import compose_mixed

    shots = [{"type": "video", "path": c, "duration": 8.0} for c in clips]

    resolution = (1080, 1920)  # 默认竖屏
    if settings.get("resolution") in ("1920x1080", "1080p", "landscape"):
        resolution = (1920, 1080)

    output_path = f"output/composed_{int(time.time())}.mp4"
    os.makedirs("output", exist_ok=True)

    bgm = settings.get("bgm", "")

    result_path = compose_mixed(shots, output_path, bgm_path=bgm if bgm else None, resolution=resolution)

    return (
        f"✅ 合成完成: {result_path}",
        {"status": "ok", "output_path": result_path, "clip_count": len(clips), "resolution": resolution},
    )


def _generate_card(args: dict) -> tuple[str, dict]:
    """生成推广卡片/封面图 — 使用 Playwright 渲染 HTML 截图"""
    text = args.get("text", "")
    style = args.get("style", {})

    if not text:
        return ("请提供卡片文案", {"status": "error", "error": "缺少text"})

    from brain.engines.card_engine import CardGenerator

    gen = CardGenerator()
    output_path = gen.generate_cover(
        title=text,
        subtitle=style.get("subtitle", ""),
        tags=style.get("tags", []),
    )

    logger.info("[Tool:generate_card] text=%s, output=%s", text[:30], output_path)
    return (
        f"卡片已生成: {output_path}",
        {"status": "ok", "output_path": output_path, "text": text, "style": style},
    )


def _generate_image(args: dict) -> tuple[str, dict]:
    """根据配置的模型文生图 — 即梦/DALL·E/Gemini/主模型"""
    prompt = args.get("prompt", "")
    width = args.get("width", 1024)
    height = args.get("height", 1024)
    logger.info("[Tool:generate_image] prompt=%s, size=%dx%d", prompt[:50], width, height)

    if not prompt:
        return ("请提供图片描述(prompt)", {"status": "error", "error": "缺少prompt"})

    from config_loader import get
    gen_provider = get("image_gen.provider", "jimeng")
    gen_api_key = get("image_gen.api_key", "")

    if gen_provider == "jimeng":
        from brain.engines.jimeng_engine import generate_image
        try:
            local_path = generate_image(prompt, width=width, height=height)
            return (f"✅ 图片已生成: {local_path}", {"status": "ok", "output_path": local_path, "prompt": prompt})
        except ValueError as e:
            return (f"即梦API未配置: {e}", {"status": "error", "error": str(e), "tool": "generate_image"})
        except Exception as e:
            return (f"即梦API调用失败: {e}", {"status": "error", "error": str(e), "tool": "generate_image"})

    elif gen_provider == "openai":
        if not gen_api_key:
            return ("OpenAI API Key 未配置", {"status": "error", "error": "缺少 OpenAI Key", "tool": "generate_image"})
        try:
            from openai import OpenAI
            client = OpenAI(api_key=gen_api_key, timeout=60)
            resp = client.images.generate(model="dall-e-3", prompt=prompt, n=1,
                                           size=f"{width}x{height}" if width == height else "1024x1024")
            img_url = resp.data[0].url
            import urllib.request, os, datetime
            os.makedirs("output/images", exist_ok=True)
            local_path = f"output/images/openai_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            urllib.request.urlretrieve(img_url, local_path)
            return (f"✅ 图片已生成: {local_path}", {"status": "ok", "output_path": local_path, "prompt": prompt})
        except Exception as e:
            return (f"DALL·E 调用失败: {e}", {"status": "error", "error": str(e), "tool": "generate_image"})

    elif gen_provider == "main_model":
        # 用主模型生图（需主模型支持多模态输出）
        return ("主模型生图功能暂未实现", {"status": "error", "error": "主模型生图未实现", "tool": "generate_image"})

    else:
        return (f"未知的生图提供商: {gen_provider}", {"status": "error", "error": f"provider={gen_provider}", "tool": "generate_image"})


def _generate_video(args: dict) -> tuple[str, dict]:
    """根据配置的模型文生视频 — 即梦/DALL·E/Gemini/主模型"""
    prompt = args.get("prompt", "")
    image_url = args.get("image_url", "")
    logger.info("[Tool:generate_video] prompt=%s", prompt[:50])

    if not prompt:
        return ("请提供视频描述(prompt)", {"status": "error", "error": "缺少prompt"})

    from config_loader import get
    gen_provider = get("image_gen.provider", "jimeng")

    if gen_provider == "jimeng":
        from brain.engines.jimeng_engine import generate_video
        try:
            local_path = generate_video(prompt, image_url=image_url or None)
            return (f"✅ 视频已生成: {local_path}", {"status": "ok", "output_path": local_path, "prompt": prompt})
        except ValueError as e:
            return (f"即梦API未配置: {e}", {"status": "error", "error": str(e), "tool": "generate_video"})
        except Exception as e:
            return (f"即梦API调用失败: {e}", {"status": "error", "error": str(e), "tool": "generate_video"})
    else:
        return (f"视频生成仅支持即梦，当前提供商: {gen_provider}", {"status": "error", "error": "仅即梦支持视频生成"})


def _search_web(args: dict) -> tuple[str, dict]:
    """通过 SearXNG 搜索引擎检索网络信息"""
    query = args.get("query", "")
    if not query:
        return ("请提供搜索关键词", {"status": "error", "error": "缺少query"})

    from brain.engines.searxng_search import search

    result = search(query, max_results=5)

    if result.get("error"):
        return (f"搜索失败: {result['error']}", {"status": "error", "error": result["error"]})

    lines = [f"🔍 搜索结果: {query}"]
    for r in result["results"]:
        lines.append(f"\n📌 {r['title']}")
        lines.append(f"   {r['url']}")
        if r.get("content"):
            lines.append(f"   {r['content'][:150]}")
    if not result["results"]:
        lines.append("\n（无结果）")

    return ("\n".join(lines), {"status": "ok", **result})


def _synthesize_tts(args: dict) -> tuple[str, dict]:
    """文本转语音 — 火山引擎声音复刻"""
    text = args.get("text", "")
    voice = args.get("voice", "default")

    if not text:
        return ("请提供要转语音的文本", {"status": "error", "error": "缺少text"})

    from brain.engines.tts_engine import synthesize

    # 默认用火山引擎声音复刻 Voice ID S_wHXLNCs52，其他 voice 值暂忽略
    voice_id = "S_wHXLNCs52"

    try:
        output_path = synthesize(text, voice_id=voice_id)
        return (
            f"音频已生成: {output_path}（{len(text)}字）",
            {"status": "ok", "output_path": output_path, "text_length": len(text), "voice": voice},
        )
    except Exception as e:
        return (
            f"TTS 生成失败: {e}",
            {"status": "error", "error": str(e)},
        )


# 内置工具注册元数据
_BUILTIN_TOOLS: list[dict] = [
    {
        "name": "clip_videos",
        "description": "用骨架检测智能剪辑视频，返回剪辑后片段路径（clip_paths）供合成使用。可设置每段目标时长。",
        "handler": _clip_videos,
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
        "handler": _vision_analyze,
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
        "description": "根据主题、视觉素材描述和分镜时长生成小红书种草文案（含配音脚本），会根据每镜时长自动匹配文案长度",
        "handler": _generate_copy,
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
                "storyboard": {
                    "type": "string",
                    "description": "分镜表：每镜的画面描述+目标时长（秒），格式如「镜1|白色连衣裙|8s\\n镜2|防晒衬衫|5s」，用于控制配音文案总时长",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "compose_video",
        "description": "将多个已剪辑的视频片段合成为最终视频，可配置分辨率、背景音乐、转场等",
        "handler": _compose_video,
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
                            "description": "输出分辨率，默认竖屏 1080×1920。传 'landscape' 或 '1920x1080' 为横屏",
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
        "handler": _generate_card,
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
                        "subtitle": {
                            "type": "string",
                            "description": "副标题/补充文案",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "标签列表，最多5个",
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
        "handler": _search_web,
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
        "handler": _synthesize_tts,
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
    {
        "name": "generate_image",
        "description": "AI文生图。默认跳过不执行，仅当用户明确说用即梦或主动要求生图时才调用。调即梦需消耗API额度。",
        "handler": _generate_image,
        "requires_confirm": True,
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "图片描述文本，越具体越好。如：'一只橘猫坐在沙发上，高清写实'",
                },
                "width": {
                    "type": "integer",
                    "description": "图片宽度（像素），默认1024",
                    "default": 1024,
                },
                "height": {
                    "type": "integer",
                    "description": "图片高度（像素），默认1024",
                    "default": 1024,
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "generate_video",
        "description": "调用即梦AI文生视频/图生视频（即梦视频3.0 1080P）。消耗API额度。",
        "handler": _generate_video,
        "requires_confirm": True,
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "视频描述文本，越具体越好。如：'海边日落，唯美风格'",
                },
                "image_url": {
                    "type": "string",
                    "description": "可选，参考图片URL。传了就是图生视频模式",
                },
            },
            "required": ["prompt"],
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


def format_tool_result(tool_name: str, result: dict, tool_call_id: str = "") -> dict:
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
        "tool_call_id": tool_call_id or tool_name,
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


# ═══════════════════════════════════════════════
# 小红书搜索工具注册（懒加载）
# ═══════════════════════════════════════════════

_xhs_search_loaded = False


def _ensure_xhs_tools():
    """惰性加载并注册小红书搜索工具（Rnote + 公开搜索，已移除 Playwright）"""
    global _xhs_search_loaded
    if _xhs_search_loaded:
        return
    _xhs_search_loaded = True

    try:
        import importlib
        mod = importlib.import_module(
            "skills.xhs-content-factory.tools.xhs_search"
        )
        search_xhs_notes = mod.search_xhs_notes
        analyze_xhs_keywords = mod.analyze_xhs_keywords
    except (ImportError, AttributeError) as e:
        logger.debug("xhs_search 模块不可用: %s", e)

    reg = get_registry()

    def _search(args):
        kw = args.get("keyword", "")
        mr = int(args.get("max_results", 20))
        result = search_xhs_notes(kw, mr)
        notes = result.get("notes", [])
        summary = result.get("summary", "")
        lines = [f"📱 小红书「{kw}」搜索结果"]
        if summary:
            lines.append(f"   {summary}")
        for n in notes[:10]:
            lines.append(f"\n📌 {n.get('title','无标题')}")
            lines.append(f"   作者: {n.get('author','?')}  ❤{n.get('likes',0)}  "
                         f"💬{n.get('comments',0)}  ⭐{n.get('collects',0)}")
            desc = n.get("description", "")[:80]
            if desc:
                lines.append(f"   {desc}")
        if not notes:
            lines.append("\n（无小红书笔记结果，请参考公开搜索结果）")
        return (
            "\n".join(lines),
            {"status": "ok", "notes": notes, "count": result["count"], "summary": summary},
        )

    def _analyze(args):
        seeds = args.get("seed_keywords", [])
        result = analyze_xhs_keywords(seeds)
        if not result["success"]:
            return (f"分析失败: {result.get('error', '未知错误')}", {"status": "error"})
        lines = ["📊 关键词分析"]
        if result.get("suggestions"):
            lines.append(f"\n🔥 推荐: {' → '.join(result['suggestions'])}")
        for kw in result.get("keywords", [])[:15]:
            likes = kw.get("likes", 0)
            if likes > 0:
                lines.append(f"  [❤{likes}] {kw['keyword']}")
        lines.append(f"\n共 {result['count']} 个关键词")
        return (
            "\n".join(lines),
            {"status": "ok", "keywords": result.get("keywords", []), "count": result["count"],
             "suggestions": result.get("suggestions", [])},
        )

    reg.register("search_xhs", "搜索小红书内容 — Rnote API + 公开搜索（SearXNG），有 Rnote Key 则调两次，无 Key 则仅公开搜索", _search,
                 {"type": "object", "properties": {
                     "keyword": {"type": "string"},
                     "max_results": {"type": "integer", "default": 20},
                 }, "required": ["keyword"]})
    reg.register("analyze_xhs_keywords", "小红书关键词挖掘（基于 Rnote 搜索）", _analyze,
                 {"type": "object", "properties": {
                     "seed_keywords": {"type": "array", "items": {"type": "string"}},
                 }, "required": ["seed_keywords"]})
    logger.info("XHS 搜索工具已注册（Rnote + 公开搜索）")
