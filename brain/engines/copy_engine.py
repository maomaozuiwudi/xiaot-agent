"""
文案生成引擎 — 直接调用 Agent 配置的 LLM
============================================
不依赖旧项目 providers/copy/ 插槽层，直接使用 OpenAI-compatible API
调用 DeepSeek/Kimi 等模型生成小红书文案。
"""

import json
import os
from openai import OpenAI


def generate_xhs_copy(topic, context="", visual_context="", storyboard=""):
    """调用配置的 LLM 生成小红书文案

    Args:
        topic: 文案主题/产品名称
        context: 补充背景信息（卖点、使用场景等）
        visual_context: 视觉素材描述
        storyboard: 分镜表（画面描述+时长）

    Returns:
        dict: {"title": str, "body": str, "tags": list}
    """
    # 从 Agent 配置读取 LLM 信息
    import sys
    sys.path.insert(
        0,
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    from config_loader import get

    api_key = get("llm.api_key", "")
    base_url = get("llm.base_url", "https://api.deepseek.com")
    model = get("llm.model", "deepseek-chat")

    if not api_key:
        return {"title": "", "body": "[错误: 未配置 LLM API Key]", "tags": []}

    client = OpenAI(api_key=api_key, base_url=base_url)

    system_prompt = """你是一个小红书内容创作专家。根据提供的信息，生成高质量小红书文案。
要求:
1. 首段直接给结论，吸引眼球
2. 内容模块化，每段50-150字
3. 口语化，有真实使用感受
4. 标题包含关键词，带emoji
5. 输出格式：JSON {"title": "标题", "body": "正文", "tags": ["#tag1", "#tag2"]}"""

    user_prompt = f"主题: {topic}\n"
    if context:
        user_prompt += f"背景信息: {context}\n"
    if visual_context:
        user_prompt += f"画面描述: {visual_context}\n"
    if storyboard:
        user_prompt += f"分镜表: {storyboard}\n"
    user_prompt += "\n请生成小红书文案（JSON格式）。"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    text = resp.choices[0].message.content or ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"title": topic, "body": text, "tags": []}
