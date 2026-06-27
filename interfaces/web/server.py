"""
小红书内容工坊 Agent — FastAPI Web 服务
"""
import sys
import os
import json
import asyncio
import uuid
from pathlib import Path
from typing import AsyncGenerator

# ── 项目根路径 ──
_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config_loader import load_config, get
from brain.agent import AgentBrain
from knowledge.memory import UserManager

# ── 初始化 ──
HOST = "0.0.0.0"
PORT = 7860

app = FastAPI(title="小t Agent - 小红书内容工坊")

# 静态文件 + 模板
_static_dir = Path(__file__).parent / "static"
_templates_dir = Path(__file__).parent / "templates"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
templates = Jinja2Templates(directory=str(_templates_dir))

# ── Agent 单例 ──
_brain: AgentBrain = None


def get_brain() -> AgentBrain:
    global _brain
    if _brain is None:
        load_config()
        api_key = get("llm.api_key", "")
        user_mgr = UserManager()
        user_mgr.login(api_key)
        _brain = AgentBrain(user_mgr)
    return _brain


# ── 模型 ──


class ChatRequest(BaseModel):
    message: str = ""


class HistoryItem(BaseModel):
    role: str
    content: str


# ── 路由 ──


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """返回聊天页面"""
    html_path = _templates_dir / "chat.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
    else:
        html = "<h1>小t Agent</h1><p>聊天页面加载失败</p>"
    return HTMLResponse(html)


@app.post("/api/chat")
async def chat_sync(req: Request):
    """同步聊天 — 收集全部回复后返回"""
    body = await req.body()
    data = json.loads(body) if body else {}
    message = data.get("message", "")
    if not message:
        return JSONResponse({"error": "消息不能为空"}, status_code=400)

    brain = get_brain()
    response_text = ""
    steps = []
    try:
        for result in brain.chat(message):
            t = result["type"]
            c = result["content"]
            if t == "text":
                response_text = c
            elif t == "step":
                steps.append(c)
            elif t == "error":
                response_text = c
            elif t == "tool_call":
                steps.append(c)
            elif t == "done":
                break
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return {
        "response": response_text,
        "steps": steps,
    }


@app.post("/api/chat/stream")
async def chat_stream(req: Request):
    """SSE 流式聊天"""
    body = await req.body()
    data = json.loads(body) if body else {}
    message = data.get("message", "")

    if not message:
        return JSONResponse({"error": "消息不能为空"}, status_code=400)

    brain = get_brain()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            for result in brain.chat(message):
                t = result["type"]
                c = result["content"]
                if t == "text":
                    yield f"data: {json.dumps({'type': 'text', 'content': c}, ensure_ascii=False)}\n\n"
                elif t == "step":
                    yield f"data: {json.dumps({'type': 'step', 'content': c}, ensure_ascii=False)}\n\n"
                elif t == "tool_call":
                    yield f"data: {json.dumps({'type': 'tool_call', 'content': c}, ensure_ascii=False)}\n\n"
                elif t == "error":
                    yield f"data: {json.dumps({'type': 'error', 'content': c}, ensure_ascii=False)}\n\n"
                elif t == "done":
                    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/history")
async def get_history():
    """获取对话历史"""
    brain = get_brain()
    memory = brain.memory
    history = memory.get_context()
    # 只返回 user / assistant 角色，过滤 system
    items = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            items.append({"role": role, "content": content})
    return {"history": items}


@app.post("/api/reset")
async def reset_chat():
    """重置对话"""
    brain = get_brain()
    brain.reset()
    return {"status": "ok", "message": "对话已重置"}


# ── 启动 ──


def run():
    """启动 uvicorn 服务器"""
    import uvicorn
    print(f"[Web] 🐱 小t Agent 启动在 http://localhost:{PORT}")
    print(f"[Web] 按 Ctrl+C 停止服务")
    uvicorn.run(
        "interfaces.web.server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
