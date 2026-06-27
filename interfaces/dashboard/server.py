"""
仪表盘 — FastAPI 服务
实时监控 楠楠/CC/Codex/Kimi 工作进程
"""
import sys
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from interfaces.dashboard.monitor import get_monitor
from interfaces.dashboard.state import DashboardState

app = FastAPI(title="小t Agent 仪表盘")

_STATIC_DIR = Path(__file__).parent / "static"

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="dashboard_static")


@app.get("/")
async def index():
    """仪表盘首页"""
    html_path = _STATIC_DIR / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>仪表盘</h1><p>页面加载失败</p>")


@app.get("/manifest.json")
async def manifest():
    return FileResponse(_STATIC_DIR / "manifest.json", media_type="application/manifest+json")


@app.get("/icon.svg")
async def icon():
    return FileResponse(_STATIC_DIR / "icon.svg", media_type="image/svg+xml")


@app.get("/api/status")
async def get_status():
    """实时状态 API（前端轮询用）—— 返回进程+管线+余额+系统"""
    monitor = get_monitor()
    return monitor.get_full_status()


@app.get("/api/state")
async def get_state():
    """只返回共享状态（管线 + Agent 任务描述）"""
    return DashboardState.get_state()


def run(host="0.0.0.0", port=7861):
    """启动仪表盘"""
    import uvicorn
    print(f"[仪表盘] 🖥️  打开浏览器访问 http://localhost:{port}")
    print(f"[仪表盘] 按 Ctrl+C 停止")
    uvicorn.run(
        "interfaces.dashboard.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="warning",
    )


if __name__ == "__main__":
    run()
