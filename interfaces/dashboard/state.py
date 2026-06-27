"""共享状态系统 — 统一管理 ~/.xiaot_agent/dashboard_state.json"""

import json
from pathlib import Path
from typing import Optional

STATE_PATH = Path.home() / ".xiaot_agent" / "dashboard_state.json"

DEFAULT_STATE = {
    "pipeline": {
        "title": "",
        "steps": [],
    },
    "agents": {
        "nannan": {"status": "idle", "task": ""},
        "cc": {"status": "idle", "task": ""},
        "codex": {"status": "idle", "task": ""},
        "kimi": {"status": "idle", "task": ""},
    },
    "balances": {},
}


class DashboardState:
    """共享状态管理器"""

    @staticmethod
    def get_state() -> dict:
        """读取当前状态，返回完整状态结构"""
        try:
            if STATE_PATH.exists():
                raw = STATE_PATH.read_text(encoding="utf-8")
                data = json.loads(raw)
                # 合并默认值，确保结构完整
                state = DEFAULT_STATE.copy()
                state.update(data)
                # 确保 agents 子键存在
                agents = state.setdefault("agents", {})
                for aid in DEFAULT_STATE["agents"]:
                    if aid not in agents:
                        agents[aid] = {"status": "idle", "task": ""}
                return state
        except (json.JSONDecodeError, OSError) as e:
            print(f"[DashboardState] 读取失败: {e}")
        return DEFAULT_STATE.copy()

    @staticmethod
    def set_state(state: dict):
        """写入状态（给Agent调用）"""
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            # 只写必要字段，减少写入量
            minimal = {
                "pipeline": state.get("pipeline", DEFAULT_STATE["pipeline"]),
                "agents": state.get("agents", DEFAULT_STATE["agents"]),
            }
            STATE_PATH.write_text(
                json.dumps(minimal, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            print(f"[DashboardState] 写入失败: {e}")

    @staticmethod
    def update_agent(agent_id: str, status: str, task: str = ""):
        """更新单个Agent状态（便捷方法）"""
        state = DashboardState.get_state()
        agents = state.setdefault("agents", {})
        if agent_id not in agents:
            agents[agent_id] = {}
        agents[agent_id]["status"] = status
        agents[agent_id]["task"] = task
        DashboardState.set_state(state)

    @staticmethod
    def update_pipeline(title: Optional[str] = None, steps: Optional[list] = None):
        """更新管线（便捷方法）"""
        state = DashboardState.get_state()
        pipeline = state.setdefault("pipeline", {})
        if title is not None:
            pipeline["title"] = title
        if steps is not None:
            pipeline["steps"] = steps
        DashboardState.set_state(state)
