"""
进程监控模块 — 采集四个Agent的状态

- 楠楠 (Hermes Agent)
- CC (Claude Code CLI)
- Codex (Codex CLI)
- Kimi (Kimi API)
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None


# ── Agent 定义 ──

AGENTS = {
    "楠楠": {
        "id": "nannan",
        "icon": "🧠",
        "desc": "Hermes Agent · 主调度",
        "process_keywords": ["hermes_cli.main", "hermes agent", "gateway run"],
    },
    "CC": {
        "id": "cc",
        "icon": "⚡",
        "desc": "Claude Code CLI · 执行者",
        "process_keywords": ["claude", "claude-code"],
    },
    "Codex": {
        "id": "codex",
        "icon": "💻",
        "desc": "Codex CLI · 编码",
        "process_keywords": ["codex", "codex_cli", "codex_proxy"],
    },
    "Kimi": {
        "id": "kimi",
        "icon": "👁️",
        "desc": "Kimi K2.5 · 视觉分析",
        "process_keywords": ["kimi", "moonshot"],
    },
}

# 端口 → Agent 映射
PORT_AGENTS = {
    5000: "Codex",
    7860: "小t Web",
    7861: "仪表盘",
}


class ProcessMonitor:
    """系统进程监控器"""

    def __init__(self):
        self._psutil_available = psutil is not None
        self._last_poll = 0
        self._cache = {}
        self._cache_ttl = 2  # 2秒缓存

    def poll(self) -> dict:
        """采集一次所有状态"""
        now = time.time()
        if now - self._last_poll < self._cache_ttl and self._cache:
            return self._cache

        result = {
            "agents": {},
            "system": {},
            "ports": [],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

        if self._psutil_available:
            result["agents"] = self._scan_processes()
            result["system"] = self._get_system_info()
            result["ports"] = self._scan_ports()
        else:
            result["agents"] = {aid: {"status": "unknown", "msg": "psutil 未安装"} for aid in AGENTS}
            result["system"] = {"cpu": 0, "memory": 0, "psutil": False}

        self._cache = result
        self._last_poll = now
        return result

    def _scan_processes(self) -> dict:
        """扫描所有进程，识别各 Agent"""
        agents = {}
        for name, info in AGENTS.items():
            agents[info["id"]] = {
                "name": name,
                "icon": info["icon"],
                "desc": info["desc"],
                "status": "offline",
                "pids": [],
                "cpu": 0,
                "mem": 0,
                "msg": "未运行",
            }

        for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_percent", "create_time"]):
            try:
                pinfo = proc.info
                cmd = " ".join(pinfo["cmdline"] or [""]).lower()
                name = (pinfo["name"] or "").lower()

                # 检查每个 Agent 的关键词
                for aname, info in AGENTS.items():
                    for kw in info["process_keywords"]:
                        if kw in cmd or kw in name:
                            aid = info["id"]
                            agents[aid]["status"] = "running"
                            agents[aid]["pids"].append(pinfo["pid"])
                            agents[aid]["cpu"] += pinfo["cpu_percent"] or 0
                            agents[aid]["mem"] += pinfo["memory_percent"] or 0
                            agents[aid]["msg"] = f"PID {pinfo['pid']} · 运行中"
                            break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # 额外检测：Hermes gateway 进程
        herm = agents.get("nannan", {})
        if herm.get("status") == "running":
            # 检查 gateway 是否在监听
            for conn in psutil.net_connections():
                if conn.status == "LISTEN" and conn.laddr.port in [9090, 9091]:
                    herm["sub_status"] = "gateway ✅"
                    break
            else:
                herm["sub_status"] = "gateway ⏳"

        return agents

    def _get_system_info(self) -> dict:
        """系统资源"""
        try:
            cpu = psutil.cpu_percent(interval=0.3)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("C:\\")
            boot = datetime.fromtimestamp(psutil.boot_time())
            return {
                "cpu": cpu,
                "memory": mem.percent,
                "mem_used_gb": round(mem.used / 1024**3, 1),
                "mem_total_gb": round(mem.total / 1024**3, 1),
                "disk": disk.percent,
                "boot_time": boot.strftime("%H:%M:%S"),
                "process_count": len(list(psutil.process_iter())),
            }
        except:
            return {"cpu": 0, "memory": 0, "disk": 0}

    def _scan_ports(self) -> list:
        """扫描关键端口"""
        ports = []
        for conn in psutil.net_connections():
            if conn.status == "LISTEN" and conn.laddr.port in PORT_AGENTS:
                try:
                    pname = psutil.Process(conn.pid).name() if conn.pid else "?"
                except:
                    pname = "?"
                ports.append({
                    "port": conn.laddr.port,
                    "service": PORT_AGENTS.get(conn.laddr.port, "?"),
                    "pid": conn.pid,
                    "process": pname,
                })
        return sorted(ports, key=lambda x: x["port"])

    def get_full_status(self) -> dict:
        """合并进程状态 + 共享状态 + 余额 + 系统信息"""
        from interfaces.dashboard.state import DashboardState
        from interfaces.dashboard.balances import get_all_balances

        # 基础状态（进程扫描+系统+端口）
        base = self.poll()

        # 读取共享状态（任务描述、管线）
        state = DashboardState.get_state()

        # 合并 Agent 状态：进程检测只判断是否活着，任务描述从 state.json 读取
        for aid, agent_info in base["agents"].items():
            task_info = state.get("agents", {}).get(aid, {})
            agent_info["task"] = task_info.get("task", "")
            agent_info["work_status"] = task_info.get("status", "idle")
            # 如果进程离线但 state 显示工作中，标记为离线
            if agent_info["status"] == "offline":
                agent_info["work_status"] = "offline"
            elif agent_info["status"] == "running":
                # 进程在跑，但实际任务可能是 idle 或 working
                pass

        # 添加管线
        base["pipeline"] = state.get("pipeline", {"title": "", "steps": []})

        # 查询余额
        base["balances"] = get_all_balances()

        base["timestamp"] = datetime.now().strftime("%H:%M:%S")
        return base


# ── 全局单例 ──
_monitor = None


def get_monitor() -> ProcessMonitor:
    global _monitor
    if _monitor is None:
        _monitor = ProcessMonitor()
    return _monitor
