"""
MCP Client Manager — 管理 MCP server 连接与工具调用
======================================================
同步包装：使用后台线程中的持久事件循环，避免每次 asyncio.run() 创建新 loop
导致流对象失效的问题。
"""
import asyncio
import json
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP SDK 未安装，MCP 功能不可用")


class MCPManager:
    """
    管理多个 MCP server 连接，提供同步接口。

    内部维护一个后台线程 + 持久事件循环，所有异步操作在同一个 loop 上执行，
    避免跨 loop 使用流对象的问题。
    """

    def __init__(self, servers_config: dict):
        """
        Args:
            servers_config: 格式 {server_name: {command: str, args: list[str]}, ...}
        """
        self._servers = servers_config or {}
        self._sessions: dict[str, ClientSession] = {}
        self._streams: dict[str, tuple] = {}  # keep references alive
        self._stdio_cms: dict[str, object] = {}  # keep async context managers alive
        self._tools_cache: dict[str, list] = {}

        # 持久事件循环（后台线程）
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="mcp-event-loop",
        )
        self._thread.start()

    def _run(self, coro, timeout: float = 60):
        """在后台事件循环上执行协程并同步等待结果"""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _stop_loop(self):
        """停止后台事件循环"""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    # ── 生命周期 ──

    def connect_all(self) -> None:
        """连接所有已配置的 MCP server（失败时静默跳过）"""
        if not MCP_AVAILABLE:
            logger.warning("MCP SDK 未安装，跳过 MCP server 连接")
            return
        if not self._servers:
            return

        logger.info("正在连接 %d 个 MCP server...", len(self._servers))
        self._run(self._connect_all_async())

    async def _connect_all_async(self):
        for name, cfg in self._servers.items():
            try:
                session = await self._connect_one_async(name, cfg)
                self._sessions[name] = session
                logger.info("MCP server '%s' 连接成功", name)
            except Exception as e:
                logger.warning("MCP server '%s' 连接失败: %s", name, e)

    async def _connect_one_async(self, name: str, cfg: dict) -> ClientSession:
        """（异步）连接单个 MCP server"""
        command = cfg.get("command", "")
        cmd_args = cfg.get("args", [])
        if not command:
            raise ValueError(f"server '{name}' 缺少 command")

        params = StdioServerParameters(command=command, args=cmd_args)
        cm = stdio_client(params)
        streams = await cm.__aenter__()
        reader, writer = streams
        session = await ClientSession(reader, writer).__aenter__()
        await session.initialize()
        self._streams[name] = streams
        self._stdio_cms[name] = cm  # 保持引用，防止 aclose()
        return session

    def disconnect_all(self) -> None:
        """断开所有 MCP server 连接"""
        for name in list(self._sessions.keys()):
            try:
                self._run(self._disconnect_session_async(name))
            except Exception:
                pass
        self._sessions.clear()
        self._streams.clear()
        self._stdio_cms.clear()
        self._tools_cache.clear()
        logger.info("所有 MCP server 已断开")

    async def _disconnect_session_async(self, name: str):
        """（异步）断开单个 session"""
        session = self._sessions.get(name)
        if session:
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
        cm = self._stdio_cms.get(name)
        if cm:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass
        streams = self._streams.get(name)
        if streams:
            try:
                reader, writer = streams
                writer.close()
            except Exception:
                pass

    def __del__(self):
        """清理资源"""
        try:
            self.disconnect_all()
            self._stop_loop()
        except Exception:
            pass

    # ── 工具列表 ──

    def list_tools(self, server_name: Optional[str] = None) -> list[dict]:
        """
        列出 MCP 工具。

        Args:
            server_name: 若指定，只列该 server 的工具；None 则列全部

        Returns:
            [{name, description, inputSchema, server_name}, ...]
        """
        if not self._sessions:
            return []
        return self._run(self._list_tools_async(server_name))

    async def _list_tools_async(self, server_name: Optional[str] = None) -> list[dict]:
        if server_name:
            session = self._sessions.get(server_name)
            if not session:
                return []
            result = await session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.inputSchema if hasattr(t, 'inputSchema') else {"type": "object", "properties": {}},
                    "server_name": server_name,
                }
                for t in result.tools
            ]
        else:
            all_tools = []
            for srv_name, session in self._sessions.items():
                result = await session.list_tools()
                for t in result.tools:
                    all_tools.append({
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.inputSchema if hasattr(t, 'inputSchema') else {"type": "object", "properties": {}},
                        "server_name": srv_name,
                    })
            return all_tools

    def get_all_tools(self) -> list[dict]:
        """返回所有 server 的工具列表（同 list_tools）"""
        return self.list_tools()

    # ── 调用工具 ──

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """
        调用 MCP 工具（同步包装）。

        Returns:
            {"success": True/False, "result": str, "metadata": {...}}
        """
        session = self._sessions.get(server_name)
        if not session:
            logger.error("MCP server '%s' 未连接", server_name)
            return {
                "success": False,
                "result": f"MCP server '{server_name}' 未连接",
                "metadata": {"error": f"Server '{server_name}' not connected"},
            }

        try:
            result = self._run(
                self._call_tool_async(session, tool_name, arguments)
            )
        except Exception as e:
            logger.exception("MCP 工具 '%s' 调用失败", tool_name)
            return {
                "success": False,
                "result": f"MCP 工具 '{tool_name}' 调用失败: {e}",
                "metadata": {"error": str(e)},
            }

        # 处理 MCP 返回结果
        if hasattr(result, 'content') and result.content:
            text_parts = []
            for item in result.content:
                if hasattr(item, 'text') and item.text:
                    text_parts.append(item.text)
                elif hasattr(item, 'data'):
                    text_parts.append(f"[数据: {len(str(item.data))} bytes]")
            result_text = "\n".join(text_parts) if text_parts else str(result)
        else:
            result_text = str(result)

        return {
            "success": True,
            "result": result_text,
            "metadata": {
                "server": server_name,
                "tool": tool_name,
                "raw": str(result),
            },
        }

    async def _call_tool_async(self, session: ClientSession, tool_name: str, arguments: dict):
        return await session.call_tool(tool_name, arguments)

    # ── 查询 ──

    def is_connected(self) -> bool:
        """是否有任何 server 已连接"""
        return len(self._sessions) > 0

    def connected_servers(self) -> list[str]:
        """返回已连接的 server 名称列表"""
        return list(self._sessions.keys())


# ======================================================================
# MCP 工具 Handler 工厂
# ======================================================================

def make_mcp_handler(mcp_manager: MCPManager, server_name: str, tool_name: str):
    """
    创建 MCP 工具的 handler 闭包，供 ToolRegistry 注册使用。

    签名: (args: dict) -> (result_text: str, metadata: dict)
    """
    def handler(args: dict) -> tuple[str, dict]:
        try:
            result = mcp_manager.call_tool(server_name, tool_name, args)
            text = result.get("result", json.dumps(result, ensure_ascii=False))
            if result.get("success"):
                return (text, {
                    "status": "ok",
                    "server": server_name,
                    "tool": tool_name,
                    "result": result,
                })
            else:
                return (f"MCP 工具 '{tool_name}' 调用失败: {result.get('result', '')}", {
                    "status": "error",
                    "error": result.get("result", ""),
                    "server": server_name,
                    "tool": tool_name,
                })
        except Exception as e:
            logger.exception("MCP handler 异常: %s/%s", server_name, tool_name)
            return (f"MCP 工具 '{tool_name}' 调用异常: {e}", {
                "status": "error",
                "error": str(e),
                "server": server_name,
                "tool": tool_name,
            })

    handler.__name__ = f"_mcp_handler_{server_name}_{tool_name}"
    return handler
