#!/usr/bin/env python3
"""
同步脚本 — 读取本地状态并推送 status.json 到 gh-pages 分支

执行流程:
1. 调用 monitor.py 获取进程/系统状态
2. 调用 balances.py 获取余额
3. 读取 ~/.xiaot_agent/dashboard_state.json 获取管线/任务描述
4. 合并成 status.json
5. 通过 gh CLI API 推送到 gh-pages 分支的 data/status.json

不包含完整 API Key，只保留余额数字。
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ── 项目路径 ──
# 兼容两种运行方式: (1) scripts/sync_dashboard.py (2) ~/hermes/scripts/sync_dashboard.py
_script_path = Path(__file__).resolve()
if _script_path.parent.name == "scripts" and _script_path.parent.parent.name != "hermes":
    # 运行在项目 scripts/ 目录下
    PROJECT_ROOT = _script_path.parent.parent.resolve()
else:
    # 运行在 hermes 脚本目录下，硬编码项目路径
    PROJECT_ROOT = Path(r"E:\任务\小红书内容工坊 Agent").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ── GitHub 参数 ──
OWNER = "maomaozuiwudi"
REPO = "xiaot-agent"
BRANCH = "gh-pages"
STATUS_PATH = "data/status.json"
STATE_FILE = Path.home() / ".xiaot_agent" / "dashboard_state.json"


def log(msg: str):
    """带时间戳的日志"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def gh_api(method: str, endpoint: str, data: dict = None, jq: str = None) -> str:
    """调用 gh CLI API"""
    cmd = ["gh", "api", endpoint, "--method", method]
    if jq:
        cmd.extend(["--jq", jq])

    input_data = json.dumps(data) if data else None
    result = subprocess.run(cmd, capture_output=True, text=True, input=input_data)
    if result.returncode != 0:
        raise RuntimeError(f"gh API failed: {result.stderr[:200]}")
    return result.stdout.strip()


def collect_status() -> dict:
    """收集所有状态数据"""
    log("收集状态数据...")

    # 1. 进程监控
    try:
        from interfaces.dashboard.monitor import ProcessMonitor
        monitor = ProcessMonitor()
        base_status = monitor.poll()
        log(f"  ✓ 进程扫描完成: {len(base_status.get('agents', {}))} agents")
    except Exception as e:
        log(f"  ⚠ 进程扫描失败: {e}")
        base_status = {
            "agents": {},
            "system": {"cpu": 0, "memory": 0, "disk": 0, "process_count": 0},
            "ports": [],
        }

    # 2. 共享状态（管线 + Agent 任务描述）
    try:
        from interfaces.dashboard.state import DashboardState
        state = DashboardState.get_state()
        for aid, agent_info in base_status.get("agents", {}).items():
            task_info = state.get("agents", {}).get(aid, {})
            agent_info["task"] = task_info.get("task", "")
            agent_info["work_status"] = task_info.get("status", "idle")
            if agent_info.get("status") == "offline":
                agent_info["work_status"] = "offline"
            elif agent_info.get("status") == "running" and agent_info.get("work_status") != "working":
                agent_info["work_status"] = "working" if agent_info.get("task") else "idle"
        base_status["pipeline"] = state.get("pipeline", {"title": "", "steps": []})
        log(f"  ✓ 共享状态读取完成")
    except Exception as e:
        log(f"  ⚠ 共享状态读取失败: {e}")
        base_status["pipeline"] = {"title": "", "steps": []}

    # 3. 余额（不含完整 Key）
    try:
        from interfaces.dashboard.balances import get_all_balances
        balances = get_all_balances()
        # 确保只保留余额数字，不包含 API Key 痕迹
        clean_balances = {}
        for platform, info in balances.items():
            clean_balances[platform] = {
                "balance": info.get("balance", "--"),
                "status": info.get("status", "⚪"),
            }
        base_status["balances"] = clean_balances
        log(f"  ✓ 余额查询完成: {len(clean_balances)} 平台")
    except Exception as e:
        log(f"  ⚠ 余额查询失败: {e}")
        base_status["balances"] = {}

    # 4. 时间戳
    base_status["timestamp"] = datetime.now().strftime("%H:%M:%S")
    base_status["_sync_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return base_status


def push_to_gh_pages(status_data: dict) -> bool:
    """通过 gh API 推送 status.json 到 gh-pages"""
    log("推送到 gh-pages...")

    # Step 1: 获取当前 gh-pages 分支的最新 commit
    try:
        latest_commit = gh_api(
            "GET",
            f"repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
            jq=".object.sha"
        )
        log(f"  ✓ 当前 ref commit: {latest_commit[:12]}...")
    except RuntimeError as e:
        log(f"  ✗ 获取 ref 失败: {e}")
        return False

    # Step 2: 获取当前分支的 tree SHA
    try:
        commit_data_raw = subprocess.run(
            ["gh", "api", f"repos/{OWNER}/{REPO}/git/commits/{latest_commit}", "--jq", ".tree.sha"],
            capture_output=True, text=True
        )
        parent_tree_sha = commit_data_raw.stdout.strip()
        log(f"  ✓ 父 tree SHA: {parent_tree_sha[:12]}...")
    except Exception as e:
        log(f"  ✗ 获取 tree 失败: {e}")
        return False

    # Step 3: 创建 status.json blob
    status_json = json.dumps(status_data, ensure_ascii=False, indent=2)
    b64_content = base64.b64encode(status_json.encode()).decode()
    try:
        blob_sha = gh_api(
            "POST",
            f"repos/{OWNER}/{REPO}/git/blobs",
            data={"content": b64_content, "encoding": "base64"},
            jq=".sha"
        )
        log(f"  ✓ status.json blob: {blob_sha[:12]}... ({len(status_json)} bytes)")
    except RuntimeError:
        # Try --field approach as alternative
        result = subprocess.run(
            ["gh", "api", f"repos/{OWNER}/{REPO}/git/blobs",
             "--field", f"content={b64_content}",
             "--field", "encoding=base64",
             "--jq", ".sha"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            log(f"  ✗ 创建 blob 失败: {result.stderr[:200]}")
            return False
        blob_sha = result.stdout.strip()
        log(f"  ✓ status.json blob (alt): {blob_sha[:12]}...")

    # Step 4: 创建新 tree（只更新 status.json）
    tree_payload = {
        "base_tree": parent_tree_sha,
        "tree": [
            {
                "path": STATUS_PATH,
                "mode": "100644",
                "type": "blob",
                "sha": blob_sha,
            }
        ],
    }
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{OWNER}/{REPO}/git/trees",
             "--input", "-", "--jq", ".sha"],
            capture_output=True, text=True,
            input=json.dumps(tree_payload)
        )
        if result.returncode != 0:
            log(f"  ✗ 创建 tree 失败: {result.stderr[:200]}")
            return False
        new_tree_sha = result.stdout.strip()
        log(f"  ✓ 新 tree SHA: {new_tree_sha[:12]}...")
    except Exception as e:
        log(f"  ✗ 创建 tree 异常: {e}")
        return False

    # Step 5: 创建 commit
    commit_payload = {
        "message": f"sync: 更新仪表盘状态 @ {status_data.get('_sync_time', '')}",
        "tree": new_tree_sha,
        "parents": [latest_commit],
        "author": {"name": "xiaot-agent-sync", "email": "sync@xiaot-agent"},
        "committer": {"name": "xiaot-agent-sync", "email": "sync@xiaot-agent"},
    }
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{OWNER}/{REPO}/git/commits",
             "--input", "-", "--jq", ".sha"],
            capture_output=True, text=True,
            input=json.dumps(commit_payload)
        )
        if result.returncode != 0:
            log(f"  ✗ 创建 commit 失败: {result.stderr[:200]}")
            return False
        new_commit_sha = result.stdout.strip()
        log(f"  ✓ 新 commit SHA: {new_commit_sha[:12]}...")
    except Exception as e:
        log(f"  ✗ 创建 commit 异常: {e}")
        return False

    # Step 6: 更新 ref
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
             "--method", "PATCH",
             "--field", f"sha={new_commit_sha}",
             "--field", "force=true",
             "--jq", ".ref"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            log(f"  ✗ 更新 ref 失败: {result.stderr[:200]}")
            return False
        log(f"  ✅ Ref 已更新!")
        return True
    except Exception as e:
        log(f"  ✗ 更新 ref 异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="同步仪表盘状态到 gh-pages")
    parser.add_argument("--once", action="store_true", help="只运行一次，不循环")
    parser.add_argument("--interval", type=int, default=60, help="同步间隔（秒）")
    args = parser.parse_args()

    log(f"🚀 小t Agent 仪表盘同步脚本启动")
    log(f"   项目: {OWNER}/{REPO}")
    log(f"   分支: {BRANCH}")
    log(f"   状态文件: {STATE_FILE}")

    if args.once:
        # 单次运行
        try:
            status = collect_status()
            ok = push_to_gh_pages(status)
            if ok:
                log("✅ 同步成功!")
                return 0
            else:
                log("❌ 同步失败")
                return 1
        except Exception as e:
            log(f"❌ 异常: {e}")
            return 1

    # 循环模式
    log(f"   间隔: {args.interval}s (按 Ctrl+C 停止)\n")
    while True:
        try:
            start = time.time()
            status = collect_status()
            ok = push_to_gh_pages(status)
            elapsed = time.time() - start

            if ok:
                log(f"✅ 同步成功! ({elapsed:.1f}s)\n")
            else:
                log(f"❌ 同步失败 ({elapsed:.1f}s)\n")

            # 等待剩余时间
            sleep_time = max(1, args.interval - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            log("收到退出信号")
            break
        except Exception as e:
            log(f"❌ 异常: {e}")
            time.sleep(10)

    return 0


if __name__ == "__main__":
    import argparse
    sys.exit(main())
