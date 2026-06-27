# 仪表盘 v2 设计方案

## 架构

```
┌─────────────────────────────────────────┐
│           仪表盘 (FastAPI :7861)          │
│  ┌────────┐  ┌────────┐  ┌──────────┐   │
│  │ 管线图  │  │ 状态卡  │  │ Key余额   │   │
│  └────────┘  └────────┘  └──────────┘   │
└──────────────────────┬──────────────────┘
                       │ /api/status
┌──────────────────────▼──────────────────┐
│         monitor.py (进程+状态采集)         │
│  ┌──────────┐  ┌──────────┐  ┌──────┐   │
│  │ 进程扫描   │  │ state.json│  │余额API│   │
│  └──────────┘  └──────────┘  └──────┘   │
└─────────────────────────────────────────┘
```

## 1. 状态共享系统

文件: `~/.xiaot_agent/dashboard_state.json`
格式:
```json
{
  "pipeline": {
    "title": "穿搭视频制作",
    "steps": [
      {"id": 1, "label": "需求解析", "agent": "楠楠", "status": "done"},
      {"id": 2, "label": "素材分析", "agent": "Kimi", "status": "working", "detail": "分析6段穿搭视频"},
      {"id": 3, "label": "骨架剪辑", "agent": "CC", "status": "pending"},
      {"id": 4, "label": "文案生成", "agent": "楠楠", "status": "pending"},
      {"id": 5, "label": "视频合成", "agent": "CC", "status": "pending"}
    ]
  },
  "agents": {
    "nannan": {"status": "working", "task": "分析视频画面"},
    "cc": {"status": "idle", "task": ""},
    "codex": {"status": "idle", "task": ""},
    "kimi": {"status": "idle", "task": ""}
  }
}
```

写入方式：谁干活谁写，一行代码的事：
```python
import json
Path.home()/".xiaot_agent"/"dashboard_state.json").write_text(json.dumps(state))
```

## 2. 余额查询

需要查的 Key 和接口：

| 平台 | API 端点 | Auth |
|------|---------|------|
| DeepSeek | GET https://api.deepseek.com/user/balance | Bearer Token |
| Kimi/Moonshot | GET https://api.moonshot.cn/v1/billing/balance | Bearer Token |
| SiliconFlow | GET https://api.siliconflow.com/v1/user/balance | Bearer Token |
| 火山引擎/豆包 | GET https://console.volcengine.com/api/billing | x-api-key |
| Rnote | GET https://rnote.dev/api/v2/user/profile | Bearer Token |

Key 来源：从各配置文件读取（config.yaml, .env, .volc_voice_key 等）

## 3. 前端管线图

SVG 流程图，风格参考电路管线：

```
[需求输入] ──→ [楠楠·调度] ──→ [CC/Codex/Kimi·执行] ──→ [汇总] ──→ [✓ 产出]
    ↑               ↑              ↑              ↑          ↑
  绿/黄/灰        状态+任务      状态+任务      状态       结果
```

每个节点颜色：🟢已完成 🟡工作中 ⚪待处理 🔴失败

## 4. 手机适配

- 卡片竖排（管线图在上，Agent下面，Key余额最下）
- 触控友好
- PWA manifest 已配，可添加到主屏幕

## 实施步骤

1. 创建 `interfaces/dashboard/state.py` — 状态读写+余额查询
2. 更新 `interfaces/dashboard/monitor.py` — 集成 state + balance
3. 重建 `interfaces/dashboard/static/dashboard.html` — 管线图+状态卡+余额
4. 更新 `interfaces/dashboard/server.py` — 新 API
