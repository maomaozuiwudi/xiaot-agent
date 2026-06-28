# 🐱 小t Agent — 小红书内容创作助手

一个**越用越懂你**的 AI 内容创作助手。支持 DeepSeek/豆包/GPT/Kimi 等多种模型，多人共享记忆，隐私自选。

## 🚀 快速开始

### 方式一：一键安装（推荐）

```bash
curl -L https://raw.githubusercontent.com/maomaozuiwudi/xiaot-agent/main/install.bat | cmd
```

装完之后终端直接输入 `xiaot` 启动。

### 方式二：手动安装

```bash
git clone https://github.com/maomaozuiwudi/xiaot-agent.git
cd xiaot-agent
pip install -e .
playwright install chromium
python main.py
```

装完可输 `xiaot` 替代 `python main.py`。

### 方式三：从 GitHub 安装

```bash
pip install git+https://github.com/maomaozuiwudi/xiaot-agent.git
```

### 方式三：本地开发

```bash
git clone https://github.com/maomaozuiwudi/xiaot-agent.git
cd xiaot-agent
pip install -e .
xiaot
```

### 启动参数

```bash
xiaot              # CLI 终端对话模式（默认）
xiaot --web        # Web 服务模式（浏览器打开）
xiaot --gui        # Desktop GUI 模式
```

启动后根据提示：
1. 选择模型（DeepSeek/豆包/GPT/Kimi/通义千问…）
2. 输入你的 API Key（花自己的额度）
3. 选择隐私模式（共享 / 私有）
4. 开始聊天

## 🎯 能干嘛

跟它说就行：

| 你说 | 它做 |
|:----|:-----|
| "帮我做6个穿搭视频，35秒" | 骨架剪辑 + 视觉分析 + 出文案 + 合成 MP4 |
| "分析一下这些素材" | Kimi 看图，分析画面内容 |
| "换个风格，改成小清新" | 自动调整文案和画面风格 |
| "保存当前BGM偏好" | 记住你的选择，越用越懂 |

## 🔒 隐私

| 模式 | 你的数据 | 你能得到 |
|:----|:---------|:---------|
| 🌐 共享 | 匿名汇入共享库 | 社区积累的优化经验 |
| 🔒 私有 | 完全本地，不上传 | 基础功能 |

输入 `/privacy` 随时切换。

## 🔧 系统命令

```
/help     — 帮助
/quit     — 退出
/reset    — 重置对话
/privacy  — 切换隐私模式
/prefs    — 查看你的偏好
/trends   — 社区趋势（共享模式）
/model    — 查看模型
/skill    — 技能管理
/debug    — 调试信息
```

## 🧩 技能系统

```
/skill list              # 查看已安装技能
/skill install <路径>    # 安装技能
/skill uninstall <名称>  # 卸载技能
```

## 📦 项目结构

```
agent/
├─ brain/           # AI 大脑（对话+推理+工具调度）
├─ guard/           # 守卫系统（Critic+幻觉防御）
├─ knowledge/       # 知识系统（RAG+记忆管理）
├─ skills/          # 技能系统（可安装/卸载）
├─ evolution/       # 进化引擎（审美+剪辑自进化）
├─ interfaces/      # 交互界面（CLI/Web/GUI）
├─ config.yaml      # 配置（需自行创建）
├─ main.py          # 入口
└─ xiaot_agent_entry.py  # pip 终端入口
```

## 🐱 关于

工具猫 (maomaozuiwudi) · MIT License

GitHub: https://github.com/maomaozuiwudi/xiaot-agent
