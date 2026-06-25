"""
记忆管理模块 v2 — 多用户隔离 + 隐私控制

每个用户由 API Key 哈希标识，数据隔离存储。
支持两种隐私模式：
  sync  = 本地记录 + 匿名汇入共享库 + 读取共享优化
  local = 仅本地记录，不汇出不读取共享
"""
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Optional
import yaml

from config_loader import get, resolve_path


def key_to_uid(api_key: str) -> str:
    """API Key → 用户 UID（SHA256 前 16 位）"""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


class UserManager:
    """用户管理器 — 负责登录/注册/隐私设置"""

    def __init__(self):
        self.api_key = ""
        self.uid = ""
        self.privacy_mode = "sync"    # sync | local
        self.is_new_user = False
        self.first_interaction = True

    def login(self, api_key: str, privacy_mode: str = None) -> str:
        """
        用户登录/注册
        privacy_mode: sync=共享 | local=私有
        返回欢迎消息
        """
        self.api_key = api_key
        self.uid = key_to_uid(api_key)

        user_dir = Path(resolve_path(f"data/users/{self.uid}"))
        is_new = not user_dir.exists()

        if is_new:
            user_dir.mkdir(parents=True, exist_ok=True)
            self.is_new_user = True
            # 写用户档案
            profile = {
                "uid": self.uid,
                "created_at": time.time(),
                "privacy": privacy_mode or get("user.privacy.default", "sync"),
                "session_count": 0,
            }
            with open(user_dir / "profile.yaml", "w", encoding="utf-8") as f:
                yaml.dump(profile, f, allow_unicode=True)
            self.privacy_mode = privacy_mode or get("user.privacy.default", "sync")
            return self._welcome_new()
        else:
            # 读取已有档案
            profile_path = user_dir / "profile.yaml"
            if profile_path.exists():
                with open(profile_path, "r", encoding="utf-8") as f:
                    profile = yaml.safe_load(f) or {}
                self.privacy_mode = profile.get("privacy", get("user.privacy.default", "sync"))
                profile["session_count"] = profile.get("session_count", 0) + 1
                with open(profile_path, "w", encoding="utf-8") as f:
                    yaml.dump(profile, f, allow_unicode=True)
            else:
                self.privacy_mode = privacy_mode or get("user.privacy.default", "sync")

            # 如果显式要求换隐私模式
            if privacy_mode:
                self.set_privacy(privacy_mode)

            return self._welcome_back()

    def set_privacy(self, mode: str):
        """切换隐私模式"""
        if mode not in ("sync", "local"):
            return
        self.privacy_mode = mode
        # 写入档案
        user_dir = Path(resolve_path(f"data/users/{self.uid}"))
        profile_path = user_dir / "profile.yaml"
        if profile_path.exists():
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = yaml.safe_load(f) or {}
        else:
            profile = {}
        profile["privacy"] = mode
        with open(profile_path, "w", encoding="utf-8") as f:
            yaml.dump(profile, f, allow_unicode=True)

    def is_shared_mode(self) -> bool:
        """是否共享模式"""
        return self.privacy_mode == "sync"

    def _welcome_new(self) -> str:
        name = self.uid[:8]
        mode_name = "🌐 共享模式" if self.privacy_mode == "sync" else "🔒 私有模式"
        msg = (
            f"👋 新朋友 {name}！\n"
            f"当前模式: {mode_name}\n"
        )
        if self.privacy_mode == "sync":
            msg += "你的使用习惯会匿名汇入共享知识库，同时也能享受社区累积的优化经验"
        else:
            msg += "你的数据完全私有，不共享不读取"
        msg += "\n输入 /privacy 随时切换模式"
        return msg

    def _welcome_back(self) -> str:
        name = self.uid[:8]
        mode_name = "🌐 共享" if self.privacy_mode == "sync" else "🔒 私有"
        prefs_path = Path(resolve_path(f"data/users/{self.uid}/prefs.yaml"))
        pref_count = 0
        if prefs_path.exists():
            with open(prefs_path, "r") as f:
                data = yaml.safe_load(f) or {}
                pref_count = sum(len(v) for v in data.values()) if isinstance(data, dict) else 0
        msg = f"👋 欢迎回来 {name}！ [{mode_name}]"
        if pref_count > 0:
            msg += f" · 已记录 {pref_count} 个偏好"
        return msg

    def get_user_dir(self) -> Path:
        """获取用户数据目录"""
        return Path(resolve_path(f"data/users/{self.uid}"))


class MemoryManager:
    """多用户记忆管理器"""

    def __init__(self, user_mgr: UserManager):
        self.uid = user_mgr.uid
        self.is_shared = user_mgr.is_shared_mode()
        self.user_dir = user_mgr.get_user_dir()
        self.user_dir.mkdir(parents=True, exist_ok=True)

        # 共享数据目录
        self.shared_dir = Path(resolve_path("data/shared/"))
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.rag_shared_dir = self.shared_dir / "rag"
        self.rag_shared_dir.mkdir(exist_ok=True)
        self.anony_dir = self.shared_dir / "anonymous"
        self.anony_dir.mkdir(exist_ok=True)

        self.session_id = str(int(time.time()))
        self.short_term = []
        self._load_personal()

    # ── 短期记忆 ──

    def add_message(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content, "time": time.time()})

    def get_context(self, max_turns: int = 200) -> list:
        return [{"role": m["role"], "content": m["content"]} for m in self.short_term[-max_turns:]]

    def clear_session(self):
        self._save_session_history()
        self.short_term = []

    # ── 个人偏好（按 UID 隔离） ──

    def _load_personal(self):
        """读取个人偏好"""
        self.prefs = {}
        self.aesthetic_history = []
        self.clip_history = []

        prefs_file = self.user_dir / "prefs.yaml"
        if prefs_file.exists():
            try:
                with open(prefs_file, "r", encoding="utf-8") as f:
                    self.prefs = yaml.safe_load(f) or {}
            except Exception:
                self.prefs = {}

        aesthetic_file = self.user_dir / "aesthetic.yaml"
        if aesthetic_file.exists():
            try:
                with open(aesthetic_file, "r", encoding="utf-8") as f:
                    self.aesthetic_history = yaml.safe_load(f) or []
            except Exception:
                pass

        clip_file = self.user_dir / "clip_evolution.yaml"
        if clip_file.exists():
            try:
                with open(clip_file, "r", encoding="utf-8") as f:
                    self.clip_history = yaml.safe_load(f) or []
            except Exception:
                pass

    def save_pref(self, category: str, key: str, value):
        """保存个人偏好"""
        if category not in self.prefs:
            self.prefs[category] = {}
        self.prefs[category][key] = {
            "value": value,
            "timestamp": time.time(),
        }
        self._flush_prefs()

        # 共享模式 → 匿名汇入
        if self.is_shared:
            self._anonymize_pref(category, key, value)

    def get_pref(self, category: str, key: str, default=None):
        return self.prefs.get(category, {}).get(key, {}).get("value", default)

    def get_all_prefs(self, category: str = None) -> dict:
        if category:
            return self.prefs.get(category, {})
        return self.prefs

    def _flush_prefs(self):
        filepath = self.user_dir / "prefs.yaml"
        data = {k: v for k, v in self.prefs.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    # ── 匿名共享（仅共享模式） ──

    def _anonymize_pref(self, category: str, key: str, value):
        """匿名汇入共享库"""
        anony_file = self.anony_dir / f"{category}.jsonl"
        entry = {
            "key": key,
            "value": str(value) if not isinstance(value, (int, float)) else value,
            "time": int(time.time()),
        }
        with open(anony_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_shared_trends(self, category: str, top_k: int = 5) -> list:
        """
        读取共享趋势数据（匿名聚合）
        返回高频选项排名
        """
        if not self.is_shared:
            return []

        anony_file = self.anony_dir / f"{category}.jsonl"
        if not anony_file.exists():
            return []

        counts = {}
        try:
            with open(anony_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        k = entry.get("key", "")
                        v = str(entry.get("value", ""))
                        if k and v:
                            counts[(k, v)] = counts.get((k, v), 0) + 1
                    except json.JSONDecodeError:
                        pass
        except Exception:
            return []

        sorted_items = sorted(counts.items(), key=lambda x: -x[1])
        return [{"key": k, "value": v, "count": c} for (k, v), c in sorted_items[:top_k]]

    def save_to_shared_rag(self, content: str, tags: list):
        """用户贡献的知识写入共享 RAG 库"""
        if not self.is_shared:
            return

        filename = f"user_{self.uid[:8]}_{int(time.time())}.md"
        filepath = self.rag_shared_dir / filename
        header = f"---\ntags: [{', '.join(tags)}]\ncontributor: anonymous\n---\n\n"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + content)

    # ── 反馈记录 ──

    def record_feedback(self, category: str, feedback: dict):
        feedback_dir = self.user_dir / "feedback"
        feedback_dir.mkdir(exist_ok=True)
        filepath = feedback_dir / f"{category}.jsonl"
        feedback["timestamp"] = time.time()
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback, ensure_ascii=False) + "\n")

    def get_feedback_history(self, category: str, limit: int = 20) -> list:
        filepath = self.user_dir / "feedback" / f"{category}.jsonl"
        if not filepath.exists():
            return []
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records[-limit:]

    # ── 会话持久化 ──

    def _save_session_history(self):
        if not self.short_term:
            return
        history_dir = self.user_dir / "sessions"
        history_dir.mkdir(parents=True, exist_ok=True)
        filepath = history_dir / f"session_{self.session_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": self.session_id,
                "messages": self.short_term,
                "time": time.time(),
            }, f, ensure_ascii=False, indent=2)

    # ── 进化数据 ──

    def record_aesthetic_choice(self, category: str, key: str, value, context: str = ""):
        """记录审美选择"""
        entry = {"category": category, "key": key, "value": value,
                 "context": context, "time": time.time()}
        self.aesthetic_history.append(entry)
        aesthetic_file = self.user_dir / "aesthetic.yaml"
        with open(aesthetic_file, "w", encoding="utf-8") as f:
            yaml.dump(self.aesthetic_history[-100:], f, allow_unicode=True)

        if self.is_shared:
            self._anonymize_pref(f"aesthetic_{category}", key, value)

    def record_clip_adjustment(self, param: str, old_val, new_val, context: str = ""):
        """记录剪辑参数调整"""
        entry = {"param": param, "from": old_val, "to": new_val,
                 "context": context, "time": time.time()}
        self.clip_history.append(entry)
        clip_file = self.user_dir / "clip_evolution.yaml"
        with open(clip_file, "w", encoding="utf-8") as f:
            yaml.dump(self.clip_history[-100:], f, allow_unicode=True)

        if self.is_shared:
            self._anonymize_pref("clip_params", param, new_val)
