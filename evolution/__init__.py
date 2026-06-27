"""
进化引擎 — 审美自进化 + 剪辑规则自进化

核心机制：
- 每次用户修正/反馈都记录到 feedback_file
- 定期总结规律，固化偏好，提出新规则提议
"""
import os
import json
import time
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional

from config_loader import get, resolve_path


class AestheticEvolution:
    """审美自进化引擎"""

    def __init__(self):
        self.feedback_file = Path(resolve_path(
            get("evolution.aesthetic.feedback_file", "data/user_prefs/aesthetic_memory.yaml")
        ))
        self.summarize_interval = get("evolution.aesthetic.summarize_interval", 10)
        self._history = self._load_history()
        self._session_count = len(self._history.get("sessions", []))
        self._rag_engine = None  # RAG 引擎引用，注入后自动记录

    def set_rag_engine(self, engine):
        """注入 RAG 引擎，使偏好记录自动注入知识库"""
        self._rag_engine = engine

    def _record_to_rag(self, category, key, value, context=""):
        """将用户偏好自动写入 RAG 索引"""
        if not self._rag_engine:
            return
        content = (
            f"# 用户偏好: {category}/{key}\n\n"
            f"- 选择值: {value}\n"
            f"- 上下文: {context}\n"
            f"- 时间: {time.strftime('%Y-%m-%d %H:%M')}\n"
        )
        self._rag_engine.ingest_and_persist(
            title=f"偏好_{category}_{key}",
            content=content,
            category="user_prefs",
            tags=[category, key],
            source="evolution"
        )

    def _load_history(self) -> dict:
        """加载历史审美数据"""
        if self.feedback_file.exists():
            try:
                import yaml
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {"preferences": {}, "sessions": [], "summary": None}

    def _save(self):
        """持久化"""
        import yaml
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.feedback_file, "w", encoding="utf-8") as f:
            yaml.dump(self._history, f, allow_unicode=True, default_flow_style=False)

    def record_choice(self, category: str, key: str, value, context: str = ""):
        """记录用户的选择，自动写入 RAG"""
        if category not in self._history["preferences"]:
            self._history["preferences"][category] = {}
        if key not in self._history["preferences"][category]:
            self._history["preferences"][category][key] = []

        entry = {
            "value": value,
            "context": context,
            "timestamp": time.time(),
        }
        self._history["preferences"][category][key].append(entry)
        self._session_count += 1
        self._save()

        # 自动写入 RAG
        self._record_to_rag(category, key, str(value), context)

        # 触发总结
        if self._session_count % self.summarize_interval == 0:
            return self.summarize(category)
        return None

    def record_feedback(self, category: str, old_value, new_value, reason: str = ""):
        """记录用户修正（"太丑了" → 反馈信号），自动写入 RAG"""
        entry = {
            "type": "correction",
            "old": old_value,
            "new": new_value,
            "reason": reason,
            "timestamp": time.time(),
        }
        if "corrections" not in self._history:
            self._history["corrections"] = []
        self._history["corrections"].append({
            "category": category,
            "entry": entry,
        })
        self._save()

        # 自动写入 RAG
        self._record_to_rag(category, "feedback", f"{old_value}→{new_value}", reason)

    def get_preferred(self, category: str, key: str, default=None):
        """获取最常选的值"""
        records = self._history.get("preferences", {}).get(category, {}).get(key, [])
        if not records:
            return default
        # 统计各值出现次数
        counts = defaultdict(int)
        times = defaultdict(float)  # 最近时间
        for r in records:
            val = str(r.get("value", ""))
            counts[val] += 1
            times[val] = max(times[val], r.get("timestamp", 0))
        # 加权：频率 * 80% + 新鲜度 * 20%
        total = sum(counts.values()) or 1
        max_time = max(times.values()) if times else 0
        scores = {}
        for val in counts:
            freq_score = counts[val] / total
            recency_score = (times[val] / max_time) if max_time > 0 else 0
            scores[val] = freq_score * 0.8 + recency_score * 0.2
        return max(scores, key=scores.get)

    def summarize(self, category: str = None) -> str:
        """总结审美偏好趋势"""
        prefs = self._history.get("preferences", {})
        if category:
            prefs = {category: prefs.get(category, {})}

        lines = ["📊 审美趋势报告"]
        for cat, data in prefs.items():
            lines.append(f"\n  [{cat}]")
            for key, records in data.items():
                if not records:
                    continue
                # 最近3次
                last3 = records[-3:]
                values = [str(r["value"]) for r in last3]
                # 判断稳定性
                unique = list(set(values))
                if len(unique) == 1:
                    lines.append(f"    {key}: 「{unique[0]}」(稳定)")
                elif len(unique) <= 2:
                    lines.append(f"    {key}: 最近 = {' → '.join(values)} (在收敛)")
                else:
                    lines.append(f"    {key}: 最近 = {' → '.join(values)} (不稳定)")

        self._history["summary"] = {
            "time": time.time(),
            "content": "\n".join(lines),
        }
        self._save()
        return "\n".join(lines)


class ClipRuleEvolution:
    """剪辑规则自进化引擎"""

    def __init__(self):
        self.feedback_file = Path(resolve_path(
            get("evolution.clip_rules.feedback_file", "data/user_prefs/clip_evolution.yaml")
        ))
        self.auto_propose = get("evolution.clip_rules.auto_propose", True)
        self._history = self._load_history()
        self._rag_engine = None  # RAG 引擎引用

    def set_rag_engine(self, engine):
        """注入 RAG 引擎，使剪辑偏好自动注入知识库"""
        self._rag_engine = engine

    def _record_to_rag(self, param, old_val, new_val, context=""):
        """将剪辑参数调整自动写入 RAG 索引"""
        if not self._rag_engine:
            return
        content = (
            f"# 用户剪辑偏好\n\n"
            f"- 参数: {param}\n"
            f"- 旧值: {old_val}\n"
            f"- 新值: {new_val}\n"
            f"- 上下文: {context}\n"
            f"- 时间: {time.strftime('%Y-%m-%d %H:%M')}\n"
        )
        self._rag_engine.ingest_and_persist(
            title=f"剪辑_{param}_{int(time.time())}",
            content=content,
            category="user_clip_prefs",
            tags=["clip", param],
            source="evolution_clip"
        )

    def _load_history(self) -> dict:
        if self.feedback_file.exists():
            try:
                import yaml
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {"parameter_adjustments": [], "new_patterns": [], "rules": []}

    def _save(self):
        import yaml
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.feedback_file, "w", encoding="utf-8") as f:
            yaml.dump(self._history, f, allow_unicode=True, default_flow_style=False)

    def record_adjustment(self, param: str, old_val, new_val, context: str = ""):
        """记录用户参数调整，自动写入 RAG"""
        self._history["parameter_adjustments"].append({
            "param": param,
            "from": old_val,
            "to": new_val,
            "context": context,
            "timestamp": time.time(),
        })
        self._save()

        # 自动写入 RAG
        self._record_to_rag(param, old_val, new_val, context)

        return self._detect_pattern(param)

    def record_new_pattern(self, description: str, evidence: list):
        """记录新发现的剪辑模式，自动写入 RAG"""
        self._history["new_patterns"].append({
            "description": description,
            "evidence": evidence,
            "proposed": time.time(),
            "accepted": None,
        })
        self._save()

        # 自动写入 RAG
        if self._rag_engine:
            evidence_str = "\n".join(f"  - {e}" for e in evidence[:5])
            content = (
                f"# 用户发现的新剪辑模式\n\n"
                f"描述: {description}\n\n"
                f"证据:\n{evidence_str}\n"
                f"时间: {time.strftime('%Y-%m-%d %H:%M')}\n"
            )
            self._rag_engine.ingest_and_persist(
                title=f"剪辑模式_{int(time.time())}",
                content=content,
                category="user_clip_patterns",
                tags=["clip_pattern"],
                source="evolution_clip"
            )

    def record_custom_rule(self, rule_name: str, rule_def: dict):
        """记录用户自定义规则，自动写入 RAG"""
        self._history["rules"].append({
            "name": rule_name,
            "definition": rule_def,
            "timestamp": time.time(),
        })
        self._save()

        # 自动写入 RAG
        if self._rag_engine:
            import json
            content = (
                f"# 用户自定义剪辑规则\n\n"
                f"规则名: {rule_name}\n\n"
                f"定义:\n```json\n{json.dumps(rule_def, ensure_ascii=False, indent=2)}\n```\n"
                f"时间: {time.strftime('%Y-%m-%d %H:%M')}\n"
            )
            self._rag_engine.ingest_and_persist(
                title=f"剪辑规则_{rule_name}",
                content=content,
                category="user_clip_rules",
                tags=["clip_rule", rule_name],
                source="evolution_clip"
            )

    def _detect_pattern(self, param: str) -> Optional[dict]:
        """检测参数调整是否有规律"""
        adjustments = [a for a in self._history["parameter_adjustments"]
                       if a["param"] == param]
        if len(adjustments) < 3:
            return None

        # 最近3次调整方向
        recent = adjustments[-3:]
        directions = []
        for adj in recent:
            if isinstance(adj["from"], (int, float)) and isinstance(adj["to"], (int, float)):
                directions.append(adj["to"] > adj["from"])

        # 如果趋势一致，提议新规则
        if len(set(directions)) == 1:
            direction = "增加" if directions[0] else "减少"
            proposal = {
                "param": param,
                "trend": direction,
                "confidence": len(recent) / 5,
                "suggestion": f"你最近3次都在{direction}「{param}」，要不要设为新默认值？",
            }
            return proposal
        return None

    def get_suggestions(self) -> list:
        """获取所有活跃的提议"""
        suggestions = []
        # 检测参数调整规律
        seen_params = set()
        for adj in self._history.get("parameter_adjustments", []):
            p = adj["param"]
            if p in seen_params:
                continue
            seen_params.add(p)
            pattern = self._detect_pattern(p)
            if pattern:
                suggestions.append(pattern)

        # 新发现的模式
        for pattern in self._history.get("new_patterns", []):
            if pattern.get("accepted") is None:
                suggestions.append({
                    "type": "new_rule",
                    "description": pattern["description"],
                    "evidence": pattern.get("evidence", []),
                })

        return suggestions[-3:]  # 最多返回3个
