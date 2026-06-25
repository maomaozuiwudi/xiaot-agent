"""
hallucination.py — Hallucination Defense System

Fact anchoring, confidence marking, and honesty-prompt injection for LLM
outputs in the content factory AI agent pipeline.

Three capabilities:
  1. annotate_source()   — attach 📖💡⚠️ markers based on factual support
  2. verify_against_anchor() — check AI output against known facts
  3. force_honest()           — inject honesty constraints into system prompt
"""

from __future__ import annotations

import re
from typing import Any

from config_loader import get as cfg


# ---------------------------------------------------------------------------
# Anchor type constants  (match config.yaml hallucination_guard.fact_check.anchors)
# ---------------------------------------------------------------------------

ANCHOR_SKELETON = "skeleton_data"
ANCHOR_VISION = "kimi_vision_output"
ANCHOR_CONFIG = "config_defaults"
ANCHOR_REFERENCE = "reference_docs"

_VALID_ANCHORS = frozenset({
    ANCHOR_SKELETON,
    ANCHOR_VISION,
    ANCHOR_CONFIG,
    ANCHOR_REFERENCE,
})

# ---------------------------------------------------------------------------
# Confidence-level helpers
# ---------------------------------------------------------------------------

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

_CONFIDENCE_MAP: dict[str, tuple[str, str]] = {
    # anchor_type -> (confidence_label, marker)
    ANCHOR_SKELETON: (CONFIDENCE_HIGH, "📖"),
    ANCHOR_VISION: (CONFIDENCE_HIGH, "📖"),
    ANCHOR_CONFIG: (CONFIDENCE_MEDIUM, "💡"),
    ANCHOR_REFERENCE: (CONFIDENCE_MEDIUM, "💡"),
}

# Fallback markers
_MARKER_CONFIDENT = "📖"  # data-backed
_MARKER_REFERENCE = "💡"  # reference-based
_MARKER_SPECULATION = "⚠️"  # pure AI speculation


def _anchor_confidence(anchor_type: str) -> str:
    """Return the confidence label for a given anchor type."""
    return _CONFIDENCE_MAP.get(anchor_type, (CONFIDENCE_LOW, _MARKER_SPECULATION))[0]


def _anchor_marker(anchor_type: str) -> str:
    """Return the emoji marker for a given anchor type."""
    return _CONFIDENCE_MAP.get(anchor_type, (CONFIDENCE_LOW, _MARKER_SPECULATION))[1]


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Case-insensitive word-level token set (Chinese & English)."""
    # Keep Chinese characters as whole tokens, split English on whitespace/punct
    tokens: set[str] = set()
    # Chinese sequence (2+ chars)
    for match in re.finditer(r"[\u4e00-\u9fff]{2,}", text):
        tokens.add(match.group())
    # English words
    for match in re.finditer(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", text):
        tokens.add(match.group().lower())
    return tokens


def _normalize_for_compare(text: str) -> str:
    """Lowercase + strip whitespace + collapse spaces."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# HallucinationGuard
# ---------------------------------------------------------------------------

class HallucinationGuard:
    """Hallucination defense system.

    Usage
    -----
    guard = HallucinationGuard()
    annotated = guard.annotate_source(output_text, anchors={...})
    verdict = guard.verify_against_anchor(output, "skeleton_data", {...})
    system_prompt = guard.force_honest("You are a content factory...")
    """

    def __init__(self) -> None:
        self.enabled = cfg("hallucination_guard.enabled", True)
        self.confidence_marking = cfg("hallucination_guard.confidence_marking", True)

    # ------------------------------------------------------------------
    # 1. annotate_source
    # ------------------------------------------------------------------

    def annotate_source(self, output: str, anchors: dict[str, Any]) -> str:
        """Attach 📖💡⚠️ markers to output lines based on anchor support.

        Parameters
        ----------
        output : str
            Raw text from the AI model.
        anchors : dict
            Mapping of anchor_type -> anchor_data supplied by the pipeline.
            Known keys: skeleton_data, kimi_vision_output, config_defaults,
            reference_docs.

        Returns
        -------
        str
            Marked text. Lines that match a known anchor get the corresponding
            marker prepended. Lines with no anchor match get ⚠️ speculation.
        """
        if not self.enabled or not self.confidence_marking:
            return output

        if not output.strip():
            return output

        # Build a combined lookup: token -> (best_confidence, marker)
        token_markers: dict[str, tuple[str, str]] = {}

        for anchor_type, anchor_data in anchors.items():
            if anchor_type not in _VALID_ANCHORS:
                continue
            confidence = _anchor_confidence(anchor_type)
            marker = _anchor_marker(anchor_type)

            # Extract anchor data as text strings
            anchor_texts = self._flatten_anchor(anchor_data)
            for text in anchor_texts:
                for token in _tokenize(text):
                    # Keep the highest-confidence marker for each token
                    existing = token_markers.get(token)
                    if existing is None or self._rank(existing[0]) < self._rank(confidence):
                        token_markers[token] = (confidence, marker)

        if not token_markers:
            return output

        # Annotate each line
        marked_lines: list[str] = []
        for line in output.splitlines():
            if not line.strip():
                marked_lines.append(line)
                continue

            # Find the best marker for this line
            line_lower = line.lower()
            best_marker = _MARKER_SPECULATION
            best_rank = self._rank(CONFIDENCE_LOW)

            for token, (conf, marker) in token_markers.items():
                if token.lower() in line_lower:
                    rank = self._rank(conf)
                    if rank > best_rank:
                        best_rank = rank
                        best_marker = marker

            marked_lines.append(f"{best_marker} {line}")

        return "\n".join(marked_lines)

    # ------------------------------------------------------------------
    # 2. verify_against_anchor
    # ------------------------------------------------------------------

    def verify_against_anchor(
        self,
        output: str,
        anchor_type: str,
        anchor_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Check if AI output matches known facts from a specific anchor.

        Parameters
        ----------
        output : str
            The model's generated text to verify.
        anchor_type : str
            One of: skeleton_data, kimi_vision_output, config_defaults,
            reference_docs.
        anchor_data : dict
            The known-ground-truth data from that anchor source.

        Returns
        -------
        dict with keys:
            match      — bool, whether output generally aligns
            confidence — "high" | "medium" | "low"
            issues     — list[str] of specific discrepancies found
        """
        if not self.enabled:
            return {"match": True, "confidence": CONFIDENCE_MEDIUM, "issues": []}

        if anchor_type not in _VALID_ANCHORS:
            return {
                "match": False,
                "confidence": CONFIDENCE_LOW,
                "issues": [f"未知的锚点类型 '{anchor_type}'"],
            }

        confidence = _anchor_confidence(anchor_type)
        issues: list[str] = []

        flattened = self._flatten_anchor(anchor_data)
        if not flattened:
            return {
                "match": True,
                "confidence": confidence,
                "issues": [],
            }

        output_normalized = _normalize_for_compare(output)

        # --- Skeleton data: numeric/structural checks ---
        if anchor_type == ANCHOR_SKELETON:
            if isinstance(anchor_data, dict):
                for key, expected_value in anchor_data.items():
                    key_lower = key.lower()
                    # Check if key appears in output
                    if key_lower not in output_normalized:
                        continue
                    # Try to find a numeric value near the key
                    expected_str = str(expected_value).lower()
                    if expected_str not in output_normalized:
                        issues.append(
                            f"骨架数据字段 '{key}' 期望值 '{expected_value}' "
                            "未在输出中找到对应的匹配表述"
                        )

        # --- Vision output: fact-check key observations ---
        elif anchor_type == ANCHOR_VISION:
            if isinstance(anchor_data, dict):
                for obs_key, obs_value in anchor_data.items():
                    obs_str = _normalize_for_compare(str(obs_value))
                    if obs_str and obs_str not in output_normalized:
                        issues.append(
                            f"视觉观测 '{obs_key}' 的内容 ('{obs_value}') "
                            "在输出中未被体现或表述不同"
                        )

        # --- Config defaults: check parameter values ---
        elif anchor_type == ANCHOR_CONFIG:
            if isinstance(anchor_data, dict):
                for param, default_val in anchor_data.items():
                    param_str = _normalize_for_compare(str(default_val))
                    if param_str and param_str not in output_normalized:
                        # Only flag if the param name *is* mentioned
                        if param.lower() in output_normalized:
                            issues.append(
                                f"配置默认值 '{param}' = '{default_val}' "
                                "与输出中提及的表述不一致"
                            )

        # --- Reference docs: semantic alignment ---
        elif anchor_type == ANCHOR_REFERENCE:
            if isinstance(anchor_data, (list, dict)):
                ref_texts = flattened
                output_tokens = _tokenize(output)
                for ref in ref_texts:
                    ref_norm = _normalize_for_compare(ref)
                    ref_tokens = _tokenize(ref)
                    # Check overlap ratio
                    if ref_tokens:
                        overlap = len(output_tokens & ref_tokens)
                        ratio = overlap / len(ref_tokens)
                        if ratio < 0.15:
                            # Only flag if it's a key statement (non-trivial)
                            if len(ref_tokens) >= 5:
                                issues.append(
                                    f"参考文档中的陈述与输出内容重叠度低 "
                                    f"(匹配 {overlap}/{len(ref_tokens)} 个关键概念)"
                                )

        match_final = len(issues) == 0
        min_confidence = confidence if match_final else self._downgrade(confidence)

        return {
            "match": match_final,
            "confidence": min_confidence,
            "issues": issues,
        }

    # ------------------------------------------------------------------
    # 3. force_honest
    # ------------------------------------------------------------------

    def force_honest(self, system_prompt: str) -> str:
        """Inject honesty & hallucination-avoidance constraints into the
        system prompt.

        The injection is appended as a dedicated section so that it can be
        easily inspected or stripped by downstream logic.

        Parameters
        ----------
        system_prompt : str
            Original system prompt.

        Returns
        -------
        str
            Augmented system prompt with honesty guardrails.
        """
        if not self.enabled:
            return system_prompt

        honesty_rules = (
            "\n\n"
            "【诚实约束 — Hallucination Guard 注入】\n"
            "1. 你只能基于提供的上下文、配置数据、骨架检测结果、视觉输出和\n"
            "   参考文档中的事实来回答问题。\n"
            "2. 如果你不知道某个事实，请明确说「我不知道」或「根据现有信息无法确定」。\n"
            "   绝对不要编造数据、数字、来源或统计信息。\n"
            "3. 对于推测性内容（创意、建议、文艺描述），必须使用以下标识清晰标注：\n"
            "   - 📖 = 有数据/事实支撑\n"
            "   - 💡 = 基于参考/经验的推断\n"
            "   - ⚠️ = 纯 AI 推测，请用户验证\n"
            "4. 不要为没有事实依据的陈述附加虚假引用或来源。\n"
            "5. 如果用户要求的信息不在你的知识范围内，主动提出需要补充哪些资料。\n"
        )

        # Avoid duplicate injection
        if "【诚实约束 — Hallucination Guard 注入】" in system_prompt:
            return system_prompt

        return system_prompt.rstrip() + honesty_rules

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_anchor(data: Any, max_depth: int = 3) -> list[str]:
        """Recursively extract string values from an anchor data structure."""
        if max_depth <= 0:
            return []
        texts: list[str] = []
        if isinstance(data, str):
            texts.append(data)
        elif isinstance(data, (int, float, bool)):
            texts.append(str(data))
        elif isinstance(data, (list, tuple)):
            for item in data:
                texts.extend(HallucinationGuard._flatten_anchor(item, max_depth - 1))
        elif isinstance(data, dict):
            for value in data.values():
                texts.extend(HallucinationGuard._flatten_anchor(value, max_depth - 1))
        return texts

    @staticmethod
    def _rank(confidence: str) -> int:
        """Numeric rank for comparison (higher = better)."""
        return {CONFIDENCE_HIGH: 3, CONFIDENCE_MEDIUM: 2, CONFIDENCE_LOW: 1}.get(confidence, 0)

    @staticmethod
    def _downgrade(confidence: str) -> str:
        """Downgrade one level."""
        mapping = {
            CONFIDENCE_HIGH: CONFIDENCE_MEDIUM,
            CONFIDENCE_MEDIUM: CONFIDENCE_LOW,
            CONFIDENCE_LOW: CONFIDENCE_LOW,
        }
        return mapping.get(confidence, CONFIDENCE_LOW)
