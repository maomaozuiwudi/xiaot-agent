"""
critic.py — The Critic module

Proactively reviews user requests BEFORE execution. Checks material counts vs
target durations, resolution mismatches, missing materials, parameter conflicts,
and other feasibility concerns. Non-blocking: warns & suggests, user decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config_loader import get as cfg


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CritiqueResult:
    """Structured outcome of a Critic review."""

    is_feasible: bool = True
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def merge(self, other: "CritiqueResult") -> "CritiqueResult":
        """Combine two critique results."""
        return CritiqueResult(
            is_feasible=self.is_feasible and other.is_feasible,
            warnings=self.warnings + other.warnings,
            suggestions=self.suggestions + other.suggestions,
            risks=self.risks + other.risks,
        )

    @property
    def summary(self) -> str:
        """Human-readable one-liner."""
        if self.passes:
            return "✅ 请求看起来可行"
        parts = []
        if self.warnings:
            parts.append(f"⚠️ {len(self.warnings)} 条警告")
        if self.risks:
            parts.append(f"🔥 {len(self.risks)} 项风险")
        if self.suggestions:
            parts.append(f"💡 {len(self.suggestions)} 条建议")
        return " | ".join(parts) if parts else "✅ 请求看起来可行"

    @property
    def passes(self) -> bool:
        """Alias — passes check only when feasible and no hard risks."""
        return self.is_feasible and len(self.risks) == 0


# ---------------------------------------------------------------------------
# Known request types & their expected parameter schemas
# ---------------------------------------------------------------------------

_REQUEST_SCHEMAS: dict[str, dict[str, Any]] = {
    "clip": {
        "required": ["materials", "duration"],
        "optional": [
            "resolution", "aspect_ratio", "transition", "bgm",
            "subtitle_style", "output_format",
        ],
    },
    "copywriting": {
        "required": ["topic", "style"],
        "optional": ["tone", "length", "keywords", "hook_type", "target_audience"],
    },
    "composition": {
        "required": ["subject", "style"],
        "optional": [
            "canvas_size", "elements", "color_scheme",
            "text_overlay", "reference_images",
        ],
    },
    "transcript": {
        "required": ["audio_source", "language"],
        "optional": ["speaker_diarization", "timestamp_format"],
    },
    "translate": {
        "required": ["source_text", "target_lang"],
        "optional": ["source_lang", "domain", "glossary"],
    },
    "review": {
        "required": ["content"],
        "optional": ["criteria", "strictness"],
    },
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _count_materials(materials: Any) -> int:
    """Return number of material items from various shapes."""
    if isinstance(materials, list):
        return len(materials)
    if isinstance(materials, dict):
        return len(materials)
    if isinstance(materials, str):
        # comma / newline separated list
        return len([x for x in re.split(r"[,;\n]+", materials) if x.strip()])
    return 0


# Common video resolutions (width, height)
_RESOLUTIONS = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
    "4k": (3840, 2160),
    "vertical_1080p": (1080, 1920),
    "vertical_720p": (720, 1280),
}

# Typical max clip durations (seconds) per resolution
_DURATION_LIMITS = {
    "1080p": 120,
    "720p": 180,
    "480p": 300,
    "4k": 60,
    "vertical_1080p": 120,
    "vertical_720p": 180,
}


def _validate_material_duration(materials: Any, target_duration: float) -> list[str]:
    """Check if available materials can plausibly fill requested duration."""
    warnings: list[str] = []
    count = _count_materials(materials)
    if count == 0:
        warnings.append("素材列表为空，无法生成剪辑内容")
        return warnings

    # Rough heuristic: average clip length per material ~ 4-8s
    max_possible = count * 10
    min_possible = count * 2

    if max_possible < target_duration:
        warnings.append(
            f"素材数 ({count} 个) 可能不足以填充目标时长 "
            f"({target_duration:.0f}s)。预估最大覆盖 ~{max_possible:.0f}s。"
        )
    if count >= 20 and target_duration < 30:
        warnings.append(
            f"素材数量 ({count} 个) 远超目标时长 ({target_duration:.0f}s)，"
            "可能导致过度裁剪或部分素材未被使用。"
        )
    if target_duration <= 0:
        warnings.append("目标时长必须为正数")
    return warnings


def _validate_resolution(requested: str | tuple | None) -> list[str]:
    """Check if resolution string/tuple is recognised."""
    warnings: list[str] = []
    if requested is None:
        return warnings
    if isinstance(requested, str) and requested.lower() not in _RESOLUTIONS:
        warnings.append(f"分辨率 '{requested}' 不在标准列表中 (支持: {', '.join(_RESOLUTIONS)})")
    if isinstance(requested, (list, tuple)) and len(requested) == 2:
        w, h = requested
        if w * h > 3840 * 2160:
            warnings.append(f"分辨率 {w}x{h} 超过 4K 标准，可能影响渲染性能")
    return warnings


def _validate_missing_required(request_type: str, params: dict) -> list[str]:
    """Flag missing required parameters."""
    schema = _REQUEST_SCHEMAS.get(request_type)
    if schema is None:
        return []  # unknown type — skip required-field check
    missing: list[str] = []
    for key in schema.get("required", []):
        if key not in params or params[key] is None or params[key] == "":
            missing.append(key)
    if missing:
        return [f"缺少必填参数: {', '.join(missing)}"]
    return []


def _validate_param_conflicts(request_type: str, params: dict) -> list[str]:
    """Detect contradictory parameter combinations."""
    conflicts: list[str] = []
    res = params.get("resolution")
    ar = params.get("aspect_ratio")

    if res and ar:
        # check if resolution matches aspect ratio
        if isinstance(res, (list, tuple)) and len(res) == 2:
            w, h = res
            if h > 0:
                ratio = w / h
                if isinstance(ar, str):
                    if ":" in ar:
                        ar_w, ar_h = (float(x) for x in ar.split(":", 1))
                        expected = ar_w / ar_h if ar_h else 0
                        if abs(ratio - expected) > 0.05:
                            conflicts.append(
                                f"分辨率 {w}x{h} (比例 {ratio:.2f}) "
                                f"与 aspect_ratio '{ar}' ({expected:.2f}) 不匹配"
                            )

    # Duration vs resolution
    duration = params.get("duration")
    if res and duration:
        res_key = res if isinstance(res, str) else None
        if res_key and res_key.lower() in _DURATION_LIMITS:
            limit = _DURATION_LIMITS[res_key.lower()]
            if isinstance(duration, (int, float)) and duration > limit:
                conflicts.append(
                    f"{res_key} 的推荐最大时长为 {limit}s，"
                    f"当前设定 {duration:.0f}s 可能导致渲染超时或画质下降"
                )

    # style + tone conflicts for copywriting
    if request_type == "copywriting":
        style = params.get("style", "").lower()
        tone = params.get("tone", "").lower()
        informal_styles = {"vlog", "casual", "搞笑", "vlog风格"}
        formal_tones = {"formal", "专业", "official", "正式"}
        if style in informal_styles and tone in formal_tones:
            conflicts.append(
                f"风格 '{style}' 通常搭配轻松语调，与 '{tone}' 可能存在冲突"
            )

    return conflicts


def _assess_risks(request_type: str, params: dict) -> list[str]:
    """Assess potential operational / quality risks."""
    risks: list[str] = []

    # Large clip jobs
    materials = params.get("materials")
    duration = params.get("duration")
    if request_type == "clip":
        count = _count_materials(materials)
        if count > 50:
            risks.append(f"素材数量 ({count} 个) 过多，建议分批处理")
        if isinstance(duration, (int, float)) and duration > 300:
            risks.append(f"目标时长 {duration:.0f}s > 5 分钟，长视频渲染可能耗时较长")

    # Composition with too many elements
    if request_type == "composition":
        elements = params.get("elements", [])
        if isinstance(elements, (list, dict)) and len(elements) > 20:
            risks.append(f"合成元素过多 ({len(elements)} 个)，可能影响加载和渲染性能")

    return risks


# ---------------------------------------------------------------------------
# Main Critic class
# ---------------------------------------------------------------------------

class Critic:
    """Proactive request reviewer.

    Usage
    -----
    critic = Critic()
    result = critic.review("clip", {"materials": [...], "duration": 60})
    if not result.passes:
        for w in result.warnings:
            logger.warning(w)
    """

    def __init__(self) -> None:
        self.enabled = cfg("critic.enabled", True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review(self, request_type: str, params: dict) -> CritiqueResult:
        """Run all applicable checks and return a structured result.

        Parameters
        ----------
        request_type : str
            One of the known types (clip, copywriting, composition, …)
        params : dict
            Parameters of the request.

        Returns
        -------
        CritiqueResult
        """
        if not self.enabled:
            return CritiqueResult(is_feasible=True)

        if not isinstance(params, dict):
            return CritiqueResult(
                is_feasible=False,
                warnings=["params 必须为 dict 类型"],
            )

        result = CritiqueResult()
        result = result.merge(self._check_required(request_type, params))
        result = result.merge(self._check_materials(request_type, params))
        result = result.merge(self._check_resolution(request_type, params))
        result = result.merge(self._check_conflicts(request_type, params))
        result = result.merge(self._assess_operational_risks(request_type, params))
        result = result.merge(self._suggest_improvements(request_type, params))

        return result

    # ------------------------------------------------------------------
    # Internal check groups
    # ------------------------------------------------------------------

    def _check_required(self, request_type: str, params: dict) -> CritiqueResult:
        warnings = _validate_missing_required(request_type, params)
        return CritiqueResult(
            is_feasible=len(warnings) == 0,
            warnings=warnings,
        )

    def _check_materials(self, request_type: str, params: dict) -> CritiqueResult:
        warnings: list[str] = []
        suggestions: list[str] = []

        duration = params.get("duration")
        materials = params.get("materials")

        if request_type == "clip":
            if materials is not None and duration is not None:
                material_warnings = _validate_material_duration(materials, duration)
                warnings.extend(material_warnings)
            if materials is None:
                warnings.append("缺失 'materials' 参数，无法执行剪辑")
            elif isinstance(materials, (list, dict)) and len(materials) == 0:
                warnings.append("素材列表为空")
            elif isinstance(materials, str) and not materials.strip():
                warnings.append("素材路径为空")

            if duration is None:
                warnings.append("缺失 'duration' 参数，无法确定剪辑长度")
            elif not isinstance(duration, (int, float)) or duration <= 0:
                warnings.append("'duration' 必须为正数")

        return CritiqueResult(
            is_feasible=len(warnings) == 0,
            warnings=warnings,
            suggestions=suggestions,
        )

    def _check_resolution(self, request_type: str, params: dict) -> CritiqueResult:
        warnings: list[str] = []
        if request_type in ("clip", "composition"):
            res = params.get("resolution")
            if res is not None:
                warnings.extend(_validate_resolution(res))
        return CritiqueResult(warnings=warnings)

    def _check_conflicts(self, request_type: str, params: dict) -> CritiqueResult:
        conflicts = _validate_param_conflicts(request_type, params)
        return CritiqueResult(
            is_feasible=len(conflicts) == 0,
            warnings=conflicts,
        )

    def _assess_operational_risks(self, request_type: str, params: dict) -> CritiqueResult:
        risks = _assess_risks(request_type, params)
        return CritiqueResult(risks=risks)

    def _suggest_improvements(self, request_type: str, params: dict) -> CritiqueResult:
        """Generate helpful suggestions based on params."""
        suggestions: list[str] = []

        if request_type == "clip":
            duration = params.get("duration")
            materials = params.get("materials")
            if materials and duration:
                count = _count_materials(materials)
                if isinstance(duration, (int, float)) and count > 0:
                    sec_per_clip = duration / count
                    if sec_per_clip < 2:
                        suggestions.append(
                            f"每个素材平均只分配 {sec_per_clip:.1f}s，"
                            "建议减少素材数或增加总时长以避免剪辑过于仓促"
                        )
                    elif sec_per_clip > 15:
                        suggestions.append(
                            f"每个素材平均 {sec_per_clip:.1f}s，"
                            "较长片段可能需要添加转场或 B-roll 保持节奏"
                        )

            if params.get("transition") is None and duration and duration > 30:
                suggestions.append("视频超过 30 秒，建议添加转场效果使内容更流畅")

        if request_type == "copywriting":
            if params.get("keywords") and not params.get("hook_type"):
                suggestions.append("提供了关键词但未指定吸引开头类型（hook_type），"
                                   "建议设置以提升完播率")
            if params.get("target_audience") is None:
                suggestions.append("未指定目标受众（target_audience），文案可能缺乏针对性")

        if request_type == "composition":
            if params.get("canvas_size") and params.get("reference_images") is None:
                suggestions.append("提供了画布尺寸但无参考图片，合成效果可能不够精准")

        return CritiqueResult(suggestions=suggestions)
