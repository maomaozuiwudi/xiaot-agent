"""
技能管理器 — SkillManager
============================
管理技能的安装、卸载、加载、注册与触发匹配。
类似于 Hermes Agent 的技能系统，但专为内容工坊定制。
"""

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import yaml

from config_loader import resolve_path
from brain.tools import get_registry, ToolRegistry

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_SKILL = "xhs-content-factory"
SKILL_YAML = "skill.yaml"
SKILL_DIR = None  # 由 SkillManager 构造时确定


# ──────────────────────────────────────────────────────────────────────────
# SkillManager
# ──────────────────────────────────────────────────────────────────────────

class SkillManager:
    """
    技能管理器。

    负责:
    - 扫描 skills/ 目录并加载所有 skill.yaml
    - 向 ToolRegistry 注册/注销工具
    - 安装 & 卸载技能
    - 基于用户输入的关键字匹配技能

    用法::

        mgr = SkillManager()
        mgr.load_all()
        skills = mgr.list_skills()
        skill = mgr.match_skill("帮我做个穿搭视频")
        if skill:
            tools = skill["tools"]
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        skills_dir: Optional[str] = None,
    ):
        self._registry = registry or get_registry()

        # 决定技能存放目录
        if skills_dir is not None:
            self._skills_dir = Path(skills_dir)
        else:
            # 默认: skills/ 相对于 resolve_path 后的 project root
            base = Path(resolve_path("."))
            self._skills_dir = base / "skills"

        self._skills_dir.mkdir(parents=True, exist_ok=True)

        # _skills: dict[name -> dict]  技能元数据缓存
        self._skills: dict[str, dict] = {}

        # _tool_to_skill: dict[tool_name -> skill_name]  工具归属追踪
        self._tool_to_skill: dict[str, str] = {}

        # _default_bundled: bool  标记默认技能是否已加载
        self._default_loaded = False

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def load_all(self) -> int:
        """
        扫描 skills/ 目录下所有子目录，加载其中的 skill.yaml。

        Returns:
            成功加载的技能数量。
        """
        count = 0
        errors = []

        for entry in sorted(self._skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            skill_yaml_path = entry / SKILL_YAML
            if not skill_yaml_path.exists():
                continue

            try:
                self._load_single(skill_yaml_path)
                count += 1
                logger.info("技能已加载: %s", entry.name)
            except Exception as e:
                logger.error("加载技能失败 %s: %s", entry.name, e)
                errors.append((entry.name, str(e)))

        # 如果没有任何技能被加载，自动加载默认的 bundled 技能
        if count == 0:
            logger.info("未发现外部技能，加载默认技能: %s", DEFAULT_SKILL)
            self._ensure_default()
            count = 1

        if errors:
            logger.warning("加载完成，%d 个成功，%d 个失败", count, len(errors))

        return count

    def install(self, path_or_url: str) -> dict:
        """
        安装技能。

        支持:
        - 本地目录路径 (…/my-skill/)
        - 本地 ZIP 文件路径 (…/my-skill.zip)
        - URL (尚未实现，预留未来扩展)

        Args:
            path_or_url: 技能目录路径、ZIP 路径或 URL。

        Returns:
            技能元数据字典。

        Raises:
            FileNotFoundError: 路径不存在。
            ValueError: skill.yaml 缺失或格式错误。
            NotImplementedError: URL 安装暂未实现。
        """
        source = Path(path_or_url)

        # ── URL 安装（预留） ──────────────────────────────────────────
        if str(path_or_url).startswith(("http://", "https://", "git://")):
            raise NotImplementedError(
                "远程 URL 安装暂未实现。请先 clone/下载技能到本地，然后使用本地路径安装。"
            )

        # ── ZIP 安装 ──────────────────────────────────────────────────
        if source.suffix.lower() == ".zip":
            return self._install_from_zip(source)

        # ── 目录安装 ──────────────────────────────────────────────────
        if source.is_dir():
            return self._install_from_dir(source)

        raise FileNotFoundError(f"技能源不存在: {path_or_url}")

    def uninstall(self, name: str) -> bool:
        """
        卸载指定名称的技能。

        会自动注销该技能注册的所有工具，并删除技能目录。

        Args:
            name: 技能名称（与 skill.yaml 中的 name 字段一致）。

        Returns:
            True 表示成功删除，False 表示未找到。
        """
        if name not in self._skills:
            logger.warning("尝试卸载不存在的技能: %s", name)
            return False

        # 禁止卸载默认内置技能
        if name == DEFAULT_SKILL:
            logger.warning("禁止卸载默认技能: %s", DEFAULT_SKILL)
            return False

        skill_info = self._skills[name]
        skill_dir = self._get_skill_dir(name)

        # 注销该技能注册的所有工具
        tools = skill_info.get("tools", {})
        for tool_name in tools:
            self._registry.unregister(tool_name)
            self._tool_to_skill.pop(tool_name, None)
            logger.debug("工具已注销: %s (来自技能: %s)", tool_name, name)

        # 删除技能目录
        if skill_dir.exists():
            shutil.rmtree(skill_dir)

        # 从缓存中移除
        del self._skills[name]
        logger.info("技能已卸载: %s", name)
        return True

    def list_skills(self) -> list[dict]:
        """
        返回所有已安装技能的元数据列表。

        Returns:
            [{"name": ..., "version": ..., "description": ..., ...}, ...]
        """
        return list(self._skills.values())

    def get_skill(self, name: str) -> Optional[dict]:
        """
        通过名称获取技能元数据。

        Args:
            name: 技能名称。

        Returns:
            技能元数字典，未找到时返回 None。
        """
        return self._skills.get(name)

    def get_active_skill(self) -> Optional[dict]:
        """
        返回当前"活跃"的技能。

        策略:
        1. 如果有且仅有一个技能，返回它。
        2. 如果用户通过 match_skill() 匹配过某个技能（最近一次匹配），
           返回该技能。（简单实现：返回第一个已加载的技能。）
        3. 否则返回默认技能。

        Returns:
            技能元数据字典，或 None（没有任何技能）。
        """
        if not self._skills:
            return None
        if len(self._skills) == 1:
            return next(iter(self._skills.values()))
        # 取第一个已加载的技能
        first = next(iter(self._skills.values()))
        return first

    def match_skill(self, user_input: str) -> Optional[dict]:
        """
        根据用户输入匹配最合适的技能。

        匹配规则:
        1. 遍历所有技能，检查其 triggers 列表中的关键词。
        2. 返回第一个匹配到的技能。
        3. 如果都不匹配，返回 None。

        Args:
            user_input: 用户输入的自然语言文本。

        Returns:
            匹配到的技能元数据，或 None。
        """
        if not user_input or not self._skills:
            return None

        user_input_lower = user_input.lower()

        for skill_name, skill_info in self._skills.items():
            triggers = skill_info.get("triggers", [])
            for trigger in triggers:
                if trigger and trigger.lower() in user_input_lower:
                    logger.debug(
                        "技能匹配成功: %s (触发词: %s)", skill_name, trigger
                    )
                    return skill_info

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Internal: 单技能加载
    # ──────────────────────────────────────────────────────────────────────

    def _load_single(self, yaml_path: Path) -> dict:
        """
        加载单个 skill.yaml 文件，注册其工具。

        Args:
            yaml_path: skill.yaml 的完整路径。

        Returns:
            技能元数据字典。
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            skill_def = yaml.safe_load(f)

        if not skill_def or "name" not in skill_def:
            raise ValueError(f"无效的 skill.yaml: {yaml_path} — 缺少 'name' 字段")

        name = skill_def["name"]

        # 注册工具
        tools = skill_def.get("tools", {})
        if isinstance(tools, dict):
            for tool_name, tool_def in tools.items():
                self._register_tool_from_skill(name, tool_name, tool_def)

        # 缓存技能元数据
        self._skills[name] = {
            "name": name,
            "version": skill_def.get("version", "0.0.0"),
            "description": skill_def.get("description", ""),
            "author": skill_def.get("author", ""),
            "triggers": skill_def.get("triggers", []),
            "tools": list(tools.keys()) if isinstance(tools, dict) else [],
            "references": skill_def.get("references", []),
            "dependencies": skill_def.get("dependencies", []),
            "install": skill_def.get("install", {}),
            "yaml_path": str(yaml_path),
        }

        logger.info(
            "技能已注册: %s v%s (%d 个工具)",
            name,
            self._skills[name]["version"],
            len(self._skills[name]["tools"]),
        )

        return self._skills[name]

    def _register_tool_from_skill(
        self, skill_name: str, tool_name: str, tool_def: dict
    ):
        """
        将 skill.yaml 中定义的工具有桩注册到 ToolRegistry。

        Args:
            skill_name: 技能名称。
            tool_name: 工具名称。
            tool_def: 工具定义字典（含 description, params 等）。
        """
        description = tool_def.get("description", "")
        params = tool_def.get("params", {})

        # 转换为 OpenAI 兼容的 parameters schema
        parameters_schema = self._tool_params_to_schema(tool_name, params)

        # 创建一个桩 handler，实际逻辑由技能的具体实现覆盖
        def _stub_handler(args: dict) -> tuple[str, dict]:
            logger.info("[Skill:%s] Tool:%s 被调用 args=%s", skill_name, tool_name, args)
            return (
                f"[{skill_name}] {tool_name} 已执行（桩模式）",
                {"status": "ok", "tool": tool_name, "skill": skill_name, "args": args},
            )

        # 注册到 ToolRegistry
        self._registry.register(
            name=tool_name,
            description=description,
            handler=_stub_handler,
            parameters_schema=parameters_schema,
            requires_confirm=False,
        )

        # 记录工具归属
        self._tool_to_skill[tool_name] = skill_name
        logger.debug("工具已注册: %s ← 技能: %s", tool_name, skill_name)

    @staticmethod
    def _tool_params_to_schema(tool_name: str, params: dict) -> dict:
        """
        将 skill.yaml 中简洁的 params 定义转换为 OpenAI Function Calling 的 JSON Schema。

        Args:
            tool_name: 工具名称（仅用于日志）。
            params: skill.yaml 中定义的参数字典，
                    如 {"paths": {"type": "list"}, "duration": {"type": "float"}}。

        Returns:
            OpenAI 兼容的 parameters schema。
        """
        properties = {}
        required = []

        for param_name, param_info in params.items():
            if isinstance(param_info, str):
                # 简写: "duration: float" → {"type": "string", "description": "float"}
                # 但如果值是 "list"、"dict"、"float"、"int"、"bool"，推断为类型
                type_mapping = {
                    "str": "string",
                    "string": "string",
                    "int": "integer",
                    "integer": "integer",
                    "float": "number",
                    "number": "number",
                    "bool": "boolean",
                    "boolean": "boolean",
                    "list": "array",
                    "array": "array",
                    "dict": "object",
                    "object": "object",
                }
                inferred_type = type_mapping.get(param_info.lower(), "string")
                properties[param_name] = {"type": inferred_type}
                required.append(param_name)
            elif isinstance(param_info, dict):
                # 完整定义
                ptype = param_info.get("type", "string")
                type_mapping = {
                    "str": "string",
                    "string": "string",
                    "int": "integer",
                    "integer": "integer",
                    "float": "number",
                    "number": "number",
                    "bool": "boolean",
                    "boolean": "boolean",
                    "list": "array",
                    "array": "array",
                    "dict": "object",
                    "object": "object",
                }
                schema_type = type_mapping.get(str(ptype).lower(), "string")
                prop = {
                    "type": schema_type,
                    "description": param_info.get("description", ""),
                }
                if "default" in param_info:
                    prop["default"] = param_info["default"]
                if "items" in param_info:
                    prop["items"] = param_info["items"]
                properties[param_name] = prop
                if param_info.get("required", True):
                    required.append(param_name)
            else:
                properties[param_name] = {"type": "string"}
                required.append(param_name)

        schema = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        return schema

    # ──────────────────────────────────────────────────────────────────────
    # Internal: 安装逻辑
    # ──────────────────────────────────────────────────────────────────────

    def _install_from_dir(self, source_dir: Path) -> dict:
        """
        从本地目录安装技能。

        Args:
            source_dir: 技能目录路径，必须包含 skill.yaml。

        Returns:
            技能元数据字典。
        """
        skill_yaml = source_dir / SKILL_YAML
        if not skill_yaml.exists():
            raise ValueError(f"目录中未找到 {SKILL_YAML}: {source_dir}")

        # 读取 yaml 获取技能名称
        with open(skill_yaml, "r", encoding="utf-8") as f:
            skill_def = yaml.safe_load(f)
        if not skill_def or "name" not in skill_def:
            raise ValueError(f"无效的 {SKILL_YAML}: 缺少 'name' 字段")

        name = skill_def["name"]

        # 如果已存在同名技能，先卸载
        if name in self._skills:
            logger.warning("技能 '%s' 已存在，将被覆盖安装", name)
            self.uninstall(name)

        # 复制到 skills 目录
        target_dir = self._skills_dir / name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)

        # 从新位置加载
        return self._load_single(target_dir / SKILL_YAML)

    def _install_from_zip(self, zip_path: Path) -> dict:
        """
        从 ZIP 文件安装技能。

        Args:
            zip_path: ZIP 文件路径。

        Returns:
            技能元数据字典。
        """
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP 文件不存在: {zip_path}")

        # 解压到临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)

            extracted = Path(tmpdir)
            # 查找 skill.yaml
            skill_yaml = self._find_skill_yaml(extracted)
            if skill_yaml is None:
                raise ValueError(
                    f"ZIP 文件中未找到 {SKILL_YAML}。"
                    f"请确保 ZIP 根目录或第一级子目录包含 {SKILL_YAML}。"
                )

            # 使用包含 skill.yaml 的目录进行安装
            return self._install_from_dir(skill_yaml.parent)

    @staticmethod
    def _find_skill_yaml(base_dir: Path) -> Optional[Path]:
        """
        在解压后的目录树中查找 skill.yaml。

        搜索策略:
        1. 检查基目录。
        2. 检查第一级子目录（处理常见的单目录包裹情况）。

        Args:
            base_dir: 解压后的基目录。

        Returns:
            skill.yaml 的路径，未找到时返回 None。
        """
        # 直接检查基目录
        candidate = base_dir / SKILL_YAML
        if candidate.exists():
            return candidate

        # 检查第一级子目录
        for child in base_dir.iterdir():
            if child.is_dir():
                candidate = child / SKILL_YAML
                if candidate.exists():
                    return candidate

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Internal: 默认技能
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_default(self):
        """确保默认的 xhs-content-factory 技能始终可用。"""
        if self._default_loaded:
            return

        # 检查默认技能目录是否有 skill.yaml
        default_skill_dir = self._skills_dir / DEFAULT_SKILL
        default_yaml = default_skill_dir / SKILL_YAML

        if default_yaml.exists():
            self._load_single(default_yaml)
            self._default_loaded = True
            return

        # 如果没有 skill.yaml，创建一个内置的最小定义
        self._register_default_builtin()
        self._default_loaded = True

    def _register_default_builtin(self):
        """默认技能的内置注册（无需实际 skill.yaml 文件）。"""
        default_def = {
            "name": DEFAULT_SKILL,
            "version": "2.0.0",
            "description": "小红书穿搭视频自动剪辑合成",
            "author": "工具猫",
            "triggers": ["穿搭", "出片", "视频剪辑", "小红书"],
            "tools": [
                "clip_videos",
                "vision_analyze",
                "generate_copy",
                "compose_video",
                "generate_card",
                "search_web",
                "synthesize_tts",
            ],
            "references": ["references/"],
            "dependencies": [],
            "install": {"pip": ["mediapipe", "opencv-python", "moviepy", "edge-tts"]},
        }

        self._skills[DEFAULT_SKILL] = default_def
        logger.info("默认技能已注册（内置）: %s v%s", DEFAULT_SKILL, default_def["version"])

    # ──────────────────────────────────────────────────────────────────────
    # Internal: 辅助方法
    # ──────────────────────────────────────────────────────────────────────

    def _get_skill_dir(self, name: str) -> Path:
        """获取技能目录路径。"""
        return self._skills_dir / name

    @property
    def skills_dir(self) -> Path:
        """技能文件存放根目录。"""
        return self._skills_dir

    @property
    def registry(self) -> ToolRegistry:
        """关联的 ToolRegistry 实例。"""
        return self._registry

    def __repr__(self) -> str:
        return (
            f"<SkillManager skills_dir={self._skills_dir} "
            f"loaded={len(self._skills)}>"
        )
