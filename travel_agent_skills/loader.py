# src/travel_agent_skills/loader.py
from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Optional, Tuple

try:
    import yaml  # 用于解析 front matter
except Exception:
    yaml = None  # 保守兜底，若没安装则降级处理

class SkillInfo:
    def __init__(self, name: str, description: str, content: str):
        self.name = name
        self.description = description
        self.content = content

class SkillLoader:
    """
    读取 travel_agent_skills/skills 下的技能描述与内容。
    SKILL.md 的格式示例（简化版本）：
    ---
    name: math
    description: 使用这个技能来进行各种数学运算，在需要计算预算等情景下使用。
    ---
    
    # Calculator Skill
    ...
    """
    def __init__(self, root: Optional[Path] = None):
        # 默认为 package 里的技能目录
        self.root = Path(root) if root else Path(__file__).resolve().parents[1] / "skills"

    def _parse_front_matter(self, text: str) -> Tuple[dict, str]:
        data = {}
        content = text
        if text.startswith("---"):
            # 找到第二个 --- 的结束位置
            end = text.find("\n---", 3)
            if end != -1:
                front = text[3:end]
                content = text[end + 4 :].lstrip()
                if yaml:
                    try:
                        data = yaml.safe_load(front) or {}
                    except Exception:
                        data = {}
        return data or {}, content

    def load_all(self) -> Dict[str, SkillInfo]:
        skills: Dict[str, SkillInfo] = {}

        if not self.root.exists():
            return skills

        for item in self.root.iterdir():
            if not item.is_dir():
                continue
            skill_md = item / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                text = skill_md.read_text(encoding="utf-8")
                front, content = self._parse_front_matter(text)
                name = front.get("name") or item.name
                description = front.get("description") or ""
                skill = SkillInfo(name=name, description=description, content=content.strip())
                skills[name] = skill
            except Exception:
                # 跳过格式异常的技能
                continue

        return skills
