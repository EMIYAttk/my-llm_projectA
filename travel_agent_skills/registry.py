# src/travel_agent_skills/registry.py
from __future__ import annotations

from typing import Dict

# 全局技能内容注册表：{ skill_name: {"description": ..., "content": ...} }
_SKILL_REGISTRY: Dict[str, Dict[str, str]] = {}

def set_skill_registry(registry: Dict[str, Dict[str, str]]):
    global _SKILL_REGISTRY
    _SKILL_REGISTRY = registry

def get_skill_content(skill_name: str) -> str:
    entry = _SKILL_REGISTRY.get(skill_name)
    if entry:
        return entry.get("content", "")
    return ""

def get_skill_description(skill_name: str) -> str:
    entry = _SKILL_REGISTRY.get(skill_name)
    if entry:
        return entry.get("description", "")
    return ""