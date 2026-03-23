
import yaml
from pathlib import Path
from typing import Dict, TypedDict,NotRequired,List
from langchain_core.tools import BaseTool
#读取skills/目录，覆盖获取需要的技能给SKILLS

class Skill(TypedDict):
    """可以逐步披露给智能体的技能"""
    name: str  # 技能的唯一标识符
    description: str  # 显示在系统提示词中的简短描述
    tools : NotRequired[List[BaseTool]]  # 说明这个技能拥有哪些工具函数，可为空
    content: str  # 包含详细指令的完整技能内容[4](@ref)

def build_skill_index(skills_root: str = "./skills") -> list[Skill]:
    """
    扫描 skills 下的 SKILL.md，将内容按格式加入SKILLS: list[Skill]中，供create_agent.py导入:
    待完善
    """
    # index = {}
    # root = Path(skills_root)
    # for md in root.glob("*/SKILL.md"):
    #     text = md.read_text(encoding="utf-8")
    #     # 解析第一个 front-matter --- ... ---
    #     parts = text.split("---")
    #     if len(parts) >= 3:
    #         fm_text = parts[1]
    #         try:
    #             meta = yaml.safe_load(fm_text) or {}
    #         except Exception:
    #             meta = {}
    #     else:
    #         meta = {}

    #     name = meta.get("name") or md.parent.name
    #     tools_module = meta.get("compatibility") 
    #     index[name] = {
    #         "description": meta.get("description", ""),
    #         "tools_module": tools_module,
    #         "path": str(md.parent),
    #     }
    # return index
    
    
#SKILLS: list[Skill] = build_skill_index()  下面那个赋值作为占位，实际靠这个完善后的函数获得
    
SKILLS: List[Skill] = [
        
        {
        "name": "gaode_navigation",
        "description": "高德地图服务：查询天气、地图信息、查询各种地址和规划行程路线",
        "content": """# 高德地图导航技能

## 功能范围
- 实时天气查询：获取指定位置的天气状况
- 地图信息查询：搜索地点、POI信息
- 路线规划：驾车、步行、公交路线规划
- 地理编码：地址与坐标互相转换
- 各种地址查询：包括酒店、加油站、餐馆和学校等各种地图上可查的地址标记

## 使用规范
1. 当用户询问天气、位置、路线时使用此技能
2. 需要明确具体的地理位置信息
3. 路线规划需提供起点和终点

## 部分工具说明
- weather_query: 查询天气信息
- map_search: 地图搜索
- route_planning: 路线规划
- geocoding: 地理编码转换"""
    },
        
        
        
        
        
    ]
    
    