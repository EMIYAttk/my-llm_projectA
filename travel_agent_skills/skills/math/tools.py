from typing import List
from langchain_core.tools import tool

@tool
def calculate_budget(activities: List[str], base_cost: float = 500.0) -> str:
    """
    根据景点/活动列表计算旅行预算。交通+门票+餐饮估算。
    
    Args:
        activities: 活动列表，从 search 结果传入
        base_cost: 基础预算（默认500元，包含交通住宿）
    
    Returns:
        详细预算明细和总额。
    
    Examples:
        >>> calculate_budget(["故宫", "颐和园", "798艺术区"])
    """
    activity_costs = {"景点": 80, "艺术区": 50, "商业街": 30}
    extra = sum(activity_costs.get(act.split()[0], 50) for act in activities)
    total = base_cost + extra
    return f"""
预算明细:
- 基础: {base_cost}元 (交通住宿)
- 活动门票/餐饮: {extra}元
- 总预算: {total}元 (预留10%弹性)
建议: {total < 800 and '经济型行程' or '中等消费'}"""


TOOLS = [calculate_budget]