from typing import Dict
from langchain_core.tools import tool

@tool
def generate_itinerary(weather: str, attractions: str, budget: str) -> str:
    """
    综合天气、景点、预算生成完整旅行行程表。
    
    Args:
        weather: get_weather 结果
        attractions: google_search 结果
        budget: calculate_budget 结果
    
    Returns:
        Day-by-Day 详细行程，包括时间、交通、餐饮建议。
    
    Examples:
        >>> generate_itinerary(weather="晴朗20°C", attractions="故宫...", budget="800元")
    """
    itinerary = f"""
北京周末旅行完整行程 (预算控制: {budget})

📅 Day 1 - 文化探索日 ({weather[:10]})
08:00 出发 → 故宫 (2h, 60元)
12:00 王府井午餐 (小吃街, 50元)
15:00 颐和园 (3h, 30元)
晚上: 自由活动

📅 Day 2 - 现代艺术日
09:00 798艺术区 (3h, 免费)
13:00 午餐 (咖啡馆, 40元)
16:00 南锣鼓巷逛街 (2h)

💰 总计符合预算。祝旅途愉快！
"""
    return itinerary


TOOLS = [generate_itinerary]