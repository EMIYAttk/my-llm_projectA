import requests  # 模拟
from langchain_core.tools import tool

@tool
def google_search(query: str) -> str:
    """
    搜索景点、活动、美食等旅行相关信息。基于天气结果推荐匹配活动。
    
    Args:
        query: 搜索关键词，如"北京 周末景点"、"上海 美食推荐"
    
    Returns:
        搜索结果列表，包括名称、地址、评分、简介。
    
    Examples:
        >>> google_search("北京 周末景点")
        >>> google_search("上海 室内活动")
    """
    # 模拟 Google 搜索结果
    results = {
        "北京 周末景点": """
1. 故宫 (评分4.8): 皇家宫殿，必游
2. 颐和园 (评分4.7): 皇家园林，湖光山色
3. 798艺术区 (评分4.6): 现代艺术，拍照圣地
4. 王府井 (评分4.5): 商业街，小吃购物
        """,
        "上海 周末活动": """
1. 外滩 (评分4.8): 夜景观光
2. 东方明珠 (评分4.7): 登高远眺
3. 南京路 (评分4.6): 购物天堂
        """
    }
    return results.get(query, f"搜索 '{query}'：热门推荐列表")


TOOLS = [google_search]