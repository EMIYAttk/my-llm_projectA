from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的周末天气预报。用于旅行规划的第一步，判断户外/室内活动适宜性。
    
    Args:
        city: 城市名称，如"北京"、"上海"
    
    Returns:
        天气描述，包括温度、降雨概率、风力等关键信息。
    
    Examples:
        >>> get_weather("北京")
        >>> get_weather("上海")
    """
    # 模拟真实 API
    weather_data = {
        "北京": "周末晴转多云，20-25°C，微风，无雨，适合户外游览",
        "上海": "周末多云，22-28°C，轻雨可能，建议备伞"
    }
    return weather_data.get(city, f"{city} 周末天气：晴朗，22°C，适宜出行")

TOOLS = [get_weather]