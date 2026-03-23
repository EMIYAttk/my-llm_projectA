---
name: weather
description: 在规划行程等情景下，需要查询天气信息时使用该技能，将查询到的天气情况添加到上下文中。
compatibility: skills.weather.tools
---

# Weather  Skill

**触发条件**: 规划行程时先查天气

**步骤**:
1. 用 `get_weather("北京")` 获取天气
2. 保存到 `/plan/weather.txt`
3. 输出："天气已记录，下一步查景点"

**依赖**: 无
**输出给**: search skill