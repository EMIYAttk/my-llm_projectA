---
name: search
description: 使用这个技能来查询相关景点，并计算景点预算。
compatibility: skills.search.tools
---

#  Search Skill

**触发条件**: 天气查完后

**步骤**:
1. 读取  `/plan/weather.txt` 文件

2. `google_search("北京 周末景点")`

3. 保存 `/plan/attractions.txt`


**输入**: `/plan/weather.txt`
**依赖**: weather skill
