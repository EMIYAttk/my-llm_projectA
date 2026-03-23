---
name: math
description: 使用这个技能来进行各种数学运算，在需要计算预算等情景下使用。

compatibility: skills.math.tools
---

#  Calculator Skill

**触发条件**: 景点搜索后需要计算预算

**步骤**:
1. 读 `/plan/attractions.txt`
2. `calculate_budget(景点列表)`
3. 保存 `/plan/budget.txt`

**依赖**: search skill