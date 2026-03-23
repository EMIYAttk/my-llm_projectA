---
name: write
description: 使用这个技能来将之前处理好的结果数据汇总为最后的行程计划。
compatibility: skills.write.tools
---


# Itinerary Writer Skill

**触发条件**: 所有数据齐全

**步骤**:
1. 读 `/plan/weather/attractions/budget.txt` 文件
2. 将参数传入工具`generate_itinerary`
3. 输出最终行程

**组合所有**: 最终合成完整计划