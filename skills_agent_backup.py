from langchain.agents import create_agent, AgentState
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from llm.log_utils import log
# from llm.my_llm import llm
from mcp_tool_config import gaode_mcp_server_config, my12306_mcp_server_config, analysis_mcp_server_config
from skills_list import SKILLS
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from typing import Callable, List, Dict
OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_API_KEY="sk-0c13acfa4a454c73a23a213ffd4d08e2"
from typing_extensions import Annotated
from langgraph.types import Command
import asyncio
import logging
from langchain.tools import tool, ToolRuntime   # ToolRuntime 隐藏于模型，不会出现在工具 schema

#    导入工具；    现在还是得手动导入，自动导入还没有尝试成功

# 在skills_list.py里【即SKILLS列表】添加技能说明（可脚本化添加)；
   
# 在local_tools添加到对应的分类里（可简短逻辑直接添加)； 即简单的加入{skills.name : skills.tools  }

# 在skill_tool_mapping里添加技能对分类的映射(可自动无脑逻辑：键值对相同值 添加 如："search": all_tools.get("search", [])

# 最后别忘了system_prompt

#   自定义工具导入区：




#----------------------自定义的tools函数，可以从别的地方导入，对性能影响不大------------------------------------------------------
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


# 2) 新增日志中间件：记录每次工具调用及返回
class LoggingMiddleware(AgentMiddleware):
    async def awrap_tool_call(self, request, handler):
        # request 中通常包含 tool, kwargs, tool_call_id 等
        tool_name = getattr(request.tool, "name", repr(request.tool))
        args = getattr(request, "kwargs", None) or {}
        log.info(f"工具调用 → name: {tool_name}, args: {args}")
        try:
            result = await handler(request)
            log.info(f"工具返回 ← name: {tool_name}, result: {result}")
            return result
        except Exception as e:
            log.error(f"工具执行错误: {tool_name}: {e}")
            raise
#--------------------------------------------------------------------------------------------

# 定义技能状态Schema
class SkillState(AgentState):
    """技能状态管理"""
    skills_loaded: Annotated[List[str], lambda current, new: current + [s for s in new if s not in current]] = []


# 技能映射到具体工具的函数
def get_tools_by_skill(skill_name: str, all_tools: Dict[str, List[BaseTool]]) -> List[BaseTool]:
    """根据技能名称获取对应的工具列表"""
    skill_tool_mapping = {
        "gaode_navigation": all_tools.get("gaode", []),
        "railway_booking": all_tools.get("12306", []),
        "data_analysis": all_tools.get("fenxi", []),
        "weather": all_tools.get("weather", []),
        "search": all_tools.get("search", []),
        "math": all_tools.get("math", []),
    }
    return skill_tool_mapping.get(skill_name, [])


class SkillMiddleware(AgentMiddleware):
    """优化后的技能中间件：减少冗余日志输出"""

    def __init__(self, all_tools: Dict[str, List[BaseTool]]):
        super().__init__()
        self.all_tools = all_tools
        self.base_tools = [load_skill]
        self._pre_register_all_tools()

        # 优化缓存机制
        self.last_skills_loaded = set()  # 使用集合提高查找效率
        self.logged_skill_tools = set()  # 新增：记录已日志过的技能工具组合

    def _pre_register_all_tools(self):
        """预先注册所有工具到中间件"""
        all_tool_instances = []
        for tool_category in self.all_tools.values():
            all_tool_instances.extend(tool_category)
        self.tools = all_tool_instances + self.base_tools

    async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """异步方法： 异步中间中间件"""

        try:
            # 获取当前已加载的技能
            current_skills = set(request.state.get('skills_loaded', []))

            # 优化：使用集合运算检测技能变化
            newly_loaded_skills = current_skills - self.last_skills_loaded
            removed_skills = self.last_skills_loaded - current_skills

            # 只在技能状态实际变化时输出日志
            if newly_loaded_skills or removed_skills:
                log.info(f"技能状态变化: 加载→{newly_loaded_skills}, 移除→{removed_skills}")
                self.last_skills_loaded = current_skills.copy()

                # 技能变化时重置工具日志记录
                self.logged_skill_tools.clear()

            # 构建动态工具列表
            dynamic_tools = self.base_tools.copy()

            # 关键修复：确保每个技能的工具加载日志只输出一次
            for skill in current_skills:
                skill_tools = get_tools_by_skill(skill, self.all_tools)
                if skill_tools:
                    # 为当前技能生成唯一标识
                    skill_tools_key = f"{skill}_{len(skill_tools)}"

                    # 只有未记录过的技能工具组合才输出日志
                    if skill_tools_key not in self.logged_skill_tools:
                        log.info(f"加载技能 '{skill}' 的工具，数量: {len(skill_tools)}")
                        self.logged_skill_tools.add(skill_tools_key)

                    dynamic_tools.extend(skill_tools)

            # 调试信息：只在调试模式下输出详细工具列表
            tool_names = [tool.name for tool in dynamic_tools if hasattr(tool, 'name')]
            log.debug(f"当前可用工具: {tool_names}")

            # 构建简化的技能提示信息
            skills_prompt = self._build_skills_prompt(current_skills)

            # 更新系统消息（避免重复累积）
            new_system_message = self._update_system_message(request, skills_prompt)

            # 创建修改后的请求
            modified_request = request.override(
                tools=dynamic_tools,
                system_message=new_system_message
            )

            # 异步调用处理器
            response = await handler(modified_request)

            return response

        except Exception as e:
            log.error(f"技能中间件执行错误: {e}")
            # 出错时回退到基础工具
            fallback_request = request.override(tools=self.base_tools)
            return await handler(fallback_request)

    def _build_skills_prompt(self, current_skills: set) -> str:
        """构建技能提示信息"""
        if not current_skills:
            return "\n## 技能状态\n当前未加载技能，请使用 load_skill 工具加载所需技能。"

        skill_list = ", ".join(sorted(current_skills))  # 排序确保输出一致性
        return f"\n## 技能状态\n已加载技能: {skill_list}"

    def _update_system_message(self, request: ModelRequest, skills_prompt: str) -> SystemMessage:
        """更新系统消息，避免重复累积"""
        current_system_message = getattr(request, 'system_message', None)
        if current_system_message and hasattr(current_system_message, 'content'):
            current_content = str(current_system_message.content)

            # 更精确地移除旧技能提示
            lines = current_content.split('\n')
            clean_lines = []
            skip_next = False
            for line in lines:
                if line.startswith('## 技能状态'):
                    skip_next = True  # 跳过技能状态行
                    continue
                if skip_next and line.strip() == "":
                    skip_next = False  # 遇到空行后停止跳过
                    continue
                if not skip_next:
                    clean_lines.append(line)

            clean_content = '\n'.join(clean_lines).strip()
            new_content = clean_content + skills_prompt
        else:
            new_content = skills_prompt

        return SystemMessage(content=new_content)


# 优化后的 load_skill 工具
@tool
async def load_skill(skill_name: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """将技能的完整内容加载到智能体的上下文中。"""

    log.info(f"正在查找技能: {skill_name}")

    for skill in SKILLS:
        if skill["name"] == skill_name:
            log.info(f"✅ 技能加载成功: {skill_name}")
            return Command(
                update={
                    "messages": [ToolMessage(content=f"技能 '{skill_name}' 已加载。请使用对应工具获取详细数据。", tool_call_id=tool_call_id)],

                    "skills_loaded": [skill_name]
                }
            )

    # 未找到技能
    available = ", ".join(s["name"] for s in SKILLS)
    log.warning(f"技能未找到: {skill_name}，可用技能: {available}")
    return Command(
        update={
            "messages": [ToolMessage(
                content=f"未找到技能 '{skill_name}'。可用技能: {available}",
                tool_call_id=tool_call_id
            )]
        }
    )


# 创建基于Skills架构的智能体
async def create_skills_based_agent():
    # 创建MCP客户端获取所有工具
    # mcp_client = MultiServerMCPClient({
    #     "gaode": gaode_mcp_server_config,
    #     "12306": my12306_mcp_server_config,
    #     "fenxi": analysis_mcp_server_config,
    # })

    # # 获取所有工具并按服务器分类
    # gaode_tools = await mcp_client.get_tools(server_name="gaode")
    # railway_tools = await mcp_client.get_tools(server_name="12306")
    # fenxi_tools = await mcp_client.get_tools(server_name="fenxi")

    # print(f'所有工具数量 - 高德: {len(gaode_tools)}, 铁路: {len(railway_tools)}, 分析: {len(fenxi_tools)}')

    # # 验证工具名称
    # print("工具名称验证:")
    # for i, tool in enumerate(gaode_tools + railway_tools + fenxi_tools):
    #     tool_name = getattr(tool, 'name', '未知名称')
    #     print(f"工具 {i + 1}: {tool_name}")

    # # 按类别组织工具
    # categorized_tools = {
    #     "gaode": gaode_tools,
    #     "12306": railway_tools,
    #     "fenxi": fenxi_tools
    # }
    local_tools = {
        "weather": [get_weather],
        "search": [google_search],
        "math" :  [calculate_budget]
    }
    # 创建具备真正技能支持的单智能体
    model = ChatOpenAI(model="qwen3.5-35b-a3b", # 使用 OPENAI_MODEL_NAME qwen3.5-flash
                       base_url=OPENAI_BASE_URL,
                       api_key=OPENAI_API_KEY,
                       extra_body={"enable_thinking": False},
                       )
    agent = create_agent(
        model,
        tools=[load_skill],  # 初始只暴露load_skill工具
        middleware=[SkillMiddleware(local_tools),LoggingMiddleware()],  #categorized_tools
        state_schema=SkillState,
        system_prompt="""您是一个多功能智能助手，采用渐进式技能加载架构。

请**严格**遵循以下工作流程,严格按顺序执行，不要无意义地重复执行：

1) 使用 `load_skill("weather")` 加载天气技能，然后调用 `get_weather` 获取天气事实；
2) 使用 `load_skill("search")` 加载搜索技能，然后调用 `google_search(query: str)` 获取景点信息；
3) 使用 `load_skill("math")` 加载预算技能，然后调用 `calculate_budget(activities: List[str], base_cost: float = 500.0)` 计算预算；
在完成以上三步并获得各工具的返回结果后，基于所有工具输出给出最终计划。 未完成任一步骤不得输出最终答案。

可用技能领域：
- 天气领域 (weather): 获得天气信息
- 景点查询 (search): 地区经典景点查询
- 计算预算 (math): 数据统计,计算支出

请先加载技能，再使用相应工具！，**必须**用工具获取事实数据! 
联合工具执行的结果得到最后输出
""",
    )

    return agent


# 创建基于Skills架构的智能体
# 最小改动：直接运行协程并等待结果
skills_agent = asyncio.run(create_skills_based_agent())

# 等待 agent 的异步调用完成
res = asyncio.run(
    skills_agent.ainvoke({
        "messages": [{"role": "user", "content": "请帮我查询天气信息，然后规划北京周末旅行！"}]
    })
)


        
# 也可以完整打印最后一条消息，或按需查看
print(res["messages"][-1])

print("success")

# Print the conversation
# for message in result["messages"]:
#     if hasattr(message, 'pretty_print'):
#         message.pretty_print()
#     else:
#         print(f"{message.type}: {message.content}")