from langchain.agents import create_agent, AgentState
import asyncio
import importlib
from typing import Callable, List, Dict
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langgraph.types import Command
from langchain.tools import BaseTool
from typing_extensions import Annotated
from skill_index import build_skill_index
import logging
from langchain.tools import tool, ToolRuntime   # ToolRuntime 隐藏于模型，不会出现在工具 schema
log = logging.getLogger(__name__)

# agent/load_skill_tool.py
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage

# 定义技能状态Schema
class SkillState(AgentState):
    """技能状态管理"""
    skills_loaded: Annotated[List[str], lambda current, new: current + [s for s in new if s not in current]] = []
    
# 如果你使用的是 langchain 的 @tool 装饰器， 保持导入正确
@tool
def load_skill(skill_name: str, runtime: ToolRuntime) -> Command:
    """
    把技能名写入 state（由中间件在下一次请求中动态加载工具）
    runtime 是隐藏的运行时参数，可以用来获取当前 tool_call_id
    """
    tool_call_id = getattr(runtime, "tool_call_id", None)
    return Command(
        update={
            "messages": [ToolMessage(content=f"已加载技能: {skill_name}", tool_call_id=tool_call_id)],
            "skills_loaded": [skill_name]
        }
    )
    
class SkillMiddleware(AgentMiddleware):
    """
    中间件维护：
      - skill_index: 启动时的元数据（skill -> tools_module）
      - available_skill_tools: skill -> 已 import 的工具实例列表（缓存）
      - base_tools: 永远暴露给模型的基础工具（如 load_skill）
    """

    def __init__(self, skills_root: str = "./skills"):
        super().__init__()
        self.skill_index = build_skill_index(skills_root)    # 预索引元数据（轻量）
        self.available_skill_tools: Dict[str, List[BaseTool]] = {}
        self.base_tools = [load_skill]

    def _import_tools_for_skill(self, skill_name: str) -> List[BaseTool]:
        """按需动态 import 并缓存工具（如果已经缓存则直接返回）"""
        if skill_name in self.available_skill_tools:
            return self.available_skill_tools[skill_name]

        meta = self.skill_index.get(skill_name)
        if not meta:
            log.warning(f"未在 skill_index 中找到技能: {skill_name}")
            self.available_skill_tools[skill_name] = []
            return []

        module_path = meta.get("tools_module")
        try:
            module = importlib.import_module(module_path)
            tools = getattr(module, "TOOLS", [])
            # 可验证每个 tool 是否有 name/description 等
            self.available_skill_tools[skill_name] = tools
            log.info(f"技能 {skill_name} 的工具已导入，数量: {len(tools)}")
            return tools
        except Exception as e:
            log.exception(f"导入技能 {skill_name} 的工具模块 {module_path} 失败: {e}")
            self.available_skill_tools[skill_name] = []
            return []

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        try:
            current_skills = set(request.state.get("skills_loaded", []))
            dynamic_tools = self.base_tools.copy()

            # 对每个当前已加载技能按需 import（首次加载会动态 import）
            for skill in current_skills:
                # 把同步导入放到线程池中，避免阻塞事件循环
                skill_tools = await asyncio.to_thread(self._import_tools_for_skill, skill)
                dynamic_tools.extend(skill_tools)

            # 可在此处修改 system_message（例如加入技能说明）
            new_system_message = getattr(request, "system_message", None)
            if new_system_message and hasattr(new_system_message, "content"):
                # 可选择把技能描述注入 system prompt（可选）
                pass

            modified_request = request.override(
                tools=dynamic_tools,
                system_message=new_system_message
            )

            response = await handler(modified_request)
            return response

        except Exception as e:
            log.exception(f"SkillMiddleware 运行出错: {e}")
            # 回退到基础工具
            fallback_request = request.override(tools=self.base_tools)
            return await handler(fallback_request)