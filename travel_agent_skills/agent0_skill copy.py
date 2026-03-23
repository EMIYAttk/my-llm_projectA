import sys
from pathlib import Path
import os
import re
from typing import TypedDict, Annotated, Sequence, Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import importlib

from langchain_openai import ChatOpenAI

# 添加项目根目录到路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # src目录
sys.path.insert(0, str(project_root))

from env_utils import OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL_NAME

# 在文件的顶部，或者在 skill_planner 函数之前，定义技能的严格顺序
SKILL_ORDER = ["weather", "search", "math", "write"]

SKILLS_DIR = Path("skills")
os.makedirs("plan", exist_ok=True)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    loaded_skills: List[str]
    available_tools: Dict[str, Any]
    plan_files: Dict[str, str]

SKILL_MAP = {
    "weather": "weather", "search": "search", 
    "math": "math", "write": "write",
    "预算": "math", "景点": "search", "行程": "write"
}

@tool
def load_skill(skill_name: str) -> str:
    """加载技能文件，自动注册工具."""
    skill_name = SKILL_MAP.get(skill_name.lower(), skill_name.lower())
    skill_path = SKILLS_DIR / skill_name / "SKILL.md"
    
    if not skill_path.exists():
        return f"❌ 未找到 {skill_name} (路径: {skill_path}). 可用: {list(SKILLS_DIR.glob('*'))}"
    
    with open(skill_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 动态加载工具
    try:
        tool_module = importlib.import_module(f"tools.{skill_name}")
        tool_name = {
            "weather": "get_weather", "search": "google_search",
            "math": "calculate_budget", "write": "generate_itinerary"
        }[skill_name]
        new_tool = getattr(tool_module, tool_name)
        return f"✅ [{skill_name.upper()}] 加载成功！工具 '{tool_name}' 已注册\n\n{ content[:300] }..."
    except Exception as e:
        return f"❌ 工具加载失败 {skill_name}: {e}"

@tool
def read_plan_file(filename: str) -> str:
    """读取规划文件内容."""
    path = Path("plan") / filename
    return path.read_text(encoding="utf-8") if path.exists() else f"❌ {filename} 未生成"

@tool
def write_plan_file(filename: str, content: str) -> str:
    """保存中间规划结果."""
    path = Path("plan") / filename
    path.parent.mkdir(exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"💾 保存成功: plan/{filename} ({len(content)}字符)"

def skill_planner(state: AgentState) -> Dict[str, Any]:
    """LLM智能规划下一个技能."""
    current_skills = state.get("loaded_skills", [])
    
    # 程序化地确定下一个要加载的技能
    next_skill = None
    for skill in SKILL_ORDER:
        if skill not in current_skills:
            next_skill = skill
            break
    
    if next_skill is None:
        return {
            "messages": [AIMessage(content="所有预定技能已加载完毕。")],
            "loaded_skills": current_skills,
            "available_tools": state.get("available_tools", {}) # 确保返回现有工具，即使没有新加载
        }

    model = ChatOpenAI(model="qwen3.5-35b-a3b", # 使用 OPENAI_MODEL_NAME qwen3.5-flash
                       base_url=OPENAI_BASE_URL,
                       api_key=OPENAI_API_KEY,
                       extra_body={"enable_thinking": False},
                       )
    
    # 调整LLM的提示，让它基于已确定的下一个技能生成规划消息
    prompt = f"""已加载技能: {current_skills}
目标: 北京周末旅行

根据预设的技能加载顺序，下一个要加载的技能是 '{next_skill}'。
请生成一个简短的规划消息，说明为什么现在要加载 '{next_skill}' 技能。
"""
    
    response_from_llm = model.invoke([HumanMessage(content=prompt)]).content
    
    print(f"🤖 规划: 选择 '{next_skill}' (当前: {len(current_skills)}/4)")
    
    # 执行加载 + 注册
    load_result = load_skill.invoke({"skill_name": next_skill})
    
    # ✅ 解析工具名并注册（状态更新）
    tool_match = re.search(r"工具 '(\w+)' 已注册", load_result)
    if tool_match:
        tool_name = tool_match.group(1)
        try:
            # 确保 tools 目录在 sys.path 中，以便 importlib 找到
            if str(Path("tools").resolve()) not in sys.path:
                sys.path.insert(0, str(Path("tools").resolve()))

            # --- 修改点 1: 修正 importlib.import_module 的路径 ---
            tool_module = importlib.import_module(f"tools.{next_skill}") 
            new_tool = getattr(tool_module, tool_name)
            state["available_tools"] = state.get("available_tools", {})
            state["available_tools"][tool_name] = new_tool
            print(f"   ✅ 注册工具: {tool_name}")
        except Exception as e:
            print(f"   ❌ 注册工具失败 {tool_name}: {e}") # 打印错误以便调试
            pass  # 容错
    
    return {
        "messages": [AIMessage(content=f"规划: {response_from_llm}\n{load_result}")],
        "loaded_skills": current_skills + [next_skill],
        # --- 修改点 2: 将更新后的 available_tools 返回到 LangGraph 状态 ---
        "available_tools": state["available_tools"] 
    }

def tools_node(state: AgentState):
    """动态工具执行."""
    avail_tools = list(state.get("available_tools", {}).values())
    if avail_tools:
        print(f"🔧 执行工具: {len(avail_tools)} 个可用")
        return ToolNode(avail_tools + [read_plan_file, write_plan_file]).invoke(state)
    return state

def route_next(state: AgentState) -> str:
    """路由逻辑."""
    skills_count = len(state.get("loaded_skills", []))
    if skills_count < 4:
        return "planner"
    return END

# 构建 + 编译
workflow = StateGraph(AgentState)
workflow.add_node("planner", skill_planner)
workflow.add_node("tools", tools_node)

workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", route_next, {"planner": "planner", END: END})
workflow.add_edge("tools", END) # 当工具执行完毕后，直接结束

app = workflow.compile(checkpointer=MemorySaver())

# ✅ 完整 inputs 初始化
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "perfect_travel"}}
    inputs = {
        "messages": [HumanMessage(content="规划北京周末旅行！")],
        "loaded_skills": [],
        "available_tools": {},
        "plan_files": {}
    }
    
    print("🚀 纯动态加载：从零工具开始")
    for event in app.stream(inputs, config, stream_mode="updates"):
        node, update = next(iter(event.items()))
        print(f"\n📦 [{node.upper()}]")
        
        skills = update.get('loaded_skills', [])
        tools = list(update.get('available_tools', {}).keys())
        print(f"  技能({len(skills)}): {skills}")
        print(f"  工具({len(tools)}): {tools}")
        
        if "messages" in update and update["messages"]:
            msg = update["messages"][-1].content
            print(f"  输出: {msg[:100]}{'...' if len(msg)>100 else ''}")
    
    # 最终验证
    final = app.get_state(config).values
    print(f"\n🏆 完美完成！")
    print(f"技能链: {final['loaded_skills']}")
    print(f"工具集: {list(final['available_tools'].keys())}")
    print(f"规划文件: {list(final['plan_files'].keys())}")
