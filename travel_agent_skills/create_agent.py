# agent/create_agent.py
import uuid
from typing import TypedDict, NotRequired
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing import Callable
import asyncio
from pathlib import Path
from langchain.agents import create_agent
from middleware import SkillMiddleware,load_skill
import sys
from middleware import SkillState
from langchain_openai import ChatOpenAI
from skill_index import  SKILLS
#from agent.state import SkillState   # 你的状态 schema，如果有的话
# 添加项目根目录到路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # src目录
sys.path.insert(0, str(project_root))

from env_utils import OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL_NAME
model = ChatOpenAI(model="qwen3.5-35b-a3b", # 使用 OPENAI_MODEL_NAME qwen3.5-flash
                       base_url=OPENAI_BASE_URL,
                       api_key=OPENAI_API_KEY,
                       extra_body={"enable_thinking": False},
                       )

    
# Create skill middleware
class SkillMiddleware(AgentMiddleware):
    """Middleware that injects skill descriptions into the system prompt."""

   
    #注意这里的实现只是能加载仅包含文本内容就足够的skills

    def __init__(self):
        """Initialize and generate the skills prompt from SKILLS."""
        #初始暴露的工具
        self.tools = [load_skill]
        # Build skills prompt from the SKILLS list
        skills_list = []
        for skill in SKILLS:
            skills_list.append(
                f"- **{skill['name']}**: {skill['description']}"
            )
        self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync: Inject skill descriptions into system prompt."""
        # Build the skills addendum
        skills_addendum = (
            f"\n\n## Available Skills\n\n{self.skills_prompt}\n\n"
            "Use the load_skill tool when you need detailed information "
            "about handling a specific type of request."
        )

        # Append to system message content blocks
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)

# Initialize your chat model (replace with your model)
# Example: from langchain_anthropic import ChatAnthropic
# model = ChatAnthropic(model="claude-3-5-sonnet-20241022")


# Create the agent with skill support
agent = create_agent(
    model,
    system_prompt=(
        "You are a SQL query assistant that helps users "
        "write queries against business databases."
    ),
    middleware=[SkillMiddleware()],
    checkpointer=InMemorySaver(),
)

# Example usage
if __name__ == "__main__":
    # Configuration for this conversation thread
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Ask for a SQL query
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a SQL query to find all customers "
                        "who made orders over $1000 in the last month"
                    ),
                }
            ]
        },
        config
    )

    # Print the conversation
    for message in result["messages"]:
        if hasattr(message, 'pretty_print'):
            message.pretty_print()
        else:
            print(f"{message.type}: {message.content}")