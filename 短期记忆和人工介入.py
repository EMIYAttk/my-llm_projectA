from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI
from env_utils import OPENAI_BASE_URL,OPENAI_API_KEY,OPENAI_MODEL_NAME
from langgraph.types import Command

@tool
def summarize(text: str) -> str:
    """
    总结文本中的内容
    
    Args:
        text: 原始文本
    """
    return f"Summary: {text[:8]}..."


model = ChatOpenAI(model="qwen3.5-flash",   #qwen3.5-35b-a3b
                       base_url=OPENAI_BASE_URL,
                       api_key=OPENAI_API_KEY,
                       extra_body={"enable_thinking": False},
                       
                       
                       )
agent = create_agent(
    model,
    tools=[summarize],
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"summarize": True})
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "single-agen"}}
result = agent.invoke({"messages": [{"role": "user", "content": "请总结这段文本：全球经济趋势摘要。世界的90%财富都掌握在不到1%的人手里。"}]}, config)

#展示中断信息，被中断的工具名称，参数，可以使用的决策等
print(result['__interrupt__'])


# 人工决策后 resume
result2 = agent.invoke(
    Command(
        # 决策以列表形式提供，每个待审查操作一个。
        # 决策的顺序必须与
        # `__interrupt__` 请求中列出的操作顺序匹配。
        resume={
            "decisions": [
                {
                    "type": "edit",
                    # 包含工具名称和参数的已编辑操作
                    "edited_action": {
                        # 要调用的工具名称。
                        # 通常与原始操作相同。
                        "name": "summarize",
                        # 传递给工具的参数。
                        "args": {"text":"在编辑工具参数时，请保守地进行更改"},
                    }
                }
            ]
        }
    ),
    config=config  # 相同的线程 ID 以恢复暂停的对话
)
print(result2["messages"][-1].content)        #应该是打印最后的Toolmessage里的content，能证明修改成功