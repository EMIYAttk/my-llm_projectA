from env_utils import OPENAI_BASE_URL,OPENAI_API_KEY,OPENAI_MODEL_NAME
"""
LangChain V1.0 完整示例项目
涵盖：Prompt工程、RAG、工具调用、记忆、多Agent、中间件、结构化输出,流式传输
"""

# requirements.txt
"""
langchain>=1.0
langchain-openai>=1.0
langchain-community>=1.0
chromadb
langgraph>=0.3.0
pydantic>=2.0

"""



# V1.0 新导入
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware, before_model, after_model, after_agent,SummarizationMiddleware,HumanInTheLoopMiddleware

# from langchain.agents.types import AgentConfig
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain.agents.structured_output import ToolStrategy
from langgraph.runtime import Runtime
# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# 其他
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
import os
from typing import TypedDict, NotRequired, Literal, Any,List,Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from pathlib import Path

# ==================== 全局配置 ====================

# 知识库路径
KB_DIR = Path("./knowledge_base")
KB_DIR.mkdir(exist_ok=True)

# 输出路径
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 向量数据库持久化路径
CHROMA_PERSIST_DIR = Path("./chroma_db")
CHROMA_PERSIST_DIR.mkdir(exist_ok=True)


# ==================== 1. 结构化输出定义 ====================

class ResearchOutput(BaseModel):
    """研究成果结构化输出"""
    summary: str = Field(description="研究总结")
    key_points: list[str] = Field(description="关键要点列表")
    confidence: Literal["high", "medium", "low"] = Field(description="置信度")
    sources: list[str] = Field(description="参考来源")

class ArticleOutput(BaseModel):
    """文章结构化输出"""
    title: str = Field(description="文章标题")
    content: str = Field(description="正文内容")
    tags: list[str] = Field(description="标签")
    word_count: int = Field(description="字数统计")


# ==================== 2. 中间件定义 ====================

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        state_dict = dict(state)  # 🔥 关键：转dict
        
        user_id = state_dict.get("user_id", "unknown")
        user_prefer = state_dict.get("preferences", {})
        
        print(f"About to call model with {len(state['messages'])} messages")
        
        print(f"""
=== 用户信息 ===
用户ID:   {user_id}
偏好设置: 
  - 兴趣: {', '.join(user_prefer.get('interests', []))}
  - 关注: {user_prefer.get('focus', '无')}
==================
""")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None


@after_model
def save_document(state: AgentState, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Writer Agent的结束中间件：自动保存结构化输出到本地文件。
    
    在模型生成响应后自动触发，检查state中的structured_response，
    将其中的title和content保存为Markdown或纯文本文件。
    
    保存路径: ./output/
    文件名: 基于title自动清理生成，如有重名则添加时间戳
    
    Args:
        state: Agent状态对象，包含messages和structured_response
        config: 可选配置，可包含format覆盖默认格式
    
    Returns:
        更新后的state，包含保存结果信息
    """
    try:
        # 找最后一个AIMessage（代理最终响应）
        last_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                last_msg = msg
                break
        
        if last_msg and last_msg.tool_calls:
            # ToolStrategy结构化输出在第一个tool_call的args
            structured_response = last_msg.tool_calls[0]["args"]
        
        # 提取title和content（支持多种可能的字段名）
        title = (
            structured_response.get("title") 
            or structured_response.get("Title") 
            or structured_response.get("文件名")
            or "untitled_document"
        )
        
        content = (
            structured_response.get("content") 
            or structured_response.get("Content")
            or structured_response.get("正文")
            or structured_response.get("body")
            or str(structured_response)  # 兜底：转为字符串
        )
        
        # 获取格式（从config或默认markdown）
        format_type = "markdown"
        if config and isinstance(config, dict):
            format_type = config.get("format", "markdown")
        
        # 验证数据
        if not title or not content:
            print(f"⚠️ [保存中间件] 数据不完整: title={bool(title)}, content={bool(content)}")
            return state
        
        # ===== 保存逻辑 =====
        
        # 清理文件名
        safe_title = "".join(c for c in str(title) if c.isalnum() or c in ('-', '_', ' '))
        safe_title = safe_title.strip().replace(' ', '_')[:50]
        
        if not safe_title:
            safe_title = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 确定扩展名
        ext = ".md" if format_type.lower() in ["markdown", "md"] else ".txt"
        filename = f"{safe_title}{ext}"
        file_path = OUTPUT_DIR / filename
        
        # 处理重名
        if file_path.exists():
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"{safe_title}_{timestamp}{ext}"
            file_path = OUTPUT_DIR / filename
        
        # 构建文件内容
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if format_type.lower() in ["markdown", "md"]:
            header = f"""---
title: {title}
created_at: {now_str}
format: markdown
source: AI生成
---

# {title}

生成时间: {now_str}

"""
        else:
            header = f"""标题: {title}
创建时间: {now_str}
格式: 纯文本
来源: AI生成
{'='*50}

"""
        
        full_content = header + str(content)
        
        # 写入文件
        file_path.write_text(full_content, encoding='utf-8')
        
        # 构建保存结果
        save_result = {
            "success": True,
            "filename": filename,
            "path": str(file_path.absolute()),
            "size_bytes": file_path.stat().st_size,
            "char_count": len(str(content)),
            "timestamp": now_str
        }
        
        # 更新state（添加保存结果供后续使用）
        if hasattr(state, 'save_result'):
            state.save_result = save_result
        else:
            state["save_result"] = save_result
        
        # 打印日志
        print(f"""✅ [保存中间件] 文档已保存
📄 {filename}
📁 {file_path.absolute()}
📊 {len(str(content))} 字符
💾 {file_path.stat().st_size} 字节""")
        
        return state
        
    except Exception as e:
        error_msg = f"❌ [保存中间件] 保存失败: {str(e)}"
        print(error_msg)
        
        # 记录错误但不中断流程
        error_result = {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        if hasattr(state, 'save_result'):
            state.save_result = error_result
        else:
            state["save_result"] = error_result
            
        return state

class ApprovalMiddleware(AgentMiddleware):
    """人工审批中间件（用于关键工具）"""
    def before_model(self, state, runtime):
        # 检查是否是敏感操作
        last_msg = state.messages[-1]
        if isinstance(last_msg, HumanMessage) and "删除" in last_msg.content:
            print("⚠️ 检测到敏感操作，需要审批...")
            # 实际项目中这里可以接入审批系统
            # 返回 None 继续执行，或返回跳转指令
        return None




# ==================== 3. 工具定义 ====================

# ==================== 文档加载器工具 ====================

class TextLoader:
    """简单的文本文件加载器"""
    
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = Path(file_path)
        self.encoding = encoding
    
    def load(self) -> List[Document]:
        """加载文件内容为Document对象"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        
        content = self.file_path.read_text(encoding=self.encoding)
        
        # 创建Document对象（模拟LangChain的Document）
        return [Document(
            page_content=content,
            metadata={
                "source": str(self.file_path.name),
                "file_path": str(self.file_path.absolute()),
                "file_type": self.file_path.suffix,
                "load_time": datetime.now().isoformat()
            }
        )]


@tool
def load_document(file_name: str) -> str:
    """
    从本地知识库加载文档内容。用于读取本地文本文件、Markdown文件等获取信息。
    
    当需要查询本地资料、历史文档、已保存的研究报告时使用此工具。
    支持.txt, .md, .csv等文本格式。
    
    Args:
        file_name: 文件名（如"report.txt"或"notes.md"），文件应位于./knowledge_base/目录
    
    Returns:
        文档的完整内容，以及文件元信息。
        如果文件不存在，会提示可用的文件列表。
    
    Examples:
        >>> load_document("ai_trends.txt")
        >>> load_document("meeting_notes.md")
    """
    try:
        # 安全检查：防止目录遍历攻击
        safe_name = Path(file_name).name  # 只取文件名，去掉路径
        file_path = KB_DIR / safe_name
        
        if not file_path.exists():
            # 列出可用文件
            available_files = list(KB_DIR.glob("*.txt")) + list(KB_DIR.glob("*.md")) + list(KB_DIR.glob("*.csv"))
            available_names = [f.name for f in available_files]
            
            if available_names:
                files_list = "\n".join([f"  - {name}" for name in available_names])
                return f"""❌ 文件 '{file_name}' 不存在于知识库。
                
📁 当前可用文件：
{files_list}

💡 提示：将文件放入 ./{KB_DIR}/ 目录后重试"""
            else:
                return f"""❌ 知识库为空，未找到文件 '{file_name}'。
                
请执行以下操作：
1. 在 ./{KB_DIR}/ 目录创建文本文件
2. 或运行 create_sample_document() 创建示例文档"""
        
        # 加载文档
        loader = TextLoader(str(file_path))
        docs = loader.load()
        doc = docs[0]  # 取第一个（也是唯一一个）
        
        content = doc.page_content
        meta = doc.metadata
        
        # 格式化输出
        result = f"""📄 文档加载成功
        
文件: {meta['source']}
类型: {meta['file_type']}
大小: {len(content)} 字符
路径: {meta['file_path']}

{'='*50}
{content[:2000]}{'... [内容已截断]' if len(content) > 2000 else ''}
{'='*50}"""
        
        return result
        
    except UnicodeDecodeError:
        return f"❌ 文件编码错误: '{file_name}' 不是有效的UTF-8文本文件"
    except Exception as e:
        return f"❌ 加载失败: {str(e)}"


@tool  
def list_documents() -> str:
    """
    列出知识库中所有可用的文档文件。
    
    当不确定文件名或想查看有哪些资料可用时使用。
    
    Returns:
        文件列表，包括文件名、大小和修改时间。
    
    Examples:
        >>> list_documents()
    """
    try:
        files = []
        for pattern in ["*.txt", "*.md", "*.csv", "*.json"]:
            files.extend(KB_DIR.glob(pattern))
        
        if not files:
            return f"""📂 知识库目录: {KB_DIR.absolute()}
            
状态: 空（暂无文档）

💡 添加文档方法：
1. 手动复制文件到上述目录
2. 使用 save_document 工具保存新文档"""
        
        # 格式化文件信息
        lines = [f"📂 知识库目录: {KB_DIR.absolute()}\n", f"共找到 {len(files)} 个文件:\n"]
        
        for i, f in enumerate(sorted(files), 1):
            stat = f.stat()
            size = stat.st_size
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            
            # 格式化文件大小
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size/(1024*1024):.1f}MB"
            
            lines.append(f"{i}. {f.name:<20} {size_str:>8}  {mtime}")
        
        lines.append(f"\n💡 提示: 使用 load_document('文件名') 读取内容")
        return "\n".join(lines)
        
    except Exception as e:
        return f"❌ 无法读取目录: {str(e)}"
@tool
def rag_search(query: str) -> str:
    """检索知识库相关文档"""
    # 模拟RAG（实际项目连接真实数据库）
    mock_docs = [
        Document(page_content="AI Agent 是2025年最热门的技术趋势..."),
        Document(page_content="多Agent协作系统正在取代单一Agent架构..."),
    ]
    return "\n".join([d.page_content for d in mock_docs])

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def search_web(query: str) -> str:
    """网络搜索（模拟）"""
    return f"搜索结果: {query} 的最新信息..."




# ==================== 时间获取工具（已完善） ====================

@tool
def get_current_time(timezone: str = "local") -> str:
    """
    获取当前日期和时间。
    
    用于在回答中提供时效性信息以及检查当前回答是否过时，或作为文件时间戳。
    
    Args:
        timezone: 时区，"local"(本地时间，默认)或"UTC"
    
    Returns:
        格式化的当前时间信息。
    
    Examples:
        >>> get_current_time()
        >>> get_current_time("UTC")
    """
    try:
        if timezone.upper() == "UTC":
            now = datetime.utcnow()
            tz_name = "UTC"
        else:
            now = datetime.now()
            tz_name = "本地时间"
        
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        
        return f"""📅 当前时间 ({tz_name})
        
{now.strftime("%Y年%m月%d日")} {weekdays[now.weekday()]}
{now.strftime("%H:%M:%S")}
ISO: {now.isoformat()}"""
        
    except Exception as e:
        return f"❌ 获取时间失败: {str(e)}"
# @tool
# def save_article(title: str, content: str) -> str:
#     """保存文章到数据库"""
#     # 模拟保存
#     return f"✅ 文章《{title}》已保存（ID: {hash(title) % 10000}）"


# ==================== 4. 短期记忆 状态定义() ====================

class MultiAgentState(AgentState):
    """多Agent工作流状态（V1.0 TypedDict强制）"""
    # 继承自 AgentState 的 messages
    current_agent: NotRequired[Literal["researcher", "writer", "reviewer"]]
    research_result: NotRequired[ResearchOutput]
    article: NotRequired[ArticleOutput]
    user_id: NotRequired[str]
    session_id: NotRequired[str]

class CustomAgentState(AgentState):  # [!code highlight]
    user_id: str  # [!code highlight]
    preferences: dict  # [!code highlight]

# ==================== 5. Prompt 工程 ====================

RESEARCHER_PROMPT = """你是专业的AI研究员。你的任务是，根据用户问题和偏好：
1. 使用RAG工具检索知识库
2. 使用网络搜索获取最新信息
3. 分析并生成结构化研究报告

输出格式要求：
- 使用工具调用生成 ResearchOutput 结构化数据
- 置信度评估要客观
- 列出所有参考来源"""

WRITER_PROMPT = """你是专业的技术写手。你的任务是：
1. 根据研究报告（ResearchOutput 结构化数据）和用户偏好撰写高质量文章
2. 生成 ArticleOutput 结构化数据

写作风格：专业、清晰、有洞察力"""

REVIEWER_PROMPT = """你是内容审核员。审核文章质量，决定是否：
- APPROVE：直接发布
- REVISE：退回修改（说明原因）
- REJECT：拒绝发布"""


# ==================== 6. Agent 创建（V1.0 新API） ====================

def create_agents():
    """创建多Agent系统"""
    config = {"configurable": {"thread_id": "test-memory-123"}}
    # V1.0: 使用模型字符串而非实例
    model = ChatOpenAI(model="qwen3.5-flash",
                       base_url=OPENAI_BASE_URL,
                       api_key=OPENAI_API_KEY,
                       extra_body={"enable_thinking": False},
                       
                       
                       )
    
    # 研究员 Agent
    researcher = create_agent(
        model,
        tools=[load_document, list_documents,search_web, calculator,get_current_time],
        system_prompt=RESEARCHER_PROMPT,
        state_schema=CustomAgentState,  # [!code highlight]
        #checkpointer=InMemorySaver(),   短期记忆设置，但这个agent在这个项目暂且不需要这个功能
        response_format=ToolStrategy(ResearchOutput),
        middleware=[
            SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=3000,  # 在 4000 个 token 时触发摘要
            messages_to_keep=20,  # 摘要后保留最近 20 条消息
            summary_prompt= """你是研究员Agent的上下文管理助手。当对话历史过长时，你的任务是对历史消息进行智能摘要，确保研究员能继续高效工作。

## 摘要原则

### 1. 信息保留优先级（高到低）
- **研究结论**：已确认的事实、数据、分析结果
- **工具调用记录**：已执行的搜索查询和关键发现
- **用户原始需求**：研究主题、特殊要求、约束条件
- **待验证假设**：正在调查但未确认的观点
- **过程性内容**：思考过程、试错记录（可简化） 
"""
            ),
            LoggingMiddleware()],
        name="researcher"
    )
    
    # 写手 Agent
    writer = create_agent(
        model,
        tools=[get_current_time],
        system_prompt=WRITER_PROMPT,
        state_schema=CustomAgentState,
        response_format=ToolStrategy(ArticleOutput),
        middleware=[LoggingMiddleware(),save_document],
        name="writer"
    )
    
    reviewer_schema = {
            "type": "object",
            "properties": {
                "decision": {"enum": ["APPROVE", "REVISE", "REJECT"]},
                "reason": {"type": "string"},
                "suggestions": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["decision", "reason"]
        }
    
    # 审核员 Agent（简单版本，使用结构化输出判断）
    reviewer = create_agent(
        model,
        tools=[],
        system_prompt=REVIEWER_PROMPT,
        state_schema=CustomAgentState,
        response_format = reviewer_schema,
        name="reviewer"
    )
    
    return researcher, writer, reviewer


# ==================== 7. 多Agent 工作流（Handoff 模式） ====================

class SupervisorState(TypedDict):
    messages: list
    next: NotRequired[Literal["researcher", "writer", "reviewer", "end"]]
    research_data: NotRequired[dict]
    article_data: NotRequired[dict]
    user_id: str  # [!code highlight]
    preferences: dict  # [!code highlight]
    

def create_supervisor_workflow():
    """创建Supervisor多Agent工作流"""
    
    researcher, writer, reviewer = create_agents()
    
    def supervisor_node(state: SupervisorState):
        """Supervisor 路由决策"""
        last_message = state["messages"][-1]
        content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # 简单的路由逻辑（实际可用LLM判断）
        if "研究" in content or "分析" in content:
            return Command(
            goto="researcher",
            update={
                "next": "researcher",
                # 传递上下文信息
                "user_id": state.get("user_id"),
                "preferences": state.get("preferences"),
            }
        )
        elif "写" in content or "文章" in content:
            return Command(
            goto="writer",
            update={
                "next": "writer",
                "user_id": state.get("user_id"),
                "preferences": state.get("preferences"),
            }
        )
        else:
            return Command(
            goto="researcher",
            update={
                "next": "researcher",
                # 传递上下文信息
                "user_id": state.get("user_id"),
                "preferences": state.get("preferences"),
            }
        )
    
    def researcher_node(state: SupervisorState):
        """研究员节点"""
        # V1.0: 使用运行时上下文传递额外信息
        config = {"configurable": {"thread_id": "test-memory-123"}}
        user_id = state.get("user_id", "anonymous")
        preferences = state.get("preferences", {})
    
        print(f"🔍 研究员收到: user_id={user_id}, prefs={preferences}")
        result = researcher.invoke(
            {"messages": state["messages"],
            "user_id": user_id,
            "preferences": preferences
           
            }
             #config    #短期记忆的线程化配置
        )
        # ===== 验证记忆：检查状态 =====
        # state1 = researcher.get_state(config)  # 🔥 关键：查看持久化状态
        # print("\n🔍 记忆状态检查:")
        # print("消息历史:", len(state1.values["messages"]))
        # print("最后一条消息:", state1.values["messages"][-1].content[:50])
     
        # print("配置:", state1.config["configurable"]["thread_id"])
        # 提取结构化输出
        research_data = None
        for msg in result["messages"]:
            if hasattr(msg, 'content_blocks'):
                for block in msg.content_blocks:
                    if block.get("type") == "structured_output":
                        research_data = block.get("data")
        
        return Command(
            goto="writer",
            update={
                "messages": result["messages"],
                "research_data": research_data,
                # 传递上下文信息
                "user_id": state.get("user_id"),
                "preferences": state.get("preferences"),
                "next": "writer"
            }
        )
    
    def writer_node(state: SupervisorState):
        """写手节点"""
        # 构建包含研究结果的提示
        messages = state["messages"] + [
            SystemMessage(content=f"基于以下研究数据写作：{state.get('research_data', {})}")
        ]
        user_id = state.get("user_id", "anonymous")
        preferences = state.get("preferences", {})
    
        print(f"🔍 写手收到: user_id={user_id}, prefs={preferences}")
        
        result = writer.invoke({"messages": messages,
                                "user_id": user_id,
                                "preferences": preferences
                                }
                               
                               
                               
                               )
        
        return Command(
            goto="reviewer",
            update={
                "messages": result["messages"],
                "next": "reviewer"
            }
        )
    
    def reviewer_node(state: SupervisorState):
        """审核员节点"""
        result = reviewer.invoke({"messages": state["messages"]})
        
        # 解析审核决定
        decision = "APPROVE"  # 简化处理
        
        if decision == "APPROVE":
            return Command(goto=END, update={"next": "end"})
        else:
            return Command(goto="writer", update={"next": "writer"})
    
    # 构建图
    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)
    
    workflow.add_edge(START, "supervisor")
    
    return workflow.compile()


# ==================== 8. 简单顺序工作流（替代方案） ====================

def create_sequential_workflow():
    """简单的顺序工作流：研究 -> 写作 -> 审核"""
    
    researcher, writer, reviewer = create_agents()
    
    class SimpleState(TypedDict):
        messages: list
        research_output: NotRequired[ResearchOutput]
        article_output: NotRequired[ArticleOutput]
    
    def research_step(state: SimpleState):
        result = researcher.invoke({"messages": state["messages"]})
        return {"messages": result["messages"]}
    
    def write_step(state: SimpleState):
        # 添加上下文提示
        enhanced_messages = state["messages"] + [
            SystemMessage(content="现在根据上述研究结果撰写文章。")
        ]
        result = writer.invoke({"messages": enhanced_messages})
        return {"messages": result["messages"]}
    
    workflow = StateGraph(SimpleState)
    workflow.add_node("research", research_step)
    workflow.add_node("write", write_step)
    
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "write")
    workflow.add_edge("write", END)
    
    return workflow.compile()

def create_sample_documents():
    """
    创建多样化的示例文档，展示不同格式和内容类型
    包括：技术趋势、产品文档、会议纪要、数据分析、教程指南
    """
    
    
    # 1. AI技术趋势报告（Markdown格式，带标题层级）
    ai_trends = KB_DIR / "ai_trends_2025.md"
    ai_trends.write_text("""# 2025年AI技术趋势报告

## 核心趋势

### 1. AI Agent爆发
AI Agent成为主流，具备自主规划、工具调用、长期记忆能力。OpenAI、Google、Anthropic竞相发布Agent框架。

### 2. 多模态统一
GPT-4V、Claude 3实现文本、图像、视频统一理解，应用场景扩展到自动驾驶、医疗影像。

### 3. 边缘AI普及
Llama 3、Phi-3等小模型可在手机运行，保护隐私的同时提供智能助手能力。

## 关键数据
- Agent市场规模：预计2025年达50亿美元
- 多模态模型准确率：图像理解超95%
- 边缘模型体积：最小可达1GB以下

> 数据来源：Gartner 2025技术成熟度曲线
""", encoding='utf-8')
    print(f"✅ 创建: {ai_trends.name}")

    # 2. 产品需求文档（Markdown表格格式）
    prd_doc = KB_DIR / "product_requirements.md"
    prd_doc.write_text("""# 智能客服系统PRD v1.0

## 功能需求

| 模块 | 功能 | 优先级 | 状态 |
|------|------|--------|------|
| 对话引擎 | 多轮对话管理 | P0 | 开发中 |
| 知识库 | RAG检索增强 | P0 | 待启动 |
| 分析看板 | 会话数据分析 | P1 | 规划中 |
| 人工接管 | 复杂问题转人工 | P1 | 设计中 |

## 非功能需求
- 响应时间：< 2秒
- 并发支持：1000 QPS
- 可用性：99.9%

## 风险点
1. 大模型幻觉问题需RAG缓解
2. 用户隐私数据合规存储
""", encoding='utf-8')
    print(f"✅ 创建: {prd_doc.name}")

    # 3. 项目会议纪要（纯文本，时间线格式）
    meeting_notes = KB_DIR / "meeting_20250115.txt"
    meeting_notes.write_text("""项目周会纪要 - 2025年1月15日

参会：张三、李四、王五、赵六

=== 上周进展 ===
[张三] 完成用户登录模块开发，通过率98%
[李四] 设计评审通过，进入编码阶段
[王五] 修复3个P1级Bug，系统稳定性提升

=== 本周计划 ===
[全员] 周三进行集成测试
[李四] 完成支付接口对接
[赵六] 准备上线文档

=== 问题与风险 ===
- 第三方API响应慢，需增加缓存层（负责人：张三，截止日期：1月20日）
- 测试环境配置延迟，可能影响进度

=== 行动项 ===
1. 申请生产环境权限 - 王五 1/17前
2. 更新接口文档 - 赵六 1/18前
3. 下周客户演示准备 - 全员 1/22前

记录人：张三
""", encoding='utf-8')
    print(f"✅ 创建: {meeting_notes.name}")

    # 4. 数据分析报告（CSV格式，结构化数据）
    sales_data = KB_DIR / "sales_q4_2024.csv"
    sales_data.write_text("""月份,产品类别,销售额(万元),订单量,客户满意度
2024-10,云服务,1250,45,4.5
2024-10,SaaS产品,890,120,4.7
2024-10,咨询服务,560,8,4.8
2024-11,云服务,1380,52,4.6
2024-11,SaaS产品,920,135,4.6
2024-11,咨询服务,620,10,4.9
2024-12,云服务,1520,58,4.4
2024-12,SaaS产品,1050,150,4.8
2024-12,咨询服务,750,12,4.7

Q4总计:
云服务: 4150万元 (155订单)
SaaS产品: 2860万元 (405订单)  
咨询服务: 1930万元 (30订单)
整体客户满意度: 4.6/5.0
""", encoding='utf-8')
    print(f"✅ 创建: {sales_data.name}")

    # 5. API接口文档（JSON格式）
    api_spec = KB_DIR / "api_endpoints.json"
    api_spec.write_text("""{
  "api_version": "v2.1",
  "base_url": "https://api.example.com",
  "endpoints": [
    {
      "path": "/auth/login",
      "method": "POST",
      "description": "用户登录获取JWT令牌",
      "params": ["username", "password"],
      "response": {"token": "string", "expires_in": 3600}
    },
    {
      "path": "/users/{id}",
      "method": "GET", 
      "description": "获取用户信息",
      "auth_required": true,
      "response": {"id": "integer", "name": "string", "role": "string"}
    },
    {
      "path": "/orders",
      "method": "POST",
      "description": "创建新订单",
      "auth_required": true,
      "rate_limit": "100/min"
    }
  ],
  "notes": "所有请求需包含Content-Type: application/json头"
}""", encoding='utf-8')
    print(f"✅ 创建: {api_spec.name}")
# ==================== 9. 运行演示 ====================

def main():
    print("=" * 50)
    print("LangChain V1.0 多Agent系统演示")
    print("=" * 50)
    #create_sample_documents()
    # 创建工作流
    app = create_supervisor_workflow()
    # 或使用简单版本：app = create_sequential_workflow()
    
    # 初始输入
    initial_state = {
        "messages": [HumanMessage(content="研究AI Agent及其应用开发最新趋势并撰写技术文章")],
        "user_id": "user_123",
        "preferences": {"interests": ["AI应用开发", "Agent框架", "多Agent系统"],
    "focus": "最新技术趋势",},
        
    }
    
    print("\n🚀 启动工作流...")
    
    # 流式执行（V1.0 新特性）
    for event in app.stream(initial_state, stream_mode="updates"):
        
        if hasattr(event, 'pretty_print'):
            event.pretty_print()
        else:
            print(f"\n📦 事件内容: {event}")
        # # 展示内容块（V1.0 新特性）
        # for step, data in event.items():
        #     print(f"step: {step}")
        #     print(f"content: {data['messages'][-1].content_blocks}")
    
    print("\n✅ 工作流完成！")

if __name__ == "__main__":
    main()