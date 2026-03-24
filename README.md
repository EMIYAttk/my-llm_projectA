当前项目的所有依赖项安装命令如下：
pip install langchain langchain-openai dotenv

pip install langchain-huggingface

pip install sqlalchemy pymysql loguru

pip install langgraph-checkpoint-postgres

安装完依赖后，要创建一个.env文件在项目根文件夹路径下，在.env文件里填入OPENAI_BASE_URL，OPENAI_API_KEY等环境变量。
在env_utils.py里进行导入，支持第三方api调用网站，只要改变OPENAI_BASE_URL的值和对应密钥即可。


多Agent文档处理系统：
设计了研究员、写手、审核员三个专用Agent，通过LangGraph Supervisor工作流实现智能路由，集成中间件，本地RAG检索等。
运行方法（在安装好依赖的虚拟环境下)：

python agent0.py

智能规划旅行智能体：

在travel_agent_skills文件夹下及含有skill的py文件，最初的目标是仿照Agent skills 架构，在skills文件夹下创建相关技能，含有.md核心介绍描述文件和对应工具函数文件等。
如果只是需要技能的文字描述加入上下文，现在可以做到在智能体创建后逐步加载工作需要的技能（如天气，计算预算等)。
可是目前无法通过这种文件夹组织形式，动态地将技能相关工具函数注册给智能体，这个问题仍未解决........

在另一个仓库my-llm_projectB里使用了另一种文件组织形式实现了Agent skills思想（即单智能体渐进式技能加载架构)

