
#   环境变量配置文件.env
from dotenv import load_dotenv
import os 
# 强制用 .env 的值覆盖系统环境变量
load_dotenv(override=True)

OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

LANGSWITH_API_KEY = os.getenv('LANGSWITH_API_KEY')

OPENAI_MODEL_NAME= os.getenv('OPENAI_MODEL_NAME')