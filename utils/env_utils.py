import os

from dotenv import load_dotenv
from langchain_community.utilities.tavily_search import TAVILY_API_URL

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

MILVUS_URI = 'http://127.0.0.1:19530'

COLLECTION_NAME = 't_collection01'
