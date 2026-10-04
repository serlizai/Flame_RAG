"""初始化聊天模型、文本向量模型、向量库、Redis 缓存和图检查点。"""

from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
import os

from backend.cache import RedisCache
from backend.embeddings import create_embeddings
from settings import CHROMA_PERSIST_DIRECTORY

# 模型配置示例：model = init_chat_model("openai:gpt-5.5", temperature=0)
model = init_chat_model(
    model="qwen3.8-27b",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
embeddings = create_embeddings()
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIRECTORY,
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
cache = RedisCache(REDIS_URL)

memory = MemorySaver()
