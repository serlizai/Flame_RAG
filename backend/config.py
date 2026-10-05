"""初始化聊天模型、商品向量库、Redis 缓存和图检查点。"""

from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
import os

from backend.cache import RedisCache
from backend.embeddings import create_embeddings
from settings import PROJECT_ROOT

# 模型配置示例：model = init_chat_model("openai:gpt-5.5", temperature=0)
model = init_chat_model(
    model="qwen3.8-27b",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
embeddings = create_embeddings()
# 商品库的目录与集合名必须与 scripts/build_product_db.py 保持一致。
product_vector_store = Chroma(
    collection_name="products",
    embedding_function=embeddings,
    persist_directory=str(PROJECT_ROOT / "chroma_db_products"),
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
cache = RedisCache(REDIS_URL)

memory = MemorySaver()
