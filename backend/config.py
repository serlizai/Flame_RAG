from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from pathlib import Path
import os

from backend.cache import RedisCache

DATA = Path(__file__).resolve().parent.parent

load_dotenv()

# model = init_chat_model("openai:gpt-5.5", temperature=0)
model = init_chat_model(
    model="qwen3.8-27b",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("BGE_M3_PATH"),
    model_kwargs={
        "device": "mps" or "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    },
)
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=str(DATA / "chroma_db_bge"),
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
cache = RedisCache(REDIS_URL)

memory = MemorySaver()
