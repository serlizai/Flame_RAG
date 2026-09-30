from pathlib import Path
from urllib.request import urlretrieve
import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PDF_PATH = DATA_DIR / "constitution_india.pdf"

DB_PATH = BASE_DIR / "chroma_db_bge"


# LawGlance 原项目使用的印度宪法 PDF
PDF_URL = (
    "https://cdnbbsr.s3waas.gov.in/"
    "s380537a945c7aaa788ccfcdf1b99b5d8f/"
    "uploads/2024/07/20240716890312078.pdf"
)


# =========================
# 下载 PDF
# =========================

if not PDF_PATH.exists():
    print("正在下载 PDF...")
    urlretrieve(PDF_URL, PDF_PATH)


# =========================
# 加载 PDF
# =========================

print("正在读取 PDF...")

loader = PyMuPDFLoader(str(PDF_PATH))

documents = loader.load()


# 补充 citation 所需要的一些 metadata
for doc in documents:
    doc.metadata["country"] = "India"
    doc.metadata["db_owner"] = "LawGlance"
    doc.metadata["source_name"] = "Indian Constitution"
    doc.metadata["source"] = PDF_URL


# =========================
# 文档切块
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(documents)

print(f"生成 {len(chunks)} 个 chunks")


# =========================
# BGE
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("BGE_M3_PATH"),
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    },
)


# =========================
# 清理旧测试数据库
# =========================

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)


# =========================
# 写入 Chroma
# =========================

print("开始生成向量数据库...")

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(DB_PATH),
)

print()
print("================================")
print("BGE Chroma 数据库创建完成")
print(f"路径：{DB_PATH}")
print("================================")