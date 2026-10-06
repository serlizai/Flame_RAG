# Flame 商品问答助手

Flame 使用本地 BGE-M3 和 Chroma 检索商品资料，再通过 LangGraph 与聊天模型生成带来源引用的回答。项目仅保留 LangGraph Agent 问答架构，提供 FastAPI 接口和 Streamlit 聊天界面。

当前完成第二周的商品文档、Metadata、建库与基础检索。三个 vivo X500 文档为**虚构演示数据**，参数、价格和售后规则需要核验后才能用于真实商品问答。

## 1. 安装项目环境

需要 Python 3.11 及以上、[uv](https://docs.astral.sh/uv/) 和已下载的 BGE-M3 模型。完整问答服务还需要 Redis，以及支持工具调用的 OpenAI 兼容聊天服务。

从项目根目录执行：

```bash
cd /Users/lwj/AI_project/flame
uv sync --locked --python 3.11
source .venv/bin/activate
python --version
```

`uv sync` 根据 `pyproject.toml` 和 `uv.lock` 创建独立的 `.venv`，默认包含 pytest、Ruff 等开发依赖。后续使用 `uv run` 会自动选择本项目环境，无需先激活。其他机器请修改上面的项目路径。

编辑器的 Python 解释器应选择：

```text
/Users/lwj/AI_project/flame/.venv/bin/python
```

## 2. 配置环境变量

首次使用时，将 `.env.example` 复制为 `.env`；已有 `.env` 时直接编辑，保留现有配置。

```bash
cp .env.example .env
```

配置示例：

```dotenv
OPENAI_API_KEY=填写聊天服务的密钥
OPENAI_BASE_URL=填写聊天服务的 OpenAI 兼容接口地址
BGE_M3_PATH=/Users/lwj/AI_project/RAG/models/BAAI/bge-m3
EMBEDDING_DEVICE=cpu
CHROMA_PERSIST_DIRECTORY=chroma_db
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
```

`backend/config.py` 中的聊天模型名目前是 `qwen3.8-27b`，需与服务提供方支持的模型名一致。聊天服务必须支持工具调用。其他机器需要修改本地模型路径。

商品 Agent 和商品评测默认读取 `chroma_db_products/` 的 `products` 集合；`CHROMA_PERSIST_DIRECTORY` 仅用于可选的 PDF 导入工具。`.env`、`.venv`、向量库和模型文件均被 Git 忽略。

## 3. 准备商品库

商品资料位于 `data/products/`：

```text
x500.md            → vivo_x500          / SKU 100000
x500_pro.md        → vivo_x500_pro      / SKU 100001
x500_pro_max.md    → vivo_x500_pro_max  / SKU 100002
```

每个文档包含商品信息，以及商品简介、核心参数、核心卖点、使用说明、FAQ、售后说明。先按二级章节拆分，再按最多 800 字符、重叠 100 字符切分；短章节不会强制重叠，不同章节不会拼在一起。

每个 Chunk 保存正文及 `product_id`、`sku_id`、`product_name`、`category`、`source`、`section`、`chunk_index`、`chunk_id`。SKU 使用字符串，保留前导零。

```bash
# 校验文档并预览切分，不加载模型、不写库
uv run python -m scripts.build_product_db --dry-run

# 使用 .env 中配置的本地 BGE-M3 建库
uv run python -m scripts.build_product_db --device cpu

# 可选：单独导出正文和 Metadata，便于查看
uv run python -m scripts.prepare_product_chunks
```

建库直接读取 Markdown，无需先导出 JSONL。当前三个商品生成 7、7、8 个块，共 22 个 1024 维向量。修改文档后重新建库，脚本会更新记录并在新数据写入成功后删除过期的受管理块，保留其他来源记录和集合。

建库和检索测试只需要本地模型，无需聊天 API 或 Redis。若使用 `--persist-directory` 自定义建库路径，需同步调整后端商品库路径。

## 4. 测试商品检索

```bash
# 基础验收：三个问题，不加商品过滤
uv run python -m scripts.test_retrieval --device cpu

# 使用测试用例指定的商品范围过滤
uv run python -m scripts.test_retrieval --filter-products --device cpu
```

也支持直接执行：

```bash
.venv/bin/python scripts/test_retrieval.py --filter-products
```

脚本显示前 5 条结果的商品、章节、来源和正文预览；基础验收全部通过时退出码为 0。单商品问题检查首条结果的商品归属，对比问题检查是否同时召回两个目标商品；章节命中另作诊断。

现有样例基础召回为 3/3。开启过滤后，Pro 的核心卖点和核心参数分别进入第 4、5 条。过滤使用测试用例中的已知商品 ID；正式 API 和 Agent 尚未接入 `product_id`，后端 `retrieve_docs` 仍执行未过滤的检索。

## 5. 启动问答服务

先完成聊天配置和商品建库，再启动 Redis。使用 Docker 时：

```bash
docker run -d --name flame-redis -p 6379:6379 redis:7
docker exec flame-redis redis-cli ping
```

返回 `PONG` 表示 Redis 可访问。若已创建该容器，使用 `docker start flame-redis`；已有 Redis 服务可直接配置 `REDIS_URL`。

启动 FastAPI：

```bash
uv run uvicorn backend.main:app --reload
```

接口文档：<http://127.0.0.1:8000/docs>。在另一个终端发起请求：

```bash
curl -G 'http://127.0.0.1:8000/query' \
  --data-urlencode 'query=X500 Pro 有哪些配置？'
```

接口返回 `answer`、`citations`、`session_id`。引用包含 `number`、`label`、`snippet` 和 `source`。继续同一会话时传回第一次响应中的 `session_id`：

```bash
curl -G 'http://127.0.0.1:8000/query' \
  --data-urlencode 'query=它有哪些核心卖点？' \
  --data-urlencode 'session_id=替换为第一次响应中的 session_id'
```

可选的 Streamlit 界面：

```bash
uv run streamlit run app.py
```

访问 <http://localhost:8501>。Streamlit 直接调用 `agent_invoke()`，它与 FastAPI 是两个入口，可以单独启动；二者都需要聊天模型、商品库和 Redis。

## 6. 完整服务链路

建库在问答前执行：

```text
商品 Markdown → 字段校验 → 按章节切分并继承 Metadata
             → 本地 BGE-M3 生成向量 → Chroma 商品库
```

一次未命中缓存的问答经过以下环节：

```text
用户请求 → FastAPI /query → agent_invoke（分配或复用 session_id）
        → 查询 Redis 回答缓存
        → LangGraph 的 llm_call：聊天模型决定是否调用工具
        → tool_node / retrieve_docs
        → 本地 BGE-M3 将检索问题编码为向量
        → Chroma products 检索前 5 个 Chunk
        → 为文本添加引用编号，并生成 Citation 对象
        → 工具结果返回模型，继续判断是否需要检索
        → final_answer：依据资料生成回答并校验引用编号
        → Redis 保存聊天记录与回答缓存
        → 返回 answer、citations、session_id
```

模型可能调用工具多次，也可能直接进入最终回答；缓存命中时直接返回已保存的回答和引用，跳过模型与检索。Streamlit 从 `agent_invoke` 进入同一后端流程。

“完整链路验收”意味着真实聊天服务、BGE、Chroma、Redis 和接口一起工作：实际调用 `/query` 得到回答及有效引用，再复用会话 ID 追问。单独检索通过、接口文档能打开或离线测试通过，都不足以证明这些环节全部正常。

目前 Agent 工作记忆保存在进程内，重启后不自动从 Redis 聊天记录恢复；回答缓存按问题共享。商品上下文接入时还需完善会话、商品及引用编号的隔离。

## 7. 开发检查与当前状态

```bash
uv run pytest tests/ -q
uv run ruff check .
uv run ruff format --check .

# 仅在真实聊天服务、Redis 和商品库都准备好后运行
RUN_AGENT_INTEGRATION=1 uv run pytest tests/test_agent.py -q
```

最后一条测试调用真实 Agent，普通测试默认跳过它；FastAPI 的 HTTP 请求需另外按第 5 节手动验收。

当前已完成独立项目环境、商品库及真实 BGE 检索。独立 `.venv` 使用 Python 3.11.15，安装 232 个依赖包；112 项测试通过、1 项真实服务测试跳过，Ruff 检查通过。已移除与新版 PyMuPDF 底层库冲突的旧 PyMuPDFb，并同步依赖声明和锁文件。

经典 RAG 入口、对话 Notebook、独立 CrewAI 示例和未使用的旧 schema 已删除；缓存集中在 `backend/cache.py`。`langchain-classic` 已从直接依赖中移除，但仍由 `langchain-community` 作为传递依赖安装，不代表项目仍保留经典问答代码。

本机 `.env` 中的聊天密钥和接口地址仍未填写，完整问答服务尚未验收。下一阶段是接入 API/State 的当前商品上下文及正式检索过滤。

主要文件：

| 文件或目录 | 职责 |
| --- | --- |
| `backend/main.py` | FastAPI 问答入口 |
| `backend/retrieval.py` | Agent 调用、会话、缓存和历史 |
| `backend/graph.py`、`backend/nodes.py` | 模型与工具的工作流 |
| `backend/config.py`、`backend/embeddings.py` | 模型、商品库、Redis 配置与本地 BGE 加载 |
| `backend/tools.py`、`citations.py` | 商品检索工具及引用处理 |
| `backend/product_documents.py` | 商品字段校验和文本切分 |
| `scripts/`、`data/products/` | 建库、检索脚本与商品文档 |
| `app.py` | Streamlit 聊天入口 |
| `backend/cache.py` | Agent 回答缓存和 Redis 聊天历史 |
| `eval/` | 商品集合抽样、Top-5 检索和引用评测 |

详细记录见[开发文档](docs/Flame_Development_Notes.md)、[第二周进度](docs/Week2_Metadata_and_Progress.md)和[圣火计划](圣火文档.md)。贡献规范见[CONTRIBUTING.md](CONTRIBUTING.md)，上游许可证见[LICENSE.md](LICENSE.md)。
