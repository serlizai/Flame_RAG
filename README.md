# Flame 商品问答助手

Flame 使用本地 BGE-M3 和 Chroma 检索商品资料，再通过 LangGraph 与聊天模型生成带来源引用的回答。项目仅保留 LangGraph Agent 问答架构，提供 FastAPI 接口和 Streamlit 聊天界面。

当前完成第二周的商品文档、Metadata、建库与基础检索，以及第三周的商品表、API 输入、State 和正式商品过滤。API 指定商品后，可以直接问“这款”，检索仅返回当前商品的资料；各商品分别保存上下文，SQLite 支持重启后续聊。第三周进度见[当前商品上下文记录](docs/Week3_Product_Context.md)，会话实现见[检查点开发记录](docs/Product_Sessions_and_Checkpoints.md)。当前已扩充为 **10 个跨品类演示商品**，包括 3 款手机和 7 款生活用品；参数、价格和售后规则需要核验后才能用于真实商品问答。扩充过程见[十商品开发记录](docs/Ten_Product_Catalog.md)。

第四周已整理商品专用提示词，统一角色、当前商品、检索时机、资料缺失及引用规则，并将正式商品工具改名为 `retrieve_product_docs`。实现与手动测试见[第四周记录](docs/Week4_Product_RAG_MVP.md)。

第五周四个正式接口已接入：商品列表、单商品基础信息、`POST /api/chat` 聊天和 `GET /api/health` 存活检查。商品查询直接读取 JSON，聊天复用现有 Agent，存活检查不访问外部依赖。接口约定和手动测试见[第五周记录](docs/Week5_Formal_API.md)。

## 1. 安装项目环境

需要 Python 3.11 及以上、[uv](https://docs.astral.sh/uv/) 和已下载的 BGE-M3 模型。完整问答服务还需要 Redis，以及支持工具调用的 OpenAI 兼容聊天服务。

本机使用已安装的 Python 3.12，项目的 `.python-version` 已固定为 `3.12`。Windows PowerShell 从项目根目录执行：

```powershell
Set-Location F:\shkj\Flame_RAG
uv sync --locked --python D:\python\python.exe --no-python-downloads
.\.venv\Scripts\python.exe --version
```

本机环境配置及剩余步骤见 [Windows 配置说明](docs/Windows_Setup.md)。其他机器按实际路径选择已安装的 Python 3.12，例如：

```bash
cd /Users/lwj/AI_project/flame
uv sync --locked --python 3.12 --no-python-downloads
source .venv/bin/activate
python --version
```

`uv sync` 根据 `pyproject.toml` 和 `uv.lock` 创建独立的 `.venv`，默认包含 pytest、Ruff 等开发依赖。后续使用 `uv run` 会自动选择本项目环境，无需先激活。其他机器请修改上面的项目路径。

本机编辑器的 Python 解释器应选择：

```text
F:\shkj\Flame_RAG\.venv\Scripts\python.exe
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
OPENAI_MODEL=填写聊天服务支持的模型名
BGE_M3_PATH=/Users/lwj/AI_project/RAG/models/BAAI/bge-m3
EMBEDDING_DEVICE=cpu
CHROMA_PERSIST_DIRECTORY=chroma_db
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
CHECKPOINT_SQLITE_PATH=data/checkpoints.sqlite
```

通过 `.env` 中的 `OPENAI_MODEL` 设置聊天模型名，未设置时默认使用 `qwen3.8-27b`，需与服务提供方支持的模型名一致。聊天服务必须支持工具调用。其他机器需要修改本地模型路径。

商品 Agent 和商品评测默认读取 `chroma_db_products/` 的 `products` 集合；`CHROMA_PERSIST_DIRECTORY` 仅用于可选的 PDF 导入工具。`.env`、`.venv`、向量库和模型文件均被 Git 忽略。

## 3. 准备商品库

商品资料位于 `data/products/`：

```text
x500.md             → vivo_x500                 / SKU 100000
x500_pro.md         → vivo_x500_pro             / SKU 100001
x500_pro_max.md     → vivo_x500_pro_max         / SKU 100002
kettle_k1.md        → flame_kettle_k1           / SKU 200001
air_fryer_a1.md     → flame_air_fryer_a1        / SKU 200002
rice_cooker_r1.md   → flame_rice_cooker_r1      / SKU 200003
desk_lamp_l1.md     → flame_desk_lamp_l1        / SKU 200004
circulation_fan_f1.md → flame_circulation_fan_f1 / SKU 200005
thermos_t1.md       → flame_thermos_t1          / SKU 200006
storage_box_s1.md   → flame_storage_box_s1      / SKU 200007
```

商品列表保存在 `data/products.json`，使用 `id`、`sku`、`name`、`category` 四个字段，与上述商品文档及 Chunk Metadata 一致。API 使用商品表校验传入的商品 ID，State 和检索过滤使用同一身份。每个 Markdown 头部增加 `category:`，分类覆盖 `smartphone`、`kitchen_appliance`、`lighting`、`home_appliance`、`drinkware`、`home_storage`。默认逐文档读取分类；`--category` 仅在需要统一覆盖所有商品分类时使用。旧文档和旧商品表省略分类时兼容 `smartphone`。

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

建库直接读取 Markdown，无需先导出 JSONL。当前 10 个商品共生成 **71 个 1024 维向量**，Pro Max 为 8 个块，其余商品各 7 个。修改文档后重新建库，脚本会更新记录并在新数据写入成功后删除过期的受管理块，保留其他来源记录和集合。

建库和检索测试只需要本地模型，无需聊天 API 或 Redis。若使用 `--persist-directory` 自定义建库路径，需同步调整后端商品库路径。

## 4. 测试商品检索

```bash
# 基础验收：11 个问题覆盖 10 个商品，不加商品过滤
uv run python -m scripts.test_retrieval --device cpu

# 使用测试用例指定的商品范围过滤
uv run python -m scripts.test_retrieval --filter-products --device cpu
```

也支持直接执行：

```bash
.venv/bin/python scripts/test_retrieval.py --filter-products
```

脚本显示前 5 条结果的商品、章节、来源和正文预览；基础验收全部通过时退出码为 0。单商品问题检查首条结果的商品归属，对比问题检查是否同时召回两个目标商品；章节命中另作诊断。

现有样例在全库和商品过滤两种模式下均为 **11/11 通过**；章节命中仍单独诊断。脚本过滤使用测试用例中的已知商品 ID；正式后端 `retrieve_product_docs` 则使用 State 中的 `product_id` 过滤。

## 5. 启动问答服务

商品列表和基础信息接口只需要商品表和 FastAPI，可直接启动服务访问。完整问答需先完成聊天配置和商品建库，再启动 Redis。使用 Docker 时：

```bash
docker run -d --name flame-redis -p 6379:6379 redis:7
docker exec flame-redis redis-cli ping
```

返回 `PONG` 表示 Redis 可访问。若已创建该容器，使用 `docker start flame-redis`；已有 Redis 服务可直接配置 `REDIS_URL`。

Redis 启用密码时，在本地 `.env` 中配置 `REDIS_URL=redis://:你的密码@localhost:6379/0`；密码中的特殊字符需 URL 编码。Agent 读取根目录配置，经典版的独立 Redis 配置不会自动用于 Agent。真实密码只保存在被 Git 忽略的本地 `.env` 中。

启动 FastAPI：

```bash
uv run uvicorn backend.main:app --reload
```

接口文档：<http://127.0.0.1:8000/docs>。在另一个终端先检查 HTTP 服务是否存活：

```bash
curl 'http://127.0.0.1:8000/api/health'
```

预期返回 `{"status":"ok","service":"flame"}`。该响应只表示 HTTP 服务可用，不探测模型、Redis 或知识库。获取商品列表：

```bash
curl 'http://127.0.0.1:8000/api/products'
```

返回 JSON 数组，每项包含 `id`、`sku`、`name`、`category`，SKU 为字符串。查询其中一个商品的基础信息：

```bash
curl 'http://127.0.0.1:8000/api/products/flame_kettle_k1'
```

单商品接口返回 JSON 对象，字段与列表项一致；商品 ID 不存在返回 404，空白 ID 返回 422，商品表配置错误返回 500。它只读取基础信息，参数及 FAQ 继续通过问答检索资料。将选中项的 `id` 作为问答请求的 `product_id`，使用正式聊天接口：

```bash
curl 'http://127.0.0.1:8000/api/chat' \
  -H 'Content-Type: application/json' \
  --data '{"product_id":"vivo_x500_pro","message":"这款支持无线充电吗？","session_id":"my-session"}'
```

接口返回 `answer`、`citations`、`session_id`、`product_id`。引用包含 `number`、`label`、`snippet` 和 `source`。继续同一会话时传回第一次响应中的 `session_id`：

```bash
curl 'http://127.0.0.1:8000/api/chat' \
  -H 'Content-Type: application/json' \
  --data '{"product_id":"vivo_x500_pro","message":"它有哪些核心卖点？","session_id":"my-session"}'
```

`product_id` 和 `session_id` 可省略或传 `null`；未指定会话时自动生成编号。原 `GET /query` 和 `POST /query` 继续兼容，例如：

```bash
curl -G 'http://127.0.0.1:8000/query' \
  --data-urlencode 'query=这款支持无线充电吗？' \
  --data-urlencode 'product_id=vivo_x500_pro' \
  --data-urlencode 'session_id=my-session'
```

新旧聊天入口共用会话，使用相同 `session_id` 和 `product_id` 即可续聊。不存在的商品返回 404，空问题及非法字段返回 422。每轮传入当前商品 ID；省略或传 `null` 表示独立的全库模式。指定 Pro 时工具只搜索 Pro，引用来源应为 `x500_pro.md`；改成 `vivo_x500`，应回答不支持无线充电且仅引用 `x500.md`。同会话的不同商品各有独立上下文，切回 Pro 会恢复之前的 Pro 对话，服务重启后也能续聊。

可选的 Streamlit 界面：

```bash
uv run streamlit run app.py
```

访问 <http://localhost:8501>。Streamlit 直接调用 `agent_invoke()`，它与 FastAPI 是两个入口，可以单独启动；二者都需要聊天模型、商品库和 Redis。目前该页面没有商品选择控件，默认全库问答，当前商品功能通过 API 验证。

## 6. 完整服务链路

建库在问答前执行：

```text
商品 Markdown → 字段校验 → 按章节切分并继承 Metadata
             → 本地 BGE-M3 生成向量 → Chroma 商品库
```

一次未命中缓存的问答经过以下环节：

```text
用户请求 → FastAPI POST /api/chat → agent_invoke（分配或复用 session_id）
        → 按 session_id 与 product_id 选择 thread_id，读取 SQLite 图状态
        → 无历史的首轮可使用 Redis 答案缓存，有历史则跳过缓存
        → LangGraph 的 llm_call：聊天模型决定是否调用工具
        → tool_node / retrieve_product_docs
        → 本地 BGE-M3 将检索问题编码为向量
        → Chroma products 按当前商品过滤，检索前 5 个 Chunk（无商品时全库）
        → 为文本添加引用编号，并生成 Citation 对象
        → 工具结果返回模型，继续判断是否需要检索
        → final_answer：依据资料生成回答并校验引用编号
        → SQLite 自动保存图状态；Redis 保存聊天记录，首轮另存答案缓存
        → 返回 answer、citations、session_id
```

模型可能调用工具多次，也可能直接进入最终回答；缓存命中时恢复回答和引用，并同步更新图状态中的本轮消息和商品 ID，跳过模型调用与检索。Streamlit 从 `agent_invoke` 进入同一后端流程。

“完整链路验收”意味着真实聊天服务、BGE、Chroma、Redis 和接口一起工作：实际调用 `/api/chat` 得到回答及有效引用，再复用会话 ID 追问。单独检索通过、健康检查返回 200、接口文档能打开或离线测试通过，都不足以证明这些环节全部正常。

图状态默认写入 `data/checkpoints.sqlite`，相对路径按项目根目录解析，可通过 `CHECKPOINT_SQLITE_PATH` 修改。前端会话 ID 保持不变，后端按会话与商品生成内部线程；Redis 汇总页面问答历史，SQLite 恢复模型上下文。答案缓存仅用于没有历史的首轮，追问既不读取也不覆盖共享缓存。完整切换与重启测试见[会话开发记录](docs/Product_Sessions_and_Checkpoints.md)。

## 7. 开发检查与当前状态

```bash
uv run pytest tests/ -q
uv run ruff check .
uv run ruff format --check .

# 仅在真实聊天服务、Redis 和商品库都准备好后运行
RUN_AGENT_INTEGRATION=1 uv run pytest tests/test_agent.py -q

# 正式接口真实 TCP 验收：临时服务、续聊、商品切换和缓存
RUN_AGENT_INTEGRATION=1 uv run pytest tests/test_formal_api_integration.py -q -s
```

两条真实测试命令共六项测试。前者覆盖 Agent 问答、商品状态、过滤、提示词和生活用品路由；后者在随机端口启动真实 Uvicorn，验证正式接口、新旧入口共用会话、商品切换和首轮缓存，使用独立临时检查点文件，结束后关闭测试服务并清理随机 Redis 数据。普通测试默认跳过它们。

当前已完成独立项目环境、商品库及真实 BGE 检索。独立 `.venv` 使用 Python 3.11.15；254 项离线测试通过、6 项真实服务测试默认跳过，修改代码的 Ruff 检查通过。已移除与新版 PyMuPDF 底层库冲突的旧 PyMuPDFb，并同步依赖声明和锁文件。

经典 RAG 入口、对话 Notebook、独立 CrewAI 示例和未使用的旧 schema 已删除；缓存集中在 `backend/cache.py`。`langchain-classic` 已从直接依赖中移除，但仍由 `langchain-community` 作为传递依赖安装，不代表项目仍保留经典问答代码。

2026-10-13 已补齐本机 Redis 认证并完成真实链路验收：真实 Agent 测试 1 项通过，确认调用 `retrieve_docs`；FastAPI `/query` 的首次问答、同会话追问和重复请求均返回 200。回答与引用符合演示资料中的 50W 无线充电和 16GB + 512GB 配置，Redis 历史及缓存检查通过。详细记录见[开发文档第 10 节](docs/Flame_Development_Notes.md#10-agent-真实链路验收2026-10-13)。下一阶段是接入 API/State 的当前商品上下文及正式检索过滤。

2026-10-14 已完成第三周第二步：JSON 问答输入、商品 ID 校验和透传、商品缓存区分。回归测试为 145 项通过、1 项真实测试默认跳过；真实 POST、兼容 GET、404 和 422 均验收通过。下一步将商品 ID 加入 State。

2026-10-15 已完成第三周第三步：商品 ID 写入 State，缓存命中也保存本轮图状态。8 项状态测试通过，新增真实商品状态测试 1 项通过。实现说明和手动测试命令见[第三周记录](docs/Week3_Product_Context.md#第三步把当前商品写入-state已完成)。下一步接入正式商品检索过滤。

2026-10-15 已完成第三周第四步：State 商品 ID 注入检索工具，Chroma 正式按商品过滤；模型接收当前商品名称，切换商品时隔离旧上下文，连续检索接续引用编号，商品缓存升级 v2。新增真实过滤测试 1 项通过，确认不写型号也能按 Pro、普通版、Max 分别回答且引用只来自当前商品；真实 HTTP 的 Pro → 普通版 → 缓存返回 Pro 三次请求均为 200，Redis 历史和缓存验收通过。开发过程及手动验收见[第四步记录](docs/Week3_Product_Context.md#第四步正式检索商品过滤已完成)。

2026-10-15 进一步完成商品独立上下文与 SQLite 持久化：按会话和商品生成线程，切回商品恢复其原对话；检查点默认保存到 `data/checkpoints.sqlite`，替换了先前切换后重新开始的方案。答案缓存仅用于无历史的首轮，追问跳过缓存。172 项离线测试、三项真实 Agent 测试通过；实际 FastAPI 重启后，Pro 和普通版各自恢复了原来的备注，五次 HTTP 请求全部为 200。配置、文件职责及手动重启续聊步骤见[开发记录](docs/Product_Sessions_and_Checkpoints.md)。

2026-10-16 完成第四周第一步：集中维护中文商品提示词，统一两阶段的当前商品、检索、缺失资料及引用规则，答案缓存加入提示词版本。183 项离线测试通过，新增真实提示词验收 1 项通过，确认已知参数有引用、未提供的 L2 容量不猜测且不附引用、身份和用户备注按角色及历史回答。工具改名留给下一步，详见[第四周记录](docs/Week4_Product_RAG_MVP.md)。

2026-10-16 完成第四周第二步：正式工具更名为 `retrieve_product_docs`，同步描述、绑定、路由和提示词，版本升级到 `product_prompt_v2`；保留旧 SQLite 会话兼容。188 项离线测试通过，两项真实测试确认新工具调用、三个商品过滤、切回续聊、缺资料说明及备注记忆均正常，完成本周商品 RAG V0.1 后端验收。实现及手动命令见[第二步记录](docs/Week4_Product_RAG_MVP.md#第二步正式商品工具名称与描述已完成)。

2026-10-16 扩充至十商品：新增七款生活用品，逐商品读取分类，JSON、Markdown、JSONL 和真实 Chroma 已同步为 10 个商品、71 个 Chunk。206 项离线测试通过，全库和过滤召回均为 11/11；新增真实生活用品 HTTP 测试通过，三次请求均返回正确参数及对应商品引用，详见[扩充记录](docs/Ten_Product_Catalog.md)。

2026-10-16 完成第五周第一步：新增正式商品列表 `GET /api/products`，返回十个商品的 ID、SKU、名称和分类；与问答校验共用商品表错误处理。7 项新增测试覆盖响应、配置错误、数据更新、OpenAPI 和新进程轻量读取。213 项离线测试通过；临时 Uvicorn 服务的真实 HTTP 请求返回 200，模型与 Redis 不可用时仍能获取列表，详见[第五周记录](docs/Week5_Formal_API.md)。

2026-10-16 完成第五周第二步：新增 `GET /api/products/{product_id}`，返回单商品基础信息，与列表共用响应模型、与问答校验共用商品查找逻辑。229 项离线测试通过，临时服务真实 HTTP 验证十个商品详情均为 200，不存在和空白 ID 分别为 404、422。下一步是正式聊天接口 `POST /api/chat`，详见[第二步记录](docs/Week5_Formal_API.md#第二步单商品基础信息已完成)。

2026-10-16 完成第五周第三、四步：新增正式 `POST /api/chat` 和轻量 `GET /api/health`，聊天入口复用原有校验、会话、过滤、引用和缓存。254 项离线测试通过；新增真实 Uvicorn TCP 测试 1 项通过，确认四个正式接口、404/422、新旧入口续聊、商品切换及模型调用为 0 的缓存命中。测试服务和随机 Redis 数据已清理，完成第五周接口验收，详见[第五周记录](docs/Week5_Formal_API.md)。

主要文件：

| 文件或目录 | 职责 |
| --- | --- |
| `backend/main.py` | FastAPI 商品列表、基础信息、正式聊天和存活检查 |
| `backend/products.py`、`data/products.json` | 商品身份读取和 API 商品校验 |
| `backend/retrieval.py` | Agent 调用、会话、缓存和历史 |
| `backend/sessions.py`、`backend/checkpoints.py` | 商品线程标识与 SQLite 检查点 |
| `backend/graph.py`、`backend/nodes.py` | 模型与工具的工作流 |
| `backend/config.py`、`backend/embeddings.py` | 模型、商品库、Redis 配置与本地 BGE 加载 |
| `backend/tools.py`、`citations.py` | 商品检索工具及引用处理 |
| `prompts.py`、`backend/prompts.py` | 商品系统提示词、回答模板及后端统一导出 |
| `backend/product_documents.py` | 商品字段校验和文本切分 |
| `scripts/`、`data/products/` | 建库、检索脚本与商品文档 |
| `app.py` | Streamlit 聊天入口 |
| `backend/cache.py` | Agent 回答缓存和 Redis 聊天历史 |
| `eval/` | 商品集合抽样、Top-5 检索和引用评测 |

详细记录见[开发文档](docs/Flame_Development_Notes.md)、[第二周进度](docs/Week2_Metadata_and_Progress.md)和[圣火计划](圣火文档.md)。贡献规范见[CONTRIBUTING.md](CONTRIBUTING.md)，上游许可证见[LICENSE.md](LICENSE.md)。
