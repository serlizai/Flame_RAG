# Flame 开发记录：项目整理与商品知识库

更新日期：2026-10-15。本文汇总已完成的修改，对应[圣火文档](../圣火文档.md)第 1～3 周的开发步骤。

当前已完成项目命名清理、三个商品文档、Chunk Metadata、商品建库脚本、后端商品库配置、基础检索验收、中文注释、独立 `.venv` 和中文 README。已使用本地 BGE-M3 建成商品库：**22 个文本块，每个向量 1024 维**。三个问题的基础商品召回通过，真实 Agent 和 FastAPI 问答、追问、缓存链路已验收；章节召回及更广泛的回答质量仍需完善。安装与完整服务流程见 [README](../README.md)。

项目现仅保留 `backend/` 中的 LangGraph Agent 问答架构；经典 RAG、独立 CrewAI 示例及旧 schema 已移除，具体清理范围见第 9 节。

## 1. 项目命名与法律内容清理

项目从 LawGlance 整理为 Flame，主要变化如下：

| 范围 | 修改结果 |
| --- | --- |
| 项目与界面 | `pyproject.toml`、`uv.lock` 和 Streamlit 界面使用 Flame 名称，移除法律主题介绍、链接和图片 |
| 问答架构 | 仅保留 LangGraph Agent；原经典 RAG 的 `flame_main.py` 和 `chains.py` 已删除 |
| 提示词 | `prompts.py` 和 `config/prompts.yaml` 改为通用文档问答，要求依据资料回答并提供编号引用 |
| 引用 | `citations.py` 支持通用文档标题、章节和页码；新增商品名支持，可显示“商品名 + 章节” |
| 缓存 | Redis 回答缓存和聊天历史使用 `flame:` 键前缀，避免混用旧项目缓存 |
| PDF 建库 | 使用 `scripts/build_bge_db.py` 导入指定的本地 PDF，移除固定法律资料下载和清空目录逻辑 |
| 评测与测试 | 法律专用样例改为通用文档或商品样例；检索评测使用共享的本地 BGE；旧法律 QA 数据清空，等待补充新参考集 |
| 文档与 Notebook | 项目文档为 `docs/Flame_Documentation.md`；仅保留 `src/pdf_ingestion.ipynb` 建库辅助示例 |

法律主题 Logo 和视频缩略图已删除。`LICENSE.md` 保留原许可证和上游版权信息；旧本地向量库保留，未迁移为商品库。LangGraph 节点与工具循环沿用现有结构。

## 2. 商品文档

新增目录与文件：

```text
data/products/
├── x500.md
├── x500_pro.md
└── x500_pro_max.md
```

| 文档 | 商品名称 | product_id | SKU | 当前块数 |
| --- | --- | --- | --- | --- |
| `x500.md` | vivo X500 | `vivo_x500` | `100000` | 7 |
| `x500_pro.md` | vivo X500 Pro | `vivo_x500_pro` | `100001` | 7 |
| `x500_pro_max.md` | vivo X500 Pro Max | `vivo_x500_pro_max` | `100002` | 8 |

每个文件包含商品名称、SKU、product_id、品牌、型号，以及六个统一章节：商品简介、核心参数、核心卖点、使用说明、FAQ、售后说明。

**当前参数、价格和售后规则均为虚构演示内容**，各文档已注明。正式使用前需要替换为经过核验的商品资料；一个 SKU 对应文档中明确的一种商品配置。

## 3. 每个文本块保存 Metadata

新增 `backend/product_documents.py`，负责读取商品信息、校验和切分。每个块同时保存正文与所属商品信息：

```json
{
  "product_id": "vivo_x500_pro",
  "sku_id": "100001",
  "product_name": "vivo X500 Pro",
  "category": "smartphone",
  "source": "x500_pro.md",
  "section": "核心参数",
  "chunk_index": 2,
  "chunk_id": "由商品信息、块序号和正文生成的 SHA-256"
}
```

上面的 `chunk_id` 是说明性占位，实际值为 64 位十六进制字符串。

| 字段 | 来源或用途 |
| --- | --- |
| `product_id`、`sku_id` | 从文档头部读取，用于识别商品和配置；SKU 保持字符串，保留前导零 |
| `product_name` | 从一级标题读取，用于展示和引用 |
| `category` | 默认 `smartphone`，可通过命令参数修改 |
| `source`、`section` | 文件名与章节名，用于定位原文；头部信息块的章节为“商品信息” |
| `chunk_index`、`chunk_id` | 商品内从 0 开始的序号与稳定标识，用于重复建库和更新 |

五个业务字段 `product_id`、`sku_id`、`product_name`、`category`、`source` 均为非空字符串，切分后完整继承。解析器会拒绝空目录、缺失或重复的头部字段、重复商品 ID/SKU、非法切分参数。

切分先按二级标题分章节，再使用 `RecursiveCharacterTextSplitter`，默认最大长度 **800 字符**、重叠 **100 字符**。不同章节独立切分；短章节不会强制补足长度或制造重叠。内容或切分设置变化时，Chunk ID 也可能变化。

## 4. 建库流程与更新规则

新增两个脚本，复用同一套商品解析和切分逻辑：

| 脚本 | 用途 | 输出 |
| --- | --- | --- |
| `scripts/prepare_product_chunks.py` | 导出文本和 Metadata，不加载模型 | `data/product_chunks.jsonl` |
| `scripts/build_product_db.py` | 生成 BGE 向量并同步商品库 | `chroma_db_products/`，集合名 `products` |
| `scripts/test_retrieval.py` | 用真实 BGE 验证三个问题的商品召回 | 控制台明细、通过结果和退出码 |

```text
商品 Markdown → 字段校验 → Document → 按章节及 800/100 切分
                                             ↓
                                本地 BGE-M3 → Chroma 商品库
```

建库脚本直接读取 Markdown，不依赖先生成 JSONL。写库时额外保存 `ingestion_pipeline=flame_product_markdown_v1`，标记本脚本管理的记录。

重复建库按 Chunk ID 写入或更新，成功后删除本脚本生成的过期块。因此修改文档或移除商品文件后，重新建库即可同步数据。其他来源的记录、其他集合和目录内无关文件保留。空目录会报错，不能用于清空商品库。

校验在加载模型和写库前完成；向量生成或新块写入失败时，不执行旧块删除。整个同步过程不是数据库事务，异常中断后应重新运行并核对结果。

## 5. 配置与常用命令

`settings.py` 统一加载项目根目录 `.env`，并解析通用数据库路径。`backend/embeddings.py` 为建库、检索和评测提供统一的本地模型加载方式，生成归一化向量。

当前本地配置：

```dotenv
BGE_M3_PATH=/Users/lwj/AI_project/RAG/models/BAAI/bge-m3
EMBEDDING_DEVICE=cpu
CHROMA_PERSIST_DIRECTORY=chroma_db
```

模型目录已确认存在，真实商品建库使用 CPU。`.env.example` 保留通用占位路径，便于其他机器配置。聊天接口另需 `OPENAI_API_KEY`、`OPENAI_BASE_URL`；Redis 使用 `REDIS_URL` 和 `CACHE_TTL`。**商品建库只需要本地模型，不依赖聊天 API 或 Redis。**

以下命令从项目根目录执行；推荐使用 Python 3.11 及以上和 `uv`：

```bash
# 建立本项目依赖环境
uv sync --locked --python 3.11

# 导出文本块及 Metadata
uv run python -m scripts.prepare_product_chunks

# 只校验和预览，不加载模型、不写数据库
uv run python -m scripts.build_product_db --dry-run

# 使用 .env 中配置的模型建库
uv run python -m scripts.build_product_db --device cpu

# 第 5 步：运行三个问题的基础检索验收
uv run python -m scripts.test_retrieval --device cpu

# 演示按测试用例指定的商品范围过滤
uv run python -m scripts.test_retrieval --filter-products --device cpu
```

可选参数：

| 参数 | 默认值或作用 |
| --- | --- |
| `--products-directory` | 输入目录，默认 `data/products` |
| `--persist-directory` | 商品数据库目录，默认 `chroma_db_products` |
| `--model-path` | 覆盖 `BGE_M3_PATH` |
| `--device` | 覆盖 `EMBEDDING_DEVICE`，未配置时使用 `cpu` |
| `--category` | 默认 `smartphone` |
| `--chunk-size`、`--chunk-overlap` | 默认 `800`、`100` |

商品脚本的相对数据路径及显式传入的模型路径均从项目根目录解析。JSONL 导出脚本可用 `--output` 指定输出文件。

第 2 周第 4 步已完成，`backend/config.py` 中的配置为：

```python
product_vector_store = Chroma(
    collection_name="products",
    embedding_function=embeddings,
    persist_directory=str(PROJECT_ROOT / "chroma_db_products"),
)
```

`backend/tools.py` 中的 `retrieve_docs` 已改为调用 `product_vector_store.similarity_search(query, k=5)`，保留编号引用。Streamlit、FastAPI 和商品评测均读取商品集合；`CHROMA_PERSIST_DIRECTORY` 仅用于可选 PDF 导入工具。若建库时使用自定义目录，需同步修改后端商品库路径。

第 5 步新增 `scripts/test_retrieval.py`，不依赖聊天模型或 Redis。单商品问题检查首条结果的 `product_id`，对比问题检查前 5 条是否包含两个商品；全部通过退出码为 0，否则为 1。当前三个问题均通过，但卖点和配置问题没有命中专门章节；该诊断单独显示，不计入基础商品归属验收。详细召回记录见[第二周进度](Week2_Metadata_and_Progress.md)。

检索脚本也支持直接运行文件。直接执行时，根据 `__file__` 将项目根目录加入导入路径，解决 `settings` 模块找不到的问题；传入脚本绝对路径时可从任意工作目录运行。`tests/test_retrieval_cli.py` 使用独立进程验证两种入口，不依赖外部 `PYTHONPATH`，不加载真实模型。

可选参数 `--filter-products` 按测试用例中的候选商品过滤：单商品使用 `filter={"product_id": "vivo_x500_pro"}`，对比使用 `$in` 同时允许 Pro 和 Pro Max。已有 Metadata 可直接筛选，无需重建库。开启后真实检索 3/3 通过，Pro 核心卖点和核心参数分别进入第 4、5 条；结果没有混入范围外商品。该开关只用于检索测试，正式 API/State 的商品上下文仍待接入。

## 6. 中文注释与版本管理

现有 Python 文件的注释及模块、类、函数 docstring 使用中文，保留的 PDF 建库 Notebook 说明也为中文。前期中文化过程中对比了移除 docstring 后的 AST，确认执行逻辑保持一致；代码标识符和运行时提示词保留原有形式。

`.gitignore` 已允许 `data/products/*.md` 纳入版本控制；`.env`、模型、向量库、生成的 JSONL、缓存和虚拟环境继续忽略。提交范围包括源码、测试、配置模板、依赖锁定文件、商品样例和开发文档；本地商品库可通过建库脚本重新生成。

## 7. 验证结果

| 检查 | 结果 |
| --- | --- |
| 独立项目环境 | `.venv` 使用 Python 3.11.15，232 个依赖包；`uv sync --locked` 和依赖一致性检查通过 |
| 自动化测试 | 在 Flame 独立 `.venv` 中，112 项通过；1 项真实服务测试按配置跳过 |
| 代码检查 | Ruff 静态检查、格式检查和 `git diff --check` 通过 |
| Notebook | 保留的 PDF 建库 Notebook 格式及代码语法检查通过 |
| 真实模型建库 | 使用指定目录的 BGE-M3 成功生成并保存 22 个向量 |
| 数据回读 | 重新打开商品库，核对唯一 ID、完整 Metadata、7/7/8 分布和 1024 维向量 |
| 后端商品库接入 | 配置回读现有 22 个块；离线测试验证检索工具读取商品集合并返回引用 |
| 商品评测接入 | 真实 Chroma 与离线向量测试确认评测读取 `products` 集合、召回 5 条并排除默认集合的其他语料 |
| 真实 BGE 检索 | 默认 k=5、不加商品过滤，基础商品召回 3/3 通过；卖点和参数章节未进入前 5 条 |
| 商品过滤演示 | 开启 `--filter-products` 后 3/3 通过；核心卖点第 4、核心参数第 5，结果只含指定商品 |

测试覆盖商品字段校验、切分继承、稳定 ID、JSONL 导出、商品引用标签，以及重复建库、文档更新、文件移除、保留非管理记录和失败时不删除旧数据。`tests/test_backend_product_store.py` 验证配置与工具能定位商品集合、保留引用，并排除其他语料。数据库与工具测试使用真实 Chroma 和替代向量模型；本地商品库及独立检索验收另行使用真实 BGE。2026-10-13 的完整聊天链路验收见第 10 节；章节召回与更广泛的回答质量仍需继续完善。

```bash
uv run pytest tests/ -q
uv run ruff check .
uv run ruff format --check .

# 配置聊天服务、Redis 和语料库后，才运行真实 Agent 测试
RUN_AGENT_INTEGRATION=1 uv run pytest tests/test_agent.py -q
```

直接执行 `python tests/test_agent.py` 不会运行 pytest 测试。前期验证使用本机已有的 Python 环境；现已建立 Flame 独立的 `.venv`，使用 Python 3.11.15，后续命令使用本项目解释器或 `uv run`。

新环境安装时发现 `PyMuPDFb==1.24.10` 与 `PyMuPDF==1.28.2` 共用底层库文件，导致 `pymupdf` 导入时出现 `Symbol not found`。已从 `pyproject.toml` 和 `requirements.txt` 移除旧 `PyMuPDFb`，同步更新 `uv.lock`；新版 PyMuPDF 自带底层库。README 补齐环境、配置、建库、检索、Redis、FastAPI、Streamlit 和完整服务链路说明。

## 8. 当前待办与下一步

按依赖顺序继续：

1. 第 1 周核心链路及第三周当前商品后端链路已验收。下一步按第四周计划整理商品提示词及工具命名。
2. 继续改善核心卖点、核心参数的章节召回，扩充检索评测问题；正式业务检索已按当前商品 Metadata 过滤。
3. 核验或替换演示商品资料，补充人工审核的评测 QA，后续按开发阶段提交改动。
4. 第四周继续完善商品专用提示词，将检索工具改名为 `retrieve_product_docs`，统一缺少可靠资料时的回答行为。

后续改进已实现商品独立线程和 SQLite 检查点，切回商品及服务重启后可恢复上下文。共享答案缓存仅供无历史的首轮使用，追问跳过缓存。长对话裁剪、正式商品对比模式及用户身份校验仍待后续开发。

更详细的阶段检查见[第二周 Metadata 与进度记录](Week2_Metadata_and_Progress.md)；通用运行说明见[项目说明](Flame_Documentation.md)和[贡献指南](../CONTRIBUTING.md)。

## 9. 移除经典 RAG，仅保留 Agent

删除以下文件：

| 文件 | 删除原因 |
| --- | --- |
| `flame_main.py`、`chains.py` | 经典 RAG 对话类及固定问答链路 |
| `test.ipynb` | 依赖经典 RAG 的对话示例 |
| `cache.py` | 原经典缓存基类；Agent 所需功能已集中到 `backend/cache.py` |
| `backend/schema.py` | 无调用方的旧结构化输出定义，避免与实际 `Citation` 混淆 |
| `examples/flame_crewai.ipynb`、`examples/requirements.txt` | 独立 CrewAI 架构示例及其专用依赖 |

Streamlit 与 FastAPI 均通过 `backend.retrieval.agent_invoke()` 调用 LangGraph。
缓存键、过期时间和 Redis 聊天历史前缀保持原 Agent 的行为。
`citations.py`、`prompts.py` 和后端导入入口继续提供实际使用的引用与提示词。
PDF 导入、BGE 模型加载、商品建库和评测工具作为辅助功能保留。

商品评测默认读取 `chroma_db_products` 的 `products` 集合，检索方式改为
`similarity`、`k=5`，与当前 Agent 一致；缺少集合时报错，不自动创建空集合。
评测 QA 集仍待人工补充，中文 Markdown 不适用原有英文段落抽样规则。

已从 `pyproject.toml`、`requirements.txt` 移除直接声明的 `langchain-classic`，
并更新 `uv.lock`。它仍由 `langchain-community` 作为传递依赖安装，后者用于
Redis 历史及 PDF 导入；应用源码已不再导入 `langchain_classic`。

## 10. Agent 真实链路验收（2026-10-13）

本次修正根目录 `.env` 中的 Redis 认证。密码只写入本地被 Git 忽略的文件，其他聊天和向量模型配置保留原值。运行使用真实 `qwen3.8-27b`、本地 BGE-M3（MPS）、现有 `products` 集合和本机 Redis。

增强 `tests/test_agent.py`：随机问题保证本次请求不复用旧答案，独立会话隔离历史；除检查回答，还检查 LangGraph 状态中实际存在 `retrieve_docs` 的工具结果、引用原文来自商品库、Redis 保存两条消息和答案缓存。结束后只删除本次测试的历史和缓存。

真实测试命令：

```bash
RUN_AGENT_INTEGRATION=1 .venv/bin/python -m pytest tests/test_agent.py -q -s
```

结果为 **1 项通过**，实际执行检索工具 1 次，回答 Pro 支持最高 50W 无线充电，返回 `x500_pro.md` 的 1 条有效引用。代码规范和测试文件格式检查通过。

FastAPI 启动命令：

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

通过真实 HTTP 请求 `GET /query`，检查结果如下：

| 请求 | 结果 |
| --- | --- |
| Pro 是否支持无线充电、最高功率是多少 | HTTP 200；回答最高 50W，返回 1 条引用 |
| 同一会话追问“它的内存和存储容量是多少” | HTTP 200；回答 16GB + 512GB，返回 2 条引用，会话编号保持一致 |
| 重复第一条问题 | HTTP 200；回答和引用与第一次一致，缓存 TTL 有效 |
| Redis 历史 | 三次请求共保存 6 条消息 |

HTTP 验收使用独立会话和随机问题，完成后只清理本次数据。接口文档地址为 <http://127.0.0.1:8000/docs>。

这次确认第 1 周核心链路可运行，不等于完成第 3 周商品上下文、第 4 周商品提示词或第 5 周正式接口。当前缓存仍只按问题生成键，工作记忆仍在进程内，多次检索的引用编号仍需统一管理。

## 11. 第三周第一步：商品表（2026-10-14）

新增 `data/products.json`，保存三个商品的 `id`、`sku`、`name`。SKU 沿用商品文档中的 `100000`、`100001`、`100002`，保持字符串；这些字段分别对应 Chunk Metadata 的 `product_id`、`sku_id`、`product_name`。

修改 `.gitignore`，允许商品表纳入版本控制；README 增加商品表说明和[第三周进度记录](Week3_Product_Context.md)。校验 JSON 格式、必填字段、ID/SKU 唯一性，并核对商品表与 Markdown 及现有商品库的身份一致。

下一步按计划修改 API 输入。当前第一步只建立商品表，API、State 和正式商品检索过滤仍待接入。

## 12. 第三周第二步：API 商品输入（2026-10-14）

`backend/main.py` 新增 `POST /query`，接收 `message`、可选的 `product_id` 和 `session_id`；原有 GET 参数方式保留，并支持商品 ID。响应增加 `product_id`，继续返回回答、引用和会话编号。非空字符串校验在接口层完成，未知商品返回 404，非法输入返回 422。

新增 `backend/products.py` 读取商品表并校验基础身份。Agent 延迟到合法请求进入时加载，商品不存在或输入错误时不加载 BGE、不调用聊天模型，**先检查请求参数，检查通过后，才加载并执行 Agent**

**校验基础身份，就是检查商品表中的 `id`、`sku`、`name` 是否完整、类型正确，并且能唯一识别商品。**

例如：

```
{
  "id": "vivo_x500_pro",
  "sku": "100001",
  "name": "vivo X500 Pro"
}
```

目前代码会检查：

- **字段齐全**：每条商品都有 `id`、`sku`、`name`。
- **非空字符串**：不能缺值或只有空格；SKU 必须写成 `"100001"`，保留字符串形式。
- **编号唯一**：不同记录不能使用相同的商品 ID 或 SKU。

接口还会检查：前端传来的 `product_id` 是否存在于商品表中，不存在就返回 **404**

`agent_invoke()` 接收商品 ID，并将它透传到 LangGraph 调用配置。`backend/cache.py` 增加独立的商品缓存前缀，按商品与问题生成键；无商品请求保留旧键。商品 ID 写入 State 和正式检索过滤分别留给第三、四步，本步骤不宣称已实现“这款”的商品指代。

新增 `tests/test_api.py`、`tests/test_products.py`、`tests/test_agent_product_input.py`，33 项测试覆盖两种 HTTP 输入、身份校验、商品透传和缓存隔离。完整离线回归为 **145 项通过、1 项真实 Agent 测试默认跳过**；Ruff 与修改文件的格式检查通过。

真实 HTTP 验收：POST 返回 200，商品与会话编号正确，回答演示资料中的最高 50W 无线充电并返回 1 条引用；GET 返回相同的商品缓存。不存在商品为 404，空问题为 422。测试使用随机问题和独立会话，结束后只清理本次 Redis 数据。

接口使用示例及阶段边界见[第三周进度](Week3_Product_Context.md)。下一步将 `product_id` 加入 `MessagesState`。

## 13. 第三周第三步：State 当前商品（2026-10-15）

在 `backend/state.py` 的 `MessagesState` 中增加 `product_id: str | None`。`backend/retrieval.py` 每轮将它放入 `agent.invoke()` 的输入；调用配置只用 `thread_id` 选择会话。商品 ID 使用新值覆盖旧值，消息及引用继续累积。省略或传 `null` 明确写入 `None`，表示全库模式。

缓存命中原来会提前返回，本步用 `agent.update_state()` 保存本轮用户消息、缓存回答、引用和商品 ID，保留已有模型调用计数，并以最终回答节点已完成的状态保存检查点。这样缓存命中的新会话也有自己的两条消息，切换商品后复用缓存也会更新当前商品。

`tests/test_agent_product_input.py` 使用真实 LangGraph 和 MemorySaver，替代模型节点和外部 Redis，8 项测试覆盖连续提问、商品切换、回到全库、会话隔离、缓存状态和引用保存。`tests/test_agent.py` 新增真实服务验收，确认 Pro → 普通版 → 缓存恢复 Pro，新会话缓存命中的模型调用计数为 0。

完整离线回归 **151 项通过、2 项真实测试默认跳过**；新增真实商品状态测试单独运行 **1 项通过**。修改代码的 Ruff 和格式检查通过。测试只清理自己生成的 Redis 数据，FastAPI 重启后加载新代码。

本步完成 State 保存，第四步的正式商品检索过滤仍待接入。模型尚未直接使用商品 ID 理解指代，提问继续写明型号；检查点仍在进程内，Redis 历史自动恢复和会话历史缓存隔离仍待完善。完整开发过程及手动测试命令见[第三周记录](Week3_Product_Context.md)。

## 14. 第三周第四步：正式商品过滤（2026-10-15）

`backend/tools.py` 的 `retrieve_docs` 新增后端注入参数。`backend/nodes.py` 从 State 读取 `product_id`，覆盖模型可能提交的同名参数；模型的工具参数只保留 `query`。指定商品时实际执行 `similarity_search(query, k=5, filter={"product_id": product_id})`；未指定时继续搜索全库，空结果不跨商品补查。原有 Chunk Metadata 已具备条件，无需重新建库。

模型调用与最终回答前补充商品表中的当前名称、ID 和指代说明，因此提问可以只写“这款”。为避免旧商品工具结果混入回答，State 新增 `product_context_start`，入口在商品切换时更新消息起点；模型和引用校验只读取该起点后的消息，完整图消息和 Redis 历史仍保存。同商品追问保留上下文，切回旧商品时从新的这一轮开始，缓存命中也执行该逻辑。

引用从当前商品上下文的工具结果及缓存回答收集；检查点将工具引用恢复为字典时转换回 `Citation`。`citations.py` 增加可选起始编号，连续工具检索按最大编号续号，默认独立调用仍从 1 开始。商品缓存前缀升级到 v2，避免复用正式过滤前的旧答案；旧键不删除，无商品缓存键保持兼容。

新增 12 项商品过滤测试和 1 项引用测试。完整离线回归 **164 项通过、3 项真实服务测试默认跳过**；修改代码的 Ruff、格式和差异空白检查通过。新增真实测试 **1 项通过**，使用真实聊天模型、BGE、Chroma 和 Redis，同一句不含型号的问题验证 Pro、普通版、Max 的回答及全部工具来源；切回 Pro 正确命中缓存。

重启本项目 FastAPI 后实际请求 `POST /query`：同会话 Pro → 普通版 → 缓存返回 Pro，三次均为 HTTP 200，回答分别符合演示资料中的最高 50W、不支持、最高 50W；引用只来自各自商品文件，缓存返回与首次一致。Redis 保存 6 条消息，缓存 TTL 有效；只清理了本次随机会话及问题缓存，服务保留运行在本机 8000 端口。

第三周四个开发步骤已完成。手动 HTTP 请求、真实测试命令和各文件职责见[第四步开发记录](Week3_Product_Context.md#第四步正式检索商品过滤已完成)。Streamlit 尚未添加商品选择控件；检查点自动持久化、包含历史的缓存策略、完整商品提示词及工具改名继续按后续计划推进。

## 15. 商品独立会话与 SQLite 检查点（2026-10-15）

新增 `backend/sessions.py`，按 `session_id` 与 `product_id` 的结构化 JSON 计算稳定摘要，生成独立的 LangGraph `thread_id`。API 及 Redis 历史仍使用原会话 ID；Pro、普通版、Max、全库分别保存模型上下文。切回 Pro 会继续原来的 Pro 对话，因此移除上一版 `product_context_start` 截取消息的方案。

新增 `backend/checkpoints.py`，使用官方 `SqliteSaver` 保存完整图检查点。默认 `data/checkpoints.sqlite`，支持 `CHECKPOINT_SQLITE_PATH`，相对路径按项目根目录解析；连接支持请求线程，退出时关闭，状态留在文件中。引用对象显式注册到序列化器。`backend/config.py` 用它替换 MemorySaver；默认数据库及辅助文件由现有忽略规则覆盖。

`backend/retrieval.py` 恢复商品线程后检查是否有历史。只有无历史的首轮读取和写入共享答案缓存，追问完全跳过，防止上下文不同却复用相同问题的答案。商品首轮缓存升级 v3，全库升级 v2，不删除旧数据；缓存命中新线程仍保存问题、回答及引用到 SQLite。

依赖与锁文件增加 `langgraph-checkpoint-sqlite==3.1.2`，其底层适配升级为 `langgraph-checkpoint==4.3.0`。新增 6 项检查点测试，覆盖连接重开、真实请求线程、类型恢复、线程删除及独立 Python 进程重启；更新原商品状态和过滤测试。完整离线回归 **172 项通过、3 项真实测试默认跳过**，三项真实 Agent 测试单独运行 **全部通过**。Ruff、格式和 235 个安装包的依赖兼容性检查通过。

真实 HTTP 验收中，Pro 和普通版分别记住随机备注，切回 Pro 正确读取其原备注，且追问跳过特意放入的错误共享缓存。停止并重启 FastAPI 后，两个商品各自恢复正确备注；五次请求均为 HTTP 200，Redis 共保存 10 条消息。仅清理本轮随机缓存、历史和 SQLite 线程，默认数据库文件保留，FastAPI 继续运行在本机 8000 端口。

从前 MemorySaver 中的状态不自动迁移，原 Redis 聊天文本继续保留；新版请求开始写入 SQLite。完整实现、存储分工及切换后重启续聊的手动步骤见[开发文档](Product_Sessions_and_Checkpoints.md)。
