# Flame 开发记录：项目整理与商品知识库

更新日期：2026-10-04。本文汇总本轮已完成的修改，对应[圣火文档](../圣火文档.md)第 1～2 周。

当前已完成项目命名清理、三个商品文档、Chunk Metadata、商品建库脚本和中文注释。已使用本地 BGE-M3 建成商品库：**22 个文本块，每个向量 1024 维**。后端接入商品库和语义检索验收仍待完成。

## 1. 项目命名与法律内容清理

项目从 LawGlance 整理为 Flame，主要变化如下：

| 范围 | 修改结果 |
| --- | --- |
| 项目与界面 | `pyproject.toml`、`uv.lock` 和 Streamlit 界面使用 Flame 名称，移除法律主题介绍、链接和图片 |
| 经典 RAG 入口 | `lawglance_main.py` 改为 `flame_main.py`，`Lawglance` 类改为 `Flame`；调用方需使用新名称 |
| 提示词 | `prompts.py` 和 `config/prompts.yaml` 改为通用文档问答，要求依据资料回答并提供编号引用 |
| 引用 | `citations.py` 支持通用文档标题、章节和页码；新增商品名支持，可显示“商品名 + 章节” |
| 缓存 | Redis 回答缓存和聊天历史使用 `flame:` 键前缀，避免混用旧项目缓存 |
| PDF 建库 | 使用 `scripts/build_bge_db.py` 导入指定的本地 PDF，移除固定法律资料下载和清空目录逻辑 |
| 评测与测试 | 法律专用样例改为通用文档或商品样例；检索评测使用共享的本地 BGE；旧法律 QA 数据清空，等待补充新参考集 |
| 文档与 Notebook | 项目文档改为 `docs/Flame_Documentation.md`；示例改为 `src/pdf_ingestion.ipynb`、`examples/flame_crewai.ipynb` |

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
uv sync

# 导出文本块及 Metadata
uv run python -m scripts.prepare_product_chunks

# 只校验和预览，不加载模型、不写数据库
uv run python -m scripts.build_product_db --dry-run

# 使用 .env 中配置的模型建库
uv run python -m scripts.build_product_db --device cpu
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

商品库有独立的默认路径和集合名。**当前后端仍读取通用 `chroma_db`，尚未接入商品库；接入时必须同时配置 `chroma_db_products` 路径和 `collection_name="products"`，仅修改 `.env` 中的路径不足以完成接入。**

## 6. 中文注释与版本管理

42 个 Python 文件的注释及模块、类、函数 docstring 已改为中文。三个 Notebook 的说明文字和函数 docstring 同步中文化。中文化过程中对比了移除 docstring 后的 AST，确认执行逻辑保持一致；代码标识符和运行时提示词保留原有形式。

`.gitignore` 已允许 `data/products/*.md` 纳入版本控制；`.env`、模型、向量库、生成的 JSONL、缓存和虚拟环境继续忽略。提交范围包括源码、测试、配置模板、依赖锁定文件、商品样例和开发文档；本地商品库可通过建库脚本重新生成。

## 7. 验证结果

| 检查 | 结果 |
| --- | --- |
| 自动化测试 | 108 项通过；1 项真实服务测试按配置跳过 |
| 代码检查 | Ruff 静态检查、格式检查和 `git diff --check` 通过 |
| Notebook | 格式校验及代码语法检查通过 |
| 真实模型建库 | 使用指定目录的 BGE-M3 成功生成并保存 22 个向量 |
| 数据回读 | 重新打开商品库，核对唯一 ID、完整 Metadata、7/7/8 分布和 1024 维向量 |

测试覆盖商品字段校验、切分继承、稳定 ID、JSONL 导出、商品引用标签，以及重复建库、文档更新、文件移除、保留非管理记录和失败时不删除旧数据。数据库单元测试使用真实 Chroma 和替代向量模型；本地商品库另行使用真实 BGE 生成。

```bash
uv run pytest tests/ -q
uv run ruff check .
uv run ruff format --check .

# 配置聊天服务、Redis 和语料库后，才运行真实 Agent 测试
RUN_AGENT_INTEGRATION=1 uv run pytest tests/test_agent.py -q
```

直接执行 `python tests/test_agent.py` 不会运行 pytest 测试。本轮验证借用了本机已有的 Python 依赖环境，当前仓库尚未建立自己的 `.venv`。

## 8. 当前待办与下一步

按依赖顺序继续：

1. 用 `uv sync` 建立本项目环境，补齐聊天 API 配置并确认 Redis；启动 FastAPI、实际调用 `/query`，验收完整 Agent 链路。
2. 完成第 2 周第 4 步：在 `backend/config.py` 中配置 `product_vector_store`，统一后端调用、商品库路径和集合名。
3. 完成第 2 周第 5 步：实际检索“X500 Pro 有什么卖点”“有哪些配置”“Pro Max 和 Pro 有什么区别”，核对返回资料及来源。当前存储校验通过，语义召回质量尚未验收。
4. 核验或替换演示商品资料，补充人工审核的评测 QA；补上 README 入口，后续按开发阶段提交改动。
5. 进入第 3～4 周后，再增加 API/State 中的 `product_id`、商品 Metadata 过滤、商品专用提示词及 `retrieve_product_docs` 工具。

现有对话机制还有三项限制：Agent 工作记忆保存在进程内、后端回答缓存仅按问题共享、多次检索的引用编号可能冲突。商品上下文接入时需要一并检查会话和商品隔离。

更详细的阶段检查见[第二周 Metadata 与进度记录](Week2_Metadata_and_Progress.md)；通用运行说明见[项目说明](Flame_Documentation.md)和[贡献指南](../CONTRIBUTING.md)。
