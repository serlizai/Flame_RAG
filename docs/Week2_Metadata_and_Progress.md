# 第二周：商品 Metadata 与进度检查

本轮全部修改的简明说明见 [Flame 开发记录](Flame_Development_Notes.md)。

按 [圣火文档](../圣火文档.md) 的第二周第 2～5 小节实现商品 Metadata、建库脚本、后端商品库配置与基础检索验收。
检查范围为当前 Flame 仓库；不将其他项目或此前手动运行的服务视为已经验收。

## 已完成的 Metadata

每个商品 Markdown 的一级标题提供 `product_name`，头部的 `SKU` 提供
`sku_id`，`product_id` 字段直接保留。来源使用文件名；当前商品分类默认为
`smartphone`，可用命令参数修改。

实际 X500 Pro 参数块的 Metadata 示例：

```json
{
  "product_id": "vivo_x500_pro",
  "sku_id": "100001",
  "product_name": "vivo X500 Pro",
  "category": "smartphone",
  "source": "x500_pro.md",
  "section": "核心参数",
  "chunk_index": 2,
  "chunk_id": "c898363842e1ce3b813b7b716a74073e5130d53f991e7b45fee1bb24eb617d08"
}
```

五个业务字段在每个 Chunk 上都存在，字段类型均为字符串。SKU 不转换成
整数，因此 `001001` 这样的编号不会丢失前导零。附加字段的用途：

| 字段 | 用途 |
| --- | --- |
| `section` | 记录商品简介、核心参数、FAQ 等章节，用于定位和引用 |
| `chunk_index` | 从 0 开始的商品内块序号 |
| `chunk_id` | 根据业务标识、序号和文本生成 SHA-256，相同输入可以复现 |

Chunk ID 会随内容或切分结果变化。建库脚本已经在新块成功写入后删除旧块，
防止修改后的文档留下过期数据；仅靠 ID 不能完成这一清理。

## 生成方式与当前产物

```bash
uv run python -m scripts.prepare_product_chunks
```

脚本先解析和校验全部商品，再按二级章节切分，最后使用
`RecursiveCharacterTextSplitter`，默认 `chunk_size=800`、`chunk_overlap=100`。
不同章节不会合成一个块；章节较短时不会强制补足 800 字符或制造重叠。

当前三个文档生成的块数量：

| 文档 | 商品 ID | SKU | Chunk 数 |
| --- | --- | --- | --- |
| `x500.md` | `vivo_x500` | `100000` | 7 |
| `x500_pro.md` | `vivo_x500_pro` | `100001` | 7 |
| `x500_pro_max.md` | `vivo_x500_pro_max` | `100002` | 8 |
| 合计 | — | — | 22 |

`data/product_chunks.jsonl` 每行保存 `page_content` 和 `metadata`。这一步不加载
BGE、不调用聊天模型、不写入 Chroma，尚不能据此判定商品检索已经完成。

解析器会拒绝缺失或重复的头部字段、不同文档复用商品 ID 或 SKU、空商品
目录和非法切分参数。FAQ 正文中的编号示例不能代替商品头部编号。
引用模块已经支持用 `product_name` 和 `section` 构造引用标签。

## 第 3 小节：商品建库脚本

已新增 `scripts/build_product_db.py`，实现以下流程：

```text
data/products/*.md
→ 商品字段校验与 Document
→ 按章节切分，再按 800/100 进行文本切分
→ 本地 BGE Embeddings
→ Chroma
→ chroma_db_products，collection=products
```

真实建库命令：

```bash
uv run python -m scripts.build_product_db --model-path /path/to/bge-m3
```

也可在 `.env` 中配置 `BGE_M3_PATH` 后省略 `--model-path`。模型目录必须已存在。
可用 `--device cpu` 或 `--device mps` 指定设备，默认读取 `EMBEDDING_DEVICE`，
未配置时使用 CPU。无需聊天 API 或 Redis。

```bash
uv run python -m scripts.build_product_db --dry-run
```

已在三个现有商品文档上完成预览和真实 BGE 建库。模型目录为
`/Users/lwj/AI_project/RAG/models/BAAI/bge-m3`，已写入本地 `.env`。
当前使用 CPU，`chroma_db_products` 的 `products` collection 保存 22 个 Chunk，
每个向量为 1024 维；已重新打开数据库核对数量和完整 Metadata。

写入时保存全部商品 Metadata，并增加 `ingestion_pipeline` 标记。脚本按
Chunk ID 执行 upsert，成功后才清理本脚本生成的过期块。重复运行不会累积
重复记录；文档修改和文件移除后，可以清理相应旧块。非本脚本写入的记录、
其他 collection 及数据库目录内其他文件不会被清除。

单元测试使用真实 Chroma 与离线 embedding 替身，已验证重复写入、文档更新、
文件移除、Metadata 保留，以及 embedding 失败时保留旧数据。默认商品库则
通过真实本地 BGE 生成。三个问题的基础召回已另行验收，结果见第 5 小节。

## 第 4 小节：后端接入商品库

`backend/config.py` 已将通用 `vector_store` 改为 `product_vector_store`，
路径固定为项目根目录下的 `chroma_db_products`，显式指定集合名 `products`。
`backend/tools.py` 中的 `retrieve_docs` 已同步使用商品库，仍返回带编号的文本和引用。

已通过后端配置回读本地商品库，确认 22 个块及 7/7/8 的商品分布。新增离线测试
使用真实 Chroma，在通用目录及商品目录的默认集合中放入其他语料，验证后端
和工具只读取正确的商品集合；从其他工作目录运行也能定位数据库。
该验证替代了聊天模型和 embedding 初始化，未调用真实聊天服务或验收语义召回。

## 第 5 小节：最简单检索测试

新增 `scripts/test_retrieval.py`，读取与后端相同的商品库和 `products` 集合，
加载真实本地 BGE，执行 `similarity_search(query, k=5)`。不调用聊天模型或
Redis，默认不使用商品 Metadata 过滤，不重建数据库。

```bash
uv run python -m scripts.test_retrieval --device cpu
```

也支持 `python scripts/test_retrieval.py`。直接执行文件时，脚本根据自身位置
将项目根目录加入导入路径；使用脚本的绝对路径时，可从其他工作目录运行。
新增入口回归测试覆盖直接执行和模块执行，不依赖外部 `PYTHONPATH`。

运行时打印每个结果的商品名、章节、来源和正文摘要。单商品问题要求首条
结果属于目标商品；对比问题要求前 5 条同时包含两个商品。全部通过返回
退出码 0，任一问题未达标返回 1；缺失数据库或非法参数返回 2。

2026-10-04 使用现有 22 个块和本地 BGE-M3 完成基础验收：

| 问题 | 实际召回 | 基础结果 | 章节诊断 |
| --- | --- | --- | --- |
| X500 Pro 有什么卖点？ | 第 1 条为 Pro 的商品简介 | 通过 | Pro 的核心卖点未进入前 5 条 |
| X500 Pro 有哪些配置？ | 第 1 条为 Pro 的商品简介 | 通过 | Pro 的核心参数未进入前 5 条 |
| X500 Pro Max 和 Pro 有什么区别？ | 第 1 条为 Pro Max FAQ，第 3 条为 Pro 商品简介 | 通过 | 两个目标商品均已召回 |

基础商品召回 **3/3 通过**。章节命中是单独诊断，不计入商品归属验收。
结果仍混有其他型号；核心卖点和核心参数的章节召回有待改善。本测试未评估
最终回答正确率，也不能替代完整评测集。可用 `--k` 修改召回数量，
用 `--persist-directory`、`--model-path`、`--device` 指定数据库和模型配置。

### 可选的商品过滤演示

2026-10-05 新增 `--filter-products`，按测试用例中明确指定的商品范围检索：

```bash
uv run python -m scripts.test_retrieval --filter-products --device cpu
```

单商品条件为 `filter={"product_id": "vivo_x500_pro"}`；对比条件为
`filter={"product_id": {"$in": ["vivo_x500_pro", "vivo_x500_pro_max"]}}`。
已有 Metadata 足以支持过滤，无需重新建库。开启后还会检查结果没有混入
候选范围外的商品。

真实 BGE 验证：三个问题仍然 3/3 通过，Pro 的核心卖点排第 4、核心参数
排第 5，均进入前 5 条；对比结果只包含 Pro 和 Pro Max。
该开关使用测试用例的已知候选范围，未实现从问题自动识别商品。
正式业务中应从 API 和 Agent State 获取当前 `product_id`，再传给检索工具。

## 第 1 周检查

| 计划项 | 当前状态 | 尚需完成的内容 |
| --- | --- | --- |
| Flame 命名与目录 | 已完成主要整理 | `backend/`、`scripts/`、`data/products/`、`config/` 均存在 |
| 百炼、本地 BGE、Redis、Agent 原链路 | 本地 BGE 与商品 Chroma 已验证；完整 Agent 链路未验收 | `.env` 已配置 BGE；仍需补充聊天 API 配置、确认 Redis，并运行真实 smoke test |
| FastAPI 启动与请求 | API 实现存在，未完成运行验收 | 启动服务并实际调用 `/query` |
| 本项目依赖环境 | 已建立独立 `.venv`，Python 3.11.15 | 使用 `uv sync --locked --python 3.11` 复现，后续通过 `uv run` 运行 |
| Git 仓库整理 | 源码、文档和商品样例已整理 | 后续按阶段提交；使用 `git status` 和 `git log` 核对工作区及提交状态 |
| README 启动说明 | 已补齐中文 [README](../README.md) | 包含安装、配置、建库、检索、服务启动与完整链路验收方式 |

真实 Agent 测试现在是 pytest 的显式可选测试。仅执行
`python tests/test_agent.py` 不会执行该测试，正确命令为：

```bash
RUN_AGENT_INTEGRATION=1 uv run pytest tests/test_agent.py -q
```

运行前需要配置聊天服务、本地 BGE、Redis 和已建好的语料库。

## 第 2 周检查

| 计划项 | 当前状态 | 下一步 |
| --- | --- | --- |
| 1. 三个商品文档 | 结构和演示内容已完成 | 当前为虚构演示数据；计划中的真实商品资料尚未核验 |
| 2. 每个 Chunk 保存 Metadata | 已完成解析、继承、引用、JSONL 落盘和商品库写入 | 建库脚本已复用 `load_product_documents()` 与 `split_product_documents()` |
| 3. 商品建库脚本 | 已实现并完成真实 BGE 建库 | `chroma_db_products` 的 `products` collection 已保存 22 个 1024 维向量 |
| 4. 后端指向商品库 | 已完成配置、工具接入与数据回读 | `product_vector_store` 指向 `chroma_db_products` 的 `products` 集合 |
| 5. Retriever 验收 | 真实 BGE 基础商品召回 3/3 通过 | 章节命中和混入其他型号的问题已记录；第 3 周继续加入商品上下文与过滤 |

之前商品文档被 `/data/` 忽略。本次已将 `.gitignore` 调整为允许
`data/products/*.md` 纳入版本控制，JSONL、向量库和其他本地数据继续忽略。

当前项目目录已生成真实的 `chroma_db_products`。之前的本地 BGE 数据库仍是
原有语料。前期测试使用本机已有 Python 环境，现已切换到 Flame 独立 `.venv`；
数据库测试只写入临时目录。实际商品建库另外调用了真实 BGE，未调用聊天服务。存储校验通过不等于
语义检索质量通过。

独立环境安装时修复了旧 `PyMuPDFb` 与新版 `PyMuPDF` 的底层库冲突，
依赖声明和锁文件同步调整。当前聊天 API 配置仍为空，环境和 README 完成
不代表 FastAPI、Redis 与真实模型的完整问答链路已经验收。

## 中文注释

现有 Python 文件的注释、模块说明和函数或类 docstring 已中文化，
保留的 PDF 建库 Notebook 说明文字及函数 docstring 也已中文化。
前期中文化时对比了移除 docstring 后的 AST，确认运行逻辑保持一致。
验证结果为 112 项测试通过、1 项真实服务测试按配置跳过；已在 Flame 独立
`.venv` 中复验，并确认真实 BGE 商品过滤检索 3/3 通过。

## 后续阶段

商品上下文、API 中的 `product_id` 和业务链路中的 Metadata Filter 属于第 3 周；
当前测试开关仅演示指定商品范围的过滤。
正式商品助手提示词和工具改名属于第 4 周。本次没有将这些后续工作算作
已经完成，也没有提前改动 LangGraph 流程。

## 架构清理

当前仅保留 LangGraph Agent 问答入口。经典 RAG 对话类、固定问答链路、
相关 Notebook、未使用的 schema 和独立 CrewAI 示例已删除。
Agent 缓存集中在 `backend/cache.py`；商品评测改为读取 `products` 集合，
使用与 Agent 一致的 Top-5 相似度检索。清理详情见开发记录第 9 节。
这项清理不等于完成第 3 周的商品上下文和业务过滤。
