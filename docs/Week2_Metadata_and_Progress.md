# 第二周：商品 Metadata 与进度检查

本轮全部修改的简明说明见 [Flame 开发记录](Flame_Development_Notes.md)。

按 [圣火文档](../圣火文档.md) 的第二周第 2、3 小节实现商品 Metadata 与建库脚本。
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
通过真实本地 BGE 生成。语义召回质量仍需按第 5 小节的问题单独验收。

## 第 1 周检查

| 计划项 | 当前状态 | 尚需完成的内容 |
| --- | --- | --- |
| Flame 命名与目录 | 已完成主要整理 | `backend/`、`scripts/`、`data/products/`、`config/` 均存在 |
| 百炼、本地 BGE、Redis、Agent 原链路 | 本地 BGE 与商品 Chroma 已验证；完整 Agent 链路未验收 | `.env` 已配置 BGE；仍需补充聊天 API 配置、确认 Redis，并运行真实 smoke test |
| FastAPI 启动与请求 | API 实现存在，未完成运行验收 | 启动服务并实际调用 `/query` |
| 本项目依赖环境 | 当前没有 `.venv` | 用 `uv sync` 建立可复现的本项目环境 |
| Git 仓库整理 | 源码、文档和商品样例已整理 | 后续按阶段提交；使用 `git status` 和 `git log` 核对工作区及提交状态 |
| README 启动说明 | `README.md` 不存在 | 启动说明已有 `CONTRIBUTING.md` 和 `docs/Flame_Documentation.md`；若严格按计划验收，仍需补 README 入口 |

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
| 4. 后端指向商品库 | 尚未完成 | `backend/config.py` 当前仍是通用 `vector_store`，默认路径为 `chroma_db`；后续改为 `product_vector_store` 和商品库路径 |
| 5. Retriever 验收 | 尚未完成 | 增加检索脚本或测试，实际验证卖点、配置、Pro Max 与 Pro 区别三个问题的召回 |

之前商品文档被 `/data/` 忽略。本次已将 `.gitignore` 调整为允许
`data/products/*.md` 纳入版本控制，JSONL、向量库和其他本地数据继续忽略。

当前项目目录已生成真实的 `chroma_db_products`。之前的本地 BGE 数据库仍是
原有语料。单元测试使用本机已有 Python 依赖环境，数据库测试只写入临时
目录；实际商品建库另外调用了真实 BGE，未调用聊天服务。存储校验通过不等于
语义检索质量通过。

## 中文注释

42 个 Python 文件的注释、模块说明和函数或类 docstring 已中文化，
3 个 Notebook 的说明文字及函数 docstring 也已中文化。
修改时对比了移除 docstring 后的 AST，确认运行逻辑保持一致。
验证结果为 108 项单元测试通过、1 项真实服务测试按配置跳过。

## 后续阶段

商品上下文、API 中的 `product_id`、Metadata Filter 属于第 3 周。
正式商品助手提示词和工具改名属于第 4 周。本次没有将这些后续工作算作
已经完成，也没有提前改动 LangGraph 流程。
