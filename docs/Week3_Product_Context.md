# 第三周：当前商品上下文

更新日期：2026-10-15。按[圣火文档](../圣火文档.md)第三周的步骤推进。

本文保留第三周的三商品验收记录。2026-10-16 商品表已扩充为 10 项，并增加 `category` 字段，最新数据规模和验收见[十商品扩充记录](Ten_Product_Catalog.md)。

第四周已将正式检索工具由本阶段的 `retrieve_docs` 改为 `retrieve_product_docs`，下文原名称保留为阶段记录；当前实现见[第四周文档](Week4_Product_RAG_MVP.md)。

当前已进一步完成[商品独立会话与 SQLite 检查点](Product_Sessions_and_Checkpoints.md)：每个会话的不同商品分别保存上下文，切回 Pro 可接着原对话，重启后可恢复。以下各步的测试数字为当时验收记录，最新离线回归见十商品扩充记录。

## 第一步：建立商品表（已完成）

新增 [data/products.json](../data/products.json)，保存商品列表，后续供商品选择、商品身份查询和 API 校验使用。

| id | sku | name |
| --- | --- | --- |
| vivo_x500 | 100000 | vivo X500 |
| vivo_x500_pro | 100001 | vivo X500 Pro |
| vivo_x500_pro_max | 100002 | vivo X500 Pro Max |

每个商品只保存三个基础字段，SKU 保持字符串。字段与现有文档及 Chunk Metadata 的对应关系如下：

| 商品表字段 | Metadata 字段 | 用途 |
| --- | --- | --- |
| id | product_id | 唯一标识商品，后续作为检索过滤条件 |
| sku | sku_id | 标识当前商品配置 |
| name | product_name | 展示名称和生成当前商品提示词 |

编号沿用现有商品文档，不照搬计划示例中的 SKU。商品规格、FAQ 和售后知识继续保存在 Markdown 中，当前仍为演示资料。

`.gitignore` 已放行商品表，其他本地数据和向量数据库继续忽略。

## 验收

已检查 JSON 格式、必填字符串字段，以及商品 ID 和 SKU 的唯一性；三个商品的 ID、SKU、名称与 Markdown 头部及现有商品库的全部 Chunk Metadata 一致。无需重新切分或建库。

## 第二步：修改 API 输入（已完成）

新增 `POST /query`，接收 JSON 请求体：

```json
{
  "product_id": "vivo_x500_pro",
  "message": "X500 Pro 支持无线充电吗？",
  "session_id": "my-session"
}
```

`message` 为必填的非空字符串。`session_id` 可省略，由后端生成；后续请求传回相同编号即可继续会话。`product_id` 可省略，此时沿用全库问答模式。两个可选字段也接受 JSON 的 `null`。字符串会去除首尾空白，字段拼写错误、空问题或空白字符串字段返回 422。

原有 `GET /query?query=...&session_id=...` 保留，并可增加 `product_id` 参数。两种方式共用校验及后端调用逻辑，响应包含 `answer`、`citations`、`session_id`、`product_id`。

新增 `backend/products.py`，从项目根目录读取商品表，校验必填字段、字符串 SKU、ID/SKU 唯一性。接口在加载 Agent 前检查商品是否存在：不存在返回 404，商品表配置错误返回 500。

`agent_invoke()` 新增可选 `product_id`，传入 LangGraph 的调用配置。商品答案缓存使用商品 ID 和问题作为键，并使用独立前缀；未指定商品时保留原有缓存键。

**本步骤完成输入接收和身份传递，还没有让模型按当前商品理解“这款”，也没有启用检索过滤。** 第三步将商品 ID 写入 State，第四步再将它传给检索工具。当前会话记忆仍保存在进程内，缓存仍未包含会话历史。

### 验证结果

- 新增 33 项离线测试，覆盖商品表读取、HTTP 路由、非法输入、商品身份透传和缓存区分。
- 回归测试：145 项通过，1 项真实 Agent 测试默认跳过；Ruff 检查和修改文件的格式检查通过。
- 真实 HTTP 验收：POST 返回 200，回答演示资料中的最高 50W 无线充电，返回 1 条引用，商品 ID 和会话编号正确；GET 返回相同商品缓存，状态为 200。
- 不存在的商品返回 404，空问题返回 422。验收后只清理本次随机会话和问题的 Redis 数据。

## 第三步：把当前商品写入 State（已完成）

### 怎么开发的

| 文件 | 修改内容 |
| --- | --- |
| `backend/state.py` | 在 `MessagesState` 中增加 `product_id: str \| None` |
| `backend/retrieval.py` | 每轮把商品 ID 作为图输入；缓存命中时也更新图状态 |
| `tests/test_agent_product_input.py` | 使用真实 LangGraph 和 MemorySaver，验证商品状态及缓存分支 |
| `tests/test_agent.py` | 增加真实模型、BGE、Chroma、Redis 的商品状态验收 |

第二步把商品 ID 放在调用配置里，本步把它放进真正的 State 输入：

```python
config = {"configurable": {"thread_id": session_id}}
result = agent.invoke(
    {"messages": [HumanMessage(content=query)], "product_id": product_id},
    config=config,
)
```

`thread_id` 用来选择会话，`product_id` 表示这一轮的当前商品。消息和引用列表继续累积保存，商品 ID 则使用新值覆盖旧值，因此切换商品后不会留下旧商品 ID。

每轮都显式写入商品 ID：传入 Pro 时保存 Pro，切换普通版时保存普通版；省略或传 `null` 时写入 `None`，表示回到全库模式。继续同一商品的会话时，每轮继续传相同商品 ID。

### 为什么还要处理缓存

缓存命中会直接返回旧答案，原来这条分支没有执行 `agent.invoke()`，也没有更新图状态。本步用 `agent.update_state()` 保存当前商品、用户问题、缓存回答和引用，并保留已有模型调用计数。

使用 `as_node="final_answer"` 标记本轮回答已完成，实际不会再次调用该节点或模型。新会话第一次命中缓存也能获得两条消息，后续追问可以看到这轮对话。

### 怎么验证的

- 8 项状态测试通过，模型节点和 Redis 使用替身，状态图及内存检查点使用真实 LangGraph。
- 覆盖连续提问、切换商品、回到全库、不同会话隔离、缓存命中新会话、缓存命中覆盖旧商品、引用保存和模型调用计数。
- 完整离线回归：**151 项通过、2 项真实测试默认跳过**；修改代码的 Ruff 和格式检查通过。
- 单独运行新增真实测试：**1 项通过**。验证 Pro → 普通版 → 缓存恢复 Pro；新会话命中缓存后保存两条消息，模型调用计数为 0。仅清理本次测试的 Redis 缓存和历史。

### 你可以怎样手动测试

先在项目根目录运行无需聊天 API 和 Redis 的状态测试：

```bash
.venv/bin/python -m pytest tests/test_agent_product_input.py -v
```

配置好聊天服务、BGE 和 Redis 后，运行真实验收：

```bash
RUN_AGENT_INTEGRATION=1 .venv/bin/python -m pytest \
  tests/test_agent.py::test_agent_keeps_product_state_on_cache_and_switch -q -s
```

该测试会打印商品切换和缓存状态结果。新版查看图状态时，使用会话和商品生成内部线程配置，SQLite 支持跨进程读取：

```python
from backend.sessions import make_thread_config

state = agent.get_state(make_thread_config(session_id, "vivo_x500_pro")).values
print(state["product_id"])
```

API 请求格式保持第二步的形式。响应中的 `product_id` 是输入回显，State 的保存由上述 `get_state()` 检查确认。现有服务需要重启以加载新代码。

### 当前范围与下一步

第三步验收时只保存 State，尚未启用检索过滤。以下第四步已继续完成正式过滤及“这款”的商品身份传递。

## 第四步：正式检索商品过滤（已完成）

### 怎么开发的

| 文件 | 修改内容 |
| --- | --- |
| `backend/tools.py` | 指定商品时向 Chroma 传入 Metadata Filter；无商品时保持全库检索 |
| `backend/nodes.py` | 从 State 注入商品 ID，补充当前商品名称，只使用当前商品的上下文和引用 |
| `backend/state.py`、`backend/retrieval.py` | 保存商品状态；后续改进已按会话及商品分别恢复完整上下文 |
| `backend/cache.py` | 使用分商品的缓存版本；后续改进仅缓存无历史的首轮回答 |
| `citations.py` | 支持指定引用起始编号，连续工具检索时接续编号 |
| `tests/test_product_filter.py`、`tests/test_citations.py`、`tests/test_agent.py` | 覆盖真实数据库过滤、正式工作流、引用和可选真实服务验收 |

数据传递顺序为：API 接收 `product_id` → 入口写入 State → `tool_node` 从 State 读取 → `retrieve_docs` 添加过滤 → Chroma 返回对应商品的资料。

指定 Pro 时，实际检索等价于：

```python
product_vector_store.similarity_search(
    query,
    k=5,
    filter={"product_id": "vivo_x500_pro"},
)
```

`product_id` 和引用起点使用 `InjectedToolArg` 标记，不出现在模型可填写的参数结构中；模型只提供 `query`。工具节点还会覆盖模型可能提交的同名字段，因此检索范围以 State 为准。没有结果时返回未查询到资料，不改为搜索其他商品。

模型调用和最终回答前都补充商品表中的名称与 ID，让“这款”“它”指向当前商品。这是本周验收所需的商品身份提示；第四周的提示词整理和工具改名仍按后续计划进行。

### 为什么还要隔离旧资料

只给 Chroma 加过滤，不能排除同会话已经保存的其他商品工具结果。第四步最初用 `product_context_start` 截取切换后的消息；后续已替换为商品独立线程，移除此字段：

- 同商品追问或切回商品：读取该会话中该商品原来的完整上下文。
- 切换其他商品或全库模式：读取各自的独立线程，不带入其他商品消息。
- SQLite 持久化每个线程，Redis 继续汇总同一个前端会话的用户问题和最终回答。
- 每个商品对话只有无历史的首轮使用共享答案缓存，已有历史时继续执行 Agent。

引用只从当前上下文中的工具结果和缓存回答收集，旧商品编号不能通过最终引用校验。连续工具检索从当前最大编号之后续号，默认独立调用仍从 1 开始。检查点恢复后，工具引用有时会变成字典，读取时统一转换回 `Citation`。

第四步的商品缓存曾使用 v2；后续为保证只复用无历史的首轮答案，商品前缀升级为 `flame:llm_cache_product_v3:`，全库前缀升级为 `flame:llm_cache_v2:`。旧键不删除、不再命中，按其原有过期设置处理。完整策略见[会话检查点记录](Product_Sessions_and_Checkpoints.md)。

### 怎么验证的

- 新增 12 项商品过滤测试，使用真实临时 Chroma、正式 LangGraph 和内存检查点，替代聊天模型与 Redis。覆盖三个商品、全库、空结果、隐藏参数、覆盖伪造商品 ID、多次工具引用、同会话切换、切回缓存及旧引用排除。
- 新增 1 项引用起始编号测试。完整离线回归：**164 项通过、3 项真实服务测试默认跳过**；修改代码的 Ruff、格式和差异空白检查通过。
- 新增真实 Agent 测试单独运行：**1 项通过**，使用真实聊天模型、本地 BGE-M3、现有 Chroma 和 Redis。同一个不含型号的问题依次选择 Pro、普通版、Pro Max：Pro 与 Max 回答最高 50W，普通版回答不支持无线充电；实际工具结果与最终引用均只来自各自文件。切回 Pro 命中正确缓存，模型调用次数不增加。
- 真实 FastAPI 验收：同一会话 Pro → 普通版 → 缓存返回 Pro，三次 `POST /query` 均为 **HTTP 200**。Pro 回答最高 50W，普通版回答不支持；引用分别只来自 `x500_pro.md`、`x500.md`，最后一次答案及引用与首次完全一致。Redis 保存 6 条消息、缓存 TTL 有效。真实测试及 HTTP 验收均只清理自己创建的随机会话和缓存。

### 你可以怎样手动测试

在项目根目录运行离线测试：

```bash
.venv/bin/python -m pytest tests/test_product_filter.py -v
```

配置好聊天服务、模型路径、Redis 和已有商品库后，运行真实验收：

```bash
RUN_AGENT_INTEGRATION=1 .venv/bin/python -m pytest \
  tests/test_agent.py::test_agent_filters_current_product_for_implicit_question -q -s
```

重启 FastAPI 以加载本次修改，然后手动发起请求：

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```bash
curl 'http://127.0.0.1:8000/query' \
  -H 'Content-Type: application/json' \
  --data '{"product_id":"vivo_x500_pro","message":"这款支持无线充电吗？","session_id":"week3-demo"}'
```

预期回答最高 50W，`citations` 中所有 `source` 均为 `x500_pro.md`。保持问题和 `session_id`，把 `product_id` 改为 `vivo_x500`，预期回答不支持，所有引用来源为 `x500.md`。再切回 Pro，应返回 Pro 的答案及引用。每轮继续传入当前商品 ID；省略或传 `null` 表示切换全库模式。

现有 Streamlit 入口还没有商品选择控件，本周通过 API 的 `product_id` 演示商品选择。无需重新建库，原有 22 个 Chunk 已有过滤字段。后续已将图检查点持久化到 SQLite，切回商品及重启后可继续原来的商品对话，详见[开发与手动测试](Product_Sessions_and_Checkpoints.md)。

## 后续步骤

| 步骤 | 状态 |
| --- | --- |
| 1. 建立简单商品表 | 已完成 |
| 2. 修改 API 输入，接收当前 product_id | 已完成 |
| 3. 修改 LangGraph State，保存当前 product_id | 已完成 |
| 4. 正式检索按 product_id 过滤 | 已完成，真实 Agent 和 HTTP 验收通过 |

第三周四个开发步骤已完成，“选择商品 → 问这款 → 检索及引用只来自当前商品”的后端链路已验收。下一阶段按第四周计划整理商品提示词、工具命名和缺少可靠资料时的回答行为。
