# 第四周：商品 RAG MVP

更新日期：2026-10-16。按[圣火文档](../圣火文档.md)第四周分步推进。

## 第一步：整理商品提示词（已完成）

本步解决四件事：助手是谁、当前商品是什么、何时检索知识库、资料缺失时如何回答。工具改名已在下方第二步继续实现。

### 怎么开发的

| 文件 | 修改内容 |
| --- | --- |
| `prompts.py` | 中文商品系统提示词、当前商品构造函数、回答模板与提示词版本 |
| `backend/prompts.py` | 统一重新导出，保持后端使用同一套提示词 |
| `backend/nodes.py` | 检索决策和最终回答均调用 `build_system_prompt()` |
| `backend/cache.py` | 缓存前缀增加提示词版本，旧答案不绕过新规则 |
| `tests/test_prompts.py`、`tests/test_product_filter.py` | 验证身份插值、模板内容、两阶段接入和旧缓存隔离 |
| `tests/test_agent.py` | 增加真实身份、已知参数、缺失参数与备注记忆验收 |

`SYSTEM_PROMPT` 定义为带商品上下文的模板，`build_system_prompt()` 接收商品表确认的名称和 ID。节点从 State 获取 ID，读取对应商品身份后构造 `SystemMessage`：

```python
product = load_products()[state["product_id"]]
prompt = build_system_prompt(product.name, product.id)
```

没有商品 ID 时使用全库提示词：用户写明型号时围绕该型号回答；仅说“这款”且历史也无法确定型号时，先请用户指定商品。商品检索的实际过滤继续由后端 State 控制，提示词不改变过滤范围。

旧代码在节点中临时追加商品说明，本步将规则集中到 `prompts.py`，避免检索决策与最终回答使用不同的商品身份或回答要求。

### 提示词规定什么

- 角色：Flame 商品 AI 助手，默认使用中文简洁回答。
- 商品事实：涉及功能、规格、特点、使用说明、FAQ、价格及售后说明时，优先使用商品库；已有可靠原文可支持连续追问，资料不足时重新检索。
- 依据：原文必须与问题直接相关。检索到了资料不代表能回答，用户推测及模型常识不能代替商品参数的证据。
- 缺失：明确回复“暂未查询到相关信息。”，可以说明缺少哪个字段，不猜测数值或承诺。
- 引用：商品事实附支持该说法的原文编号；资料缺失、问候和纯聊天记忆不添加引用。
- 聊天：身份介绍、问候、用户自己的备注或预算可按角色或历史回答，不要求检索商品库。
- 演示资料：保留其演示属性，不当成已经核验的官方商品信息。

这些规则同时写入系统提示词和 `QA_PROMPT`，保持原有引用整理、商品线程和 SQLite 检查点机制。

### 为什么调整缓存

旧缓存可能由通用文档提示词生成，直接命中时不会执行新提示词。新增 `PROMPT_VERSION = "product_prompt_v1"`，把它加入商品及全库答案缓存前缀：

```text
flame:llm_cache_product_v3:product_prompt_v1:<问题与商品摘要>
flame:llm_cache_v2:product_prompt_v1:<问题摘要>
```

旧缓存保留并按原过期设置处理；商品与问题的摘要算法、首轮才使用共享缓存的策略不变。以后修改会影响回答行为的提示词时，应同步更新 `PROMPT_VERSION`。SQLite 线程 ID 没有变化，原会话继续使用新的提示词回答。

### 怎么验证的

- 11 项新增离线测试：三个型号与全库身份、模板中的花括号保留、后端统一导出、两阶段提示词一致、可解析的旧缓存不能被返回。
- 完整离线回归：**183 项通过、4 项真实服务测试默认跳过**；新增及修改测试另行复验通过。
- 修改代码的 Ruff、格式与差异空白检查通过。
- 新增真实模型测试单独运行：**1 项通过**，实际使用聊天服务、本地 BGE、Chroma、Redis 和 SQLite。结果如下：

| 真实问题 | 验证结果 |
| --- | --- |
| 你是谁 | 回答 Flame 商品 AI 助手，图状态中没有商品工具调用，无引用 |
| 当前 Pro 的无线充电 | 回答最高 50W，引用只来自 `x500_pro.md` |
| 处理器 L2 容量 | 回答“暂未查询到相关信息。当前资料未提供处理器的 L2 缓存容量。”，无引用 |
| 同商品追问备注 | 正确返回此前随机设置的备注，无商品引用 |

资料缺失规则明确要求只说明缺少的字段，不补充其他配置或引用仅写处理器名称的段落。真实测试结束后，仅清理本次随机会话的 Redis 缓存、历史和 SQLite 线程；用户自己的 FastAPI 进程未重启。

## 第二步：正式商品工具名称与描述（已完成）

### 怎么开发的

| 文件 | 修改内容 |
| --- | --- |
| `backend/tools.py` | 将函数及模型可调用名称改为 `retrieve_product_docs`，明确商品检索用途 |
| `backend/nodes.py` | 同步导入、工具绑定、路由和 State 参数注入，兼容旧检查点的工具名 |
| `prompts.py` | 检索规则改用新名称，提示词版本升级 `product_prompt_v2` |
| `tests/test_product_filter.py` | 新增新名绑定、旧调用执行、真实 SQLite 旧会话续聊及旧缓存隔离测试 |
| `tests/test_agent.py` 等 | 实际工具调用与引用断言同步新名称 |
| 评测注释、运行及贡献文档 | 同步正式工具名称 |

工具描述明确说明：用于查询当前商品的功能、参数、特点、使用方法、FAQ 和商品知识；价格及售后问题查询的是资料中的说明。返回带编号的原文，没有可靠答案时应明确说明未查询到。

模型只绑定一个正式工具 `retrieve_product_docs`，可填写的参数仍只有 `query`。`product_id` 和引用起点继续由后端从 State 注入；指定商品时仍使用 Metadata Filter，未指定时搜索全库。

### 旧会话如何继续

SQLite 中已经存在的 AI 工具调用及工具结果可能仍叫 `retrieve_docs`。每次交给模型前，节点复制相关消息，统一标准工具调用、服务商原始调用字段及工具消息中的名称；原始数据库记录保留，引用、调用 ID 和参数继续使用。

若旧检查点还有待执行的 `retrieve_docs` 调用，工具节点先转为新名称，再执行新工具，商品范围仍覆盖为 State 的值。旧名称仅用于兼容，不额外绑定给模型，也没有保留旧 Python 工具入口。

因为提示词现在要求新工具名，`PROMPT_VERSION` 升级到 `product_prompt_v2`，避免命中第一步的 v1 缓存。会话线程和数据库路径保持不变，旧会话可以继续使用。

### 怎么验证

新增 5 项离线测试，覆盖只绑定新名、旧待执行调用的商品过滤、SQLite 旧检查点恢复后两个模型阶段的名称转换，以及商品与全库的 v1 缓存隔离。完整离线回归 **188 项通过、4 项真实服务测试默认跳过**；相关代码的 Ruff、格式和差异空白检查通过。

单独运行商品切换和商品提示词两项真实测试：**2 项全部通过**，使用真实聊天模型、本地 BGE、Chroma、Redis 和 SQLite：

- 图状态中的真实工具调用均为 `retrieve_product_docs`，Pro、普通版、Max 的召回及引用只来自对应文件。
- Pro 与 Max 回答最高 50W 无线充电，普通版回答不支持；切回 Pro 恢复原上下文。
- 未提供的 L2 容量明确回答“暂未查询到相关信息”，无引用；用户备注能按历史正确回答。
- 身份介绍不调用商品工具，测试仅清理自己创建的随机缓存、Redis 历史和 SQLite 线程。

至此，圣火文档第四周要求的“指定商品 → Agent 调用正式商品工具 → 按原文回答”和“资料没有 → 明确说明未查询到”两个后端 Demo 均已验收。更广泛的回答质量仍需继续评测。

## 自己怎样测试

先运行不需要外部聊天服务的测试：

```bash
.venv/bin/python -m pytest tests/test_prompts.py tests/test_product_filter.py -v
```

聊天 API、本地 BGE、Redis 和商品库配置好后，运行真实验收：

```bash
RUN_AGENT_INTEGRATION=1 .venv/bin/python -m pytest \
  tests/test_agent.py::test_agent_filters_current_product_for_implicit_question \
  tests/test_agent.py::test_agent_product_prompt_handles_identity_facts_unknown_and_memory -q -s
```

已运行的 FastAPI 需重启才能加载新代码。在自己的服务终端按 `Ctrl+C`，重新启动：

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/docs>，使用 `POST /query`，例如：

```json
{
  "session_id": "week4-manual-01",
  "product_id": "vivo_x500_pro",
  "message": "这款支持无线充电吗，最高功率是多少？"
}
```

保持同一商品和会话，依次修改问题：

| 问题 | 预期行为 |
| --- | --- |
| 你是谁？ | 介绍 Flame 商品 AI 助手；无需资料引用 |
| 这款支持无线充电吗，最高功率是多少？ | 根据演示资料回答最高 50W，引用来源为 `x500_pro.md` |
| 这款手机处理器的 L2 缓存容量具体是多少？ | 明确说明暂未查询到，不猜一个容量 |
| 我的备注代号是 NOTE-2468，请记住 | 根据用户消息记录备注 |
| 我刚才设定的备注代号是什么？ | 回答 `NOTE-2468`，不添加商品资料引用 |

若要确认实际工具名，使用一个新的 `session_id` 发起参数问题，再在项目 Python 环境读取同一 SQLite 检查点：

```python
from langchain_core.messages import ToolMessage
from backend.retrieval import agent
from backend.sessions import make_thread_config

state = agent.get_state(
    make_thread_config("week4-manual-01", "vivo_x500_pro")
).values
print([m.name for m in state["messages"] if isinstance(m, ToolMessage)])
```

新版对话应显示 `retrieve_product_docs`。已有会话的原始历史可能仍显示旧名称，模型输入会自动转换，无需删除历史。

## 当前进度与边界

| 第四周任务 | 状态 |
| --- | --- |
| 第一步：商品提示词及当前商品身份 | 已完成 |
| 第二步：工具改名及商品工具描述 | 已完成 |
| 全周纯后端 Demo 验收 | 两项真实测试通过，达到商品 RAG V0.1 后端节点 |

缺少资料的回答在本步由提示词约束；本轮代表性问题通过不等于所有问题都完成了人工质量评测。后续继续扩充评测问题。当前商品资料仍是演示数据。
