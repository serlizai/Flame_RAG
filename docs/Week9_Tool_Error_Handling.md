# 第九周第一步：统一工具异常兜底

更新日期：2026-11-04。对应旧版[圣火计划](../圣火文档.md)第九周的“Tool 异常 → 返回友好错误”，不包含新版企业部署需求。

## 已实现的行为

正式 Agent 的四类工具共用 `backend/tool_execution.py` 中的调用边界。未知工具、参数校验错误、检索或业务工具抛出的运行异常，以及无法使用的工具返回格式，均转为 `status="error"` 的 `ToolMessage`。保留原调用 ID，以便模型消息与工具消息对应；失败消息不携带引用。

| 场景 | 返回行为 |
| --- | --- |
| 未知工具 | 提示暂不支持本次查询方式，请重新描述商品问题 |
| 工具参数校验失败 | 提示对应查询参数无效，请重新描述问题 |
| 检索连接、运行或超时异常已经抛出 | 提示商品知识检索暂不可用，请稍后重试 |
| 商品、价格或售后业务数据读取失败 | 对应查询返回错误状态和友好提示 |
| 工具返回错误状态或损坏的引用结构 | 清除失败输出和引用，统一返回友好提示 |
| 正常检索为空、商品未录入业务资料、指定规格无报价 | 保留正常的无资料结果，不误报为工具运行故障 |

同批其他工具仍会执行，成功资料保留在图状态中，失败调用不占用引用编号。本轮只要有工具失败，就直接返回去重后的失败提示，不再调用回答模型，不用旧轮或同批其他工具的结果补成完整答案，也不返回本轮商品引用。

错误消息继续保存在本商品会话内，以便后续重试；判断失败时只检查当前轮，不让上一轮故障阻止恢复后的查询。故障答案不写共享答案缓存，正常新会话仍按原规则缓存成功答案。

程序日志仅记录受控的工具标签和异常类型，不记录调用参数、底层异常文本或堆栈。`KeyboardInterrupt` 等进程退出信号继续上抛。业务工具通过 LangChain 的 `ToolException` 与 `handle_tool_error` 标记失败，独立调用也保留友好提示。

正式 `/api/chat` 与兼容 `/query` 响应字段保持不变。工具故障作为可展示的聊天回答返回 HTTP 200，包含 `answer`、空 `citations`、原 `session_id` 和当前 `product_id`。未知商品仍走原有的 API 404 校验。

提示词版本升级为 `product_prompt_v5`，补充失败消息不能作为商品事实依据的规则，并使旧版本答案缓存失效。

## 修改文件

| 文件 | 职责 |
| --- | --- |
| `backend/tool_execution.py` | 统一调用、错误状态、友好提示和失败引用清理 |
| `backend/nodes.py` | 接入调用边界，本轮失败后停止工具循环及模型生成 |
| `backend/business_tools.py` | 业务读取故障使用错误状态，区分正常无资料 |
| `backend/retrieval.py` | 阻止故障答案进入共享缓存 |
| `prompts.py` | 补充错误依据规则并升级缓存版本 |
| `tests/test_product_filter.py`、`tests/test_business_tools.py` | 图执行、HTTP 入口、恢复、引用和缓存回归 |

## 验证结果

- 完整后端回归：**431 项通过，15 项按外部服务条件跳过**。
- 其中本次净增 25 个测试用例，覆盖未知工具、无效参数、检索异常、业务文件故障、错误输出、同批调用、旧报价隔离、故障恢复、缓存及新旧 HTTP 入口。
- 修改的 Python 文件通过 Ruff 检查、格式检查；Git 差异空白检查通过。

测试使用正式 LangGraph 工作流、临时 Chroma 和 FastAPI TestClient，模型、Redis 和检查点使用离线替代，不访问真实聊天服务；这次未运行真实模型验收。

在项目根目录复现：

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check backend/tool_execution.py backend/nodes.py backend/business_tools.py backend/retrieval.py prompts.py tests/test_product_filter.py tests/test_business_tools.py
.venv/bin/ruff format --check backend/tool_execution.py backend/nodes.py backend/business_tools.py backend/retrieval.py prompts.py tests/test_product_filter.py tests/test_business_tools.py
git diff --check
```

## 范围与剩余工作

本次完成已进入工具执行节点的异常兜底。它接住工具已经抛出的超时异常，不新增强制执行超时或自动重试机制。模型调用、Redis、SQLite 自身在工具节点之外的故障不属于本次改动。

后续已完成[业务范围边界](Week9_Business_Scope.md)及[常见提示词注入防护与验收](Week9_Prompt_Injection.md)。第九周仍需集中进行统一验收，以及更全面的无资料和复杂业务问题检查。
