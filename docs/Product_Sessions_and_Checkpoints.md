# 商品独立会话与 SQLite 检查点

更新日期：2026-10-15。本文记录第三周完成后的多轮会话改进。

## 实现了什么

同一个前端会话中，Pro、普通版、Pro Max 和全库问答分别保存自己的 LangGraph State。切回 Pro 时继续读取原来的 Pro 对话；服务正常重启后，也能从 SQLite 找回该状态。

每轮仍提交相同格式：

```json
{
  "session_id": "demo-session",
  "product_id": "vivo_x500_pro",
  "message": "刚才我说的预算是多少？"
}
```

`session_id` 保持前端会话身份，`product_id` 选择商品；省略或传 `null` 时选择独立的全库对话。它不会自动沿用上一次商品，每轮继续传入当前商品 ID。

## 怎么开发的

| 文件 | 职责 |
| --- | --- |
| `backend/sessions.py` | 根据会话 ID 和商品 ID 生成稳定的内部 `thread_id` 与调用配置 |
| `backend/checkpoints.py` | 管理 SQLite 路径、建表、序列化和连接关闭 |
| `backend/config.py` | 用持久化 `SqliteSaver` 替换进程内 `MemorySaver` |
| `backend/retrieval.py` | 恢复对应商品的状态，仅在无历史的首轮读取和写入答案缓存 |
| `backend/state.py`、`backend/nodes.py` | 使用整个商品线程的消息与引用，移除旧的消息起点截取逻辑 |
| `backend/cache.py` | 首轮缓存使用新版本，避开旧的、可能依赖历史的答案 |
| `.env.example`、`pyproject.toml`、`requirements.txt`、`uv.lock` | 补充配置示例和 SQLite 检查点依赖 |

### 会话如何找到对应商品

`make_thread_id()` 把 `session_id`、`product_id` 序列化为 JSON 后计算 SHA-256，生成带版本前缀的内部编号。结构化输入避免字符串直接拼接时产生歧义；相同身份在重启后生成相同编号。

例如，同一个 `demo-session` 对应四个不同线程：

| 请求范围 | 模型读取的上下文 |
| --- | --- |
| `vivo_x500_pro` | 该会话之前的全部 Pro 对话 |
| `vivo_x500` | 该会话之前的全部普通版对话 |
| `vivo_x500_pro_max` | 该会话之前的全部 Max 对话 |
| `null` | 该会话之前的全库对话 |

API 返回的 `session_id` 不变，Redis 页面历史也继续按这个编号汇总。SQLite 按内部 `thread_id` 存储图状态，因此模型不会读到另一个商品的对话。

查看某商品状态时统一使用辅助函数：

```python
from backend.retrieval import agent
from backend.sessions import make_thread_config

state = agent.get_state(
    make_thread_config("demo-session", "vivo_x500_pro")
).values
print(state["product_id"])
print(len(state["messages"]))
```

### SQLite 保存什么

默认文件：`data/checkpoints.sqlite`。框架管理 `checkpoints`、`writes` 等表，保存消息、工具结果、商品 ID、引用、模型调用计数和执行进度。

文件可通过环境变量配置：

```dotenv
CHECKPOINT_SQLITE_PATH=data/checkpoints.sqlite
```

相对路径按项目根目录解析，也接受绝对路径。目录自动创建，使用 WAL 模式和 30 秒连接等待；连接允许请求线程使用，由检查点适配器管理数据库操作锁。进程退出时关闭连接，文件继续保留。

引用对象使用项目已有的 `Citation`，序列化器显式允许恢复该类型。工具消息中的引用恢复为字典时，由节点转换回引用对象。

SQLite 的默认文件及 WAL/SHM 辅助文件已被现有 `/data/*` 忽略规则覆盖，不提交聊天数据。无需重新建商品向量库。

### 为什么调整缓存

“刚才的预算是多少”在不同聊天历史中应有不同答案。当前缓存键只有商品和问题，因此采用简单明确的规则：

- 当前商品线程没有消息：允许读取、写入共享的首轮答案缓存。
- 当前商品线程已有消息：跳过缓存读取，也不把该轮答案写入共享缓存。
- 首轮命中缓存：仍把问题、回答、引用和商品 ID 保存到 SQLite，以便后续追问。

商品首轮缓存升级到 `flame:llm_cache_product_v3:`，全库首轮缓存升级到 `flame:llm_cache_v2:`。旧键不删除，按其原有过期设置处理。切回聊过的商品属于有历史的请求，会重新执行 Agent。

2026-10-16 第四周第一步进一步在上述前缀后加入 `PROMPT_VERSION`，隔离不同提示词生成的答案；第二步工具改名后升级为 `product_prompt_v2`。会话线程及首轮缓存策略不变，详见[提示词与工具记录](Week4_Product_RAG_MVP.md)。

### 三种存储的分工

| 存储 | 当前用途 |
| --- | --- |
| SQLite | 恢复 LangGraph 的完整商品对话状态 |
| Redis | 首轮答案缓存，以及按前端会话汇总的用户问题和最终回答 |
| Chroma | 商品文本块、向量、Metadata |

模型续聊直接恢复 SQLite 检查点，不从 Redis 的问答文本重新拼装 State。

## 怎么验证的

- 6 项 SQLite 与线程测试：稳定且互不混淆的线程 ID、启动目录无关的路径、关闭连接后恢复消息及引用、请求线程调用、按线程删除，以及独立 Python 进程重启后通过正式入口续聊。
- 更新商品状态和过滤测试，确认切换后恢复原线程、不同商品与全库隔离、Redis 统一记录，以及有历史时不读取或覆盖共享缓存。
- 完整离线回归：**172 项通过，3 项真实服务测试默认跳过**。修改代码的 Ruff、格式检查通过，235 个已安装包依赖兼容性检查通过。
- 三项真实 Agent 测试全部通过，实际使用聊天服务、BGE、Chroma、Redis 和 SQLite，覆盖工具引用、无历史的新会话首轮缓存、商品切换和返回原上下文。验收后只删除测试自己创建的缓存、聊天记录及 SQLite 线程。
- 真实 HTTP 重启验收：Pro 与普通版分别保存随机备注，切回 Pro 正确回答原备注；特意写入错误的共享缓存后，已有历史的追问仍返回正确答案。停止 FastAPI 再重新启动，使用相同会话 ID，两种商品各自恢复原备注。五次请求全部为 **HTTP 200**，重启前 SQLite 保留 Pro 的 8 条图消息，Redis 最终保存 10 条问答消息。验收后仅删除本轮随机缓存、Redis 历史和两个 SQLite 测试线程，数据库及服务继续保留。

依赖新增 `langgraph-checkpoint-sqlite==3.1.2`；解析器同时将其底层 `langgraph-checkpoint` 升级到兼容的 4.3.0，并加入适配器所需的依赖，已同步锁文件。

## 自己怎样测试

先同步依赖，再运行无需聊天服务的测试：

```bash
uv sync --locked --python 3.11
.venv/bin/python -m pytest \
  tests/test_checkpoints.py tests/test_agent_product_input.py \
  tests/test_product_filter.py -v
```

配置好聊天服务、BGE、Redis 和商品库后，运行真实链路测试：

```bash
RUN_AGENT_INTEGRATION=1 .venv/bin/python -m pytest tests/test_agent.py -q -s
```

启动 FastAPI：

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

按下面顺序使用同一个 `session_id`：

1. Pro：提交“我给这款商品的备注代号是 PRO-2468，请记住”。
2. 普通版：提交“我给这款商品的备注代号是 BASE-1357，请记住”。
3. 切回 Pro：提交“我刚才给这款商品的备注代号是什么？”；应回答 `PRO-2468`。
4. 用 `Ctrl+C` 停止 FastAPI，再用同一启动命令重启。
5. 继续传相同会话 ID 和 Pro 商品 ID，再次追问备注；应仍能回答 `PRO-2468`。

例如第一步请求：

```bash
curl 'http://127.0.0.1:8000/query' \
  -H 'Content-Type: application/json' \
  --data '{"session_id":"checkpoint-demo","product_id":"vivo_x500_pro","message":"我给这款商品的备注代号是 PRO-2468，请记住。"}'
```

后续请求保持 `checkpoint-demo`，按上述步骤修改商品和问题。模型测试之外，可用前面的 `get_state()` 代码检查 SQLite 是否保留原消息。删除数据库文件、改用另一条文件路径或换会话 ID，都不会读取到原来的线程。

## 当前范围

之前仅存在于 MemorySaver 的图状态无法在本次重启后自动迁移。原有 Redis 历史保留，新版首次请求开始写入 SQLite；历史文本不自动导入图状态。

Streamlit 目前仍默认全库，没有商品选择控件；通过 API 演示商品切换。用户身份校验、正式商品对比模式、长对话摘要及裁剪尚未实现。SQLite 用于当前单机轻量服务，后续多人使用、部署多个服务实例时再迁移 PostgreSQL。本文不把这些后续功能算作已经完成。

适配器和线程机制依据：[LangGraph 持久化说明](https://docs.langchain.com/oss/python/langgraph/persistence)、[官方 SQLite 适配器](https://pypi.org/project/langgraph-checkpoint-sqlite/)。
