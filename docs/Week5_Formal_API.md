# 第五周：FastAPI 正式接口

更新日期：2026-10-16。按[圣火文档](../圣火文档.md)第五周列出的接口顺序逐步实现。

本周四个正式接口均已完成。最新离线回归 254 项通过，真实 Uvicorn TCP 验收 1 项通过；下文各步骤保留当时测试数字。

## 第一步：商品列表（已完成）

新增：

```text
GET /api/products
```

用途：前端先获取所有商品名称和编号，让用户选择当前商品，再将选中项的 `id` 作为后续问答的 `product_id`。当前商品表包含十个商品、六个分类。

### 怎么开发的

| 文件 | 修改内容 |
| --- | --- |
| `backend/main.py` | 商品列表路由、响应模型、统一商品表读取与错误处理 |
| `tests/test_product_api.py` | 列表、异常、即时更新、OpenAPI 和独立读取测试 |
| `README.md`、开发记录及项目说明 | 接口使用方法和阶段状态 |

复用 `backend/products.py` 的 `load_products()`，每次请求读取 `data/products.json`，沿用非空字段、ID/SKU 唯一性和分类校验。按 JSON 中的顺序返回，无需重新建库。

`ProductResponse` 限定返回四个字段，并使用数据类属性生成响应。统一的 `_load_product_catalog()` 将商品表读取异常转换为通用 HTTP 500，问答接口的商品校验也使用此函数。

商品列表只读基础信息；参数、FAQ 和售后原文继续保存在 Markdown 中。商品列表不加载 `backend.config` 或 Agent，不需要聊天 API、BGE、Chroma、Redis 或 SQLite 检查点。

### 请求与响应

该接口无需请求体。成功返回 HTTP 200，响应是 JSON 数组。下面是数组中一项的示例：

```json
{
  "id": "flame_kettle_k1",
  "sku": "200001",
  "name": "Flame K1 电热水壶",
  "category": "kitchen_appliance"
}
```

当前完整响应包含十项，SKU 始终是字符串。商品表缺失、JSON 损坏或未通过校验时，返回 HTTP 500：

```json
{"detail": "商品表配置错误。"}
```

前端得到列表后，例如选择电热水壶，当前可以向 `POST /query` 提交：

```json
{
  "product_id": "flame_kettle_k1",
  "message": "这款的容量和额定功率是多少？",
  "session_id": "week5-demo-01"
}
```

列表里的字段叫 `id`，问答输入里的字段叫 `product_id`，两者传递同一个值。

### 怎么验证的

- 新增 7 项离线测试：十商品响应；缺失、损坏、空商品表三种异常；SKU 前导零及文件更新；OpenAPI；新进程禁止导入重依赖的读取验证。
- 完整回归：**213 项通过、5 项真实服务测试默认跳过**，旧问答入口及商品校验继续通过。
- 修改代码的 Ruff 和格式检查通过。
- 在随机本地端口临时启动真实 Uvicorn，用 HTTP 获取 `/api/products` 和 `/openapi.json`。列表返回 HTTP 200，与 JSON 商品表一致；测试时聊天密钥为空、模型路径不存在、Redis 地址不可用，仍正常读取。结束后关闭临时进程。

本步验证的是正式商品列表，不需要重新调用聊天服务或生成向量。

### 自己怎么测试

在项目根目录启动 FastAPI。已有服务时，在自己的终端按 `Ctrl+C` 后重新启动，使新路由生效：

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

直接用浏览器打开 <http://127.0.0.1:8000/api/products>，或运行：

```bash
curl -i 'http://127.0.0.1:8000/api/products'
```

预期为 HTTP 200，包含三款手机和七款生活用品。也可在 <http://127.0.0.1:8000/docs> 展开“商品”分组，执行 `GET /api/products`。

自动测试：

```bash
.venv/bin/python -m pytest tests/test_product_api.py tests/test_api.py -q
```

## 第二步：单商品基础信息（已完成）

新增：

```text
GET /api/products/{product_id}
```

用途：前端选中商品后，按列表中的 `id` 查询这一项的基础信息，用于展示名称、SKU 和分类。返回单个 JSON 对象；参数、FAQ 和售后原文继续通过问答链路检索。

### 怎么开发的

- `backend/main.py` 新增 `get_product()` 路由，复用列表的 `ProductResponse`，四个字段的含义及 SKU 字符串格式保持一致。
- 新增 `_get_product()`，统一按 ID 读取商品及 404 处理；现有问答的商品身份校验也复用此函数。
- 路径参数使用 `NonEmptyString`，清除首尾空白并拒绝空白 ID；配置异常继续通过 `_load_product_catalog()` 转为通用 500 响应。
- `tests/test_product_api.py` 扩展十商品详情、未知 ID、误用 SKU、空白 ID 和配置异常测试；已有测试同时检查详情的 SKU 前导零、即时更新、OpenAPI 及新进程轻量读取。

接口仍直接读取 `data/products.json`，无需加载模型、Redis 或向量库，无需重新建库。请求里的路径值使用商品 `id`，不能用 SKU 替代。

### 请求与响应

```bash
curl -i 'http://127.0.0.1:8000/api/products/flame_kettle_k1'
```

HTTP 200 返回：

```json
{
  "id": "flame_kettle_k1",
  "sku": "200001",
  "name": "Flame K1 电热水壶",
  "category": "kitchen_appliance"
}
```

| 情况 | 状态码 | 响应 |
| --- | --- | --- |
| 商品存在 | 200 | 商品基础信息对象 |
| 不存在的商品 ID 或误用 SKU | 404 | `{"detail":"当前商品不存在。"}` |
| 路径中的 ID 只有空白 | 422 | FastAPI 参数校验错误 |
| 商品表缺失、损坏或校验失败 | 500 | `{"detail":"商品表配置错误。"}` |

### 怎么验证的

完整回归 **229 项通过、5 项真实服务测试默认跳过**。本步增加 16 个测试用例，并扩展原有更新、OpenAPI 和新进程测试；修改代码的 Ruff 和格式检查通过。

另在随机本地端口启动临时 Uvicorn，以真实 HTTP 检查列表、十个商品详情、未知商品、空白 ID 和 OpenAPI：列表及十个商品详情均返回 200，字段与 JSON 一致；未知 ID 返回 404，空白 ID 返回 422。在模型路径无效、聊天密钥为空、Redis 不可用的配置下仍正常读取；临时服务已关闭。

### 自己怎么测试

在自己的服务终端重启 FastAPI 后，浏览器打开 <http://127.0.0.1:8000/api/products/flame_kettle_k1>，或在 `/docs` 的“商品”分组执行新接口。

先获取列表，再把任意一项的 `id` 放到详情路径中。额外检查：

```bash
curl -i 'http://127.0.0.1:8000/api/products/unknown-product'
```

预期返回 404 和“当前商品不存在。”。自动测试仍使用：

```bash
.venv/bin/python -m pytest tests/test_product_api.py tests/test_api.py -q
```

## 第三步：正式聊天（已完成）

新增 `POST /api/chat`，复用 `QueryRequest`、`QueryResponse` 和 `_answer_request()`。输入商品校验通过后，再调用原有 `agent_invoke()`；商品过滤、引用、Redis 历史、首轮缓存和 SQLite 商品线程继续由原链路处理。

`GET /query`、`POST /query` 保留兼容，共用相同的会话和商品线程，不为正式入口另建一套历史。

请求示例：

```bash
curl 'http://127.0.0.1:8000/api/chat' \
  -H 'Content-Type: application/json' \
  --data '{"product_id":"flame_kettle_k1","message":"这款容量是多少？","session_id":"week5-demo-01"}'
```

响应字段：

| 字段 | 含义 |
| --- | --- |
| `answer` | Agent 最终回答 |
| `citations` | 引用编号、标签、原文和来源 |
| `session_id` | 会话编号，后续请求传回同一编号 |
| `product_id` | 本轮商品 ID，未指定时为 `null` |

`message` 为必填非空字符串，其他两项可省略或传 `null`；省略会话编号时由 Agent 自动生成，未选商品时进入全库模式。字符串清除首尾空白，拼写错误的额外字段不被接受。商品不存在返回 404，输入非法返回 422，商品表配置错误返回 500。

继续提问时保持 `session_id` 和 `product_id` 相同。可先用旧 `/query` 记录用户备注，再用 `/api/chat` 追问；两个入口使用同一份商品上下文。

## 第四步：HTTP 存活检查（已完成）

新增 `GET /api/health`：

```bash
curl 'http://127.0.0.1:8000/api/health'
```

HTTP 200 返回：

```json
{"status":"ok","service":"flame"}
```

使用 `HealthResponse` 固定响应字段及取值。该接口只证明 FastAPI 进程能够响应 HTTP，不读取商品表、不加载模型，也不探测 Redis、Chroma 或检查点文件。依赖故障时仍可能返回 200；聊天链路是否可用需要实际问答验收。

## 第三、四步如何开发和测试

| 文件 | 修改内容 |
| --- | --- |
| `backend/main.py` | 正式聊天路由、存活检查路由与响应模型，复用既有问答入口 |
| `tests/test_api.py` | 新旧 POST 入口参数化测试、全库模式、404/422/500、健康检查和 OpenAPI |
| `tests/test_product_api.py` | 新进程禁止重依赖时仍能探活，非法聊天请求在模型加载前被拒绝 |
| `tests/test_formal_api_integration.py` | 随机端口真实 HTTP 验收，独立检查点与随机 Redis 会话 |
| `README.md` 及开发记录 | 正式接口使用、健康检查边界和测试方法 |

离线回归为 **254 项通过、6 项真实服务测试默认跳过**；本步扩充 25 个离线用例并增加一项可选真实 TCP 验收。修改代码的 Ruff 和格式检查通过。

真实 TCP 验收已单独执行，**1 项通过，用时约 55 秒**：水壶回答 1.7L、保温杯回答 500mL，引用只来自各自文档；新会话缓存返回与首轮一致，SQLite 中模型调用计数为 0；旧 `/query` 记录的随机备注由新 `/api/chat` 正确恢复，Redis 共保存本次十条消息。

离线测试命令：

```bash
.venv/bin/python -m pytest tests/test_api.py tests/test_product_api.py -q
```

真实验收需要配置好聊天服务、Redis 和本地商品库：

```bash
RUN_AGENT_INTEGRATION=1 .venv/bin/python -m pytest \
  tests/test_formal_api_integration.py -q -s
```

该测试临时启动随机端口的 Uvicorn，先检查四个正式接口及非法输入，再实际进行五次问答：首轮商品检索、新会话缓存命中、旧入口记录备注、新入口追问备注、切换到保温杯。检查缓存命中线程的模型调用计数为 0、引用只属于当前商品，并核对 Redis 共保存十条问答消息。检查点使用独立临时文件；结束后关闭测试进程，只删除本次随机缓存和聊天记录。

## 第五周进度

| 顺序 | 接口 | 状态 |
| --- | --- | --- |
| 1 | `GET /api/products` | 已完成，返回商品列表 |
| 2 | `GET /api/products/{product_id}` | 已完成，返回单商品基础信息 |
| 3 | `POST /api/chat` | 已完成，复用 Agent 问答 |
| 4 | `GET /api/health` | 已完成，轻量 HTTP 存活检查 |

第五周四个接口及本轮普通 HTTP 验收已完成。下一阶段按第六周拆分业务工具；SSE 按后续周计划推进。
