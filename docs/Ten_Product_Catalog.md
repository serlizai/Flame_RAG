# 十商品跨品类知识库开发记录

更新日期：2026-10-16。本次在第四周商品 RAG 后端基础上扩充数据，继续沿用现有 `/query` 和 `retrieve_product_docs` 链路。

## 商品清单

保留原三款手机，新增七款生活用品，合计 10 个商品、6 个分类。所有参数、价格、售后均为明确标注的虚构演示数据。

| 商品 | product_id | SKU | category | 文档 | Chunk 数 |
| --- | --- | --- | --- | --- | --- |
| vivo X500 | `vivo_x500` | `100000` | `smartphone` | `x500.md` | 7 |
| vivo X500 Pro | `vivo_x500_pro` | `100001` | `smartphone` | `x500_pro.md` | 7 |
| vivo X500 Pro Max | `vivo_x500_pro_max` | `100002` | `smartphone` | `x500_pro_max.md` | 8 |
| Flame K1 电热水壶 | `flame_kettle_k1` | `200001` | `kitchen_appliance` | `kettle_k1.md` | 7 |
| Flame A1 空气炸锅 | `flame_air_fryer_a1` | `200002` | `kitchen_appliance` | `air_fryer_a1.md` | 7 |
| Flame R1 电饭煲 | `flame_rice_cooker_r1` | `200003` | `kitchen_appliance` | `rice_cooker_r1.md` | 7 |
| Flame L1 LED 台灯 | `flame_desk_lamp_l1` | `200004` | `lighting` | `desk_lamp_l1.md` | 7 |
| Flame F1 空气循环扇 | `flame_circulation_fan_f1` | `200005` | `home_appliance` | `circulation_fan_f1.md` | 7 |
| Flame T1 不锈钢保温杯 | `flame_thermos_t1` | `200006` | `drinkware` | `thermos_t1.md` | 7 |
| Flame S1 折叠收纳箱 | `flame_storage_box_s1` | `200007` | `home_storage` | `storage_box_s1.md` | 7 |
| 合计 | — | — | — | 10 个文档 | 71 |

每个文档仍包含商品简介、核心参数、核心卖点、使用说明、FAQ、售后说明。新增商品使用独立的演示品牌与型号，不套用真实品牌的未核验参数。

## 怎么开发的

1. 扩充 `data/products.json`，每项保存 `id`、`sku`、`name`、`category`；保留三款手机原有编号。`backend/products.py` 增加分类读取和非空字符串校验。
2. 在 `data/products/` 增加七个 Markdown，并为全部文档头部加入 `category:`。商品表负责身份校验，Markdown 负责参数、知识和售后原文。
3. 修改 `backend/product_documents.py`，默认从各文档头部读取分类，再随 Metadata 继承到每个 Chunk；分类缺失的旧文档继续使用 `smartphone`。拒绝空分类或重复分类字段。
4. 两个建库/导出脚本移除统一的手机默认分类。`--category` 仍可显式覆盖全部商品，混合商品库日常建库时省略该参数。
5. 切分仍为按二级章节分组，再以 800/100 切分。真实本地 BGE-M3 生成 1024 维向量，Chroma `products` 集合更新为 71 个块，成功写入后删除 3 个旧手机头部块。
6. 同步导出 `data/product_chunks.jsonl`。逐条回读数据库，核对文本、稳定 ID、商品 ID、SKU、名称、分类、写入标记和向量维度，与源文档一致。
7. Redis 答案缓存命名空间升级为 `flame:llm_cache_v3:<提示词版本>:` 和 `flame:llm_cache_product_v4:<提示词版本>:`，避免扩充后复用旧商品集合的首轮答案。既有聊天历史和 SQLite 会话继续保存。

API、State 和检索工具原本按商品 ID 工作，可以直接接入新增商品。检索范围仍由后端 `product_id` 过滤；`category` 作为商品分类保留，当前没有增加分类筛选接口。

## 验证

- 离线回归：206 项通过，5 项真实服务测试默认跳过。覆盖十商品身份/分类一致、持久化、API 接受所有商品、生活用品检索过滤、空分类和重复分类拒绝。
- 真实 BGE 检索：11 个问题覆盖全部十商品；全库检索 **11/11 通过**，按商品过滤 **11/11 通过**。章节命中仍单独诊断，这些结果只验证基础商品召回。
- 实际数据库：10 个商品、71 个 Chunk，每个向量 1024 维，分类分布为手机 3、厨房电器 3、照明 1、家用电器 1、杯具 1、收纳用品 1。
- 真实 HTTP 问答：新增集成测试 **1 项通过，包含 3 次 POST 请求，均返回 200**。水壶回答 1.7L、1800W；保温杯回答 500mL、316L；收纳箱回答 45L、单箱不超过 10kg，引用均只来自对应商品文件。Redis 保存本次 6 条问答消息，结束后已清理测试数据。
- 修改代码的 Ruff 和格式检查通过。

全库检索中，原手机问题的“核心卖点”和“核心参数”专门章节仍未进入前五条；开启商品过滤后均命中。基础商品归属通过不代表所有字段或最终答案均完成质量评测。

真实生活用品 HTTP 验收命令：

```bash
RUN_AGENT_INTEGRATION=1 .venv/bin/python -m pytest \
  tests/test_agent.py::test_api_answers_household_product_questions -q -s
```

此测试通过 FastAPI TestClient 的真实路由进入 Agent，实际调用聊天服务、BGE、Chroma、Redis 和 SQLite；不使用正在运行的 8000 端口。它分别检查水壶、保温杯、收纳箱的参数和引用范围，结束时只清理测试创建的随机缓存、历史和检查点。

## 自己怎么用

已经运行的 FastAPI 或 Streamlit 需要在自己的服务终端重启，加载新的缓存命名空间和代码。FastAPI 启动命令：

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

在 `/docs` 使用 `POST /query`：

```json
{
  "session_id": "household-demo-01",
  "product_id": "flame_kettle_k1",
  "message": "这款的容量和额定功率是多少？"
}
```

预期回答：1.7L、1800W，引用来源为 `kettle_k1.md`。同一 `session_id` 改成 `flame_thermos_t1`，问容量和内胆材质，应回答 500mL、316L 不锈钢，引用只来自 `thermos_t1.md`。

重新建库或检索：

```bash
.venv/bin/python -m scripts.build_product_db --dry-run
.venv/bin/python -m scripts.build_product_db
.venv/bin/python -m scripts.prepare_product_chunks
.venv/bin/python -m scripts.test_retrieval
.venv/bin/python -m scripts.test_retrieval --filter-products
```

以后增加商品时，同步更新 JSON 和 Markdown 的身份/分类，再运行建库命令及回归检查。向量库、JSONL 和检查点是本地产物，继续由 Git 忽略；商品表、商品文档、代码和开发记录可纳入版本控制。
