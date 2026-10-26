# Windows 本机运行环境

配置日期：2026-10-22。项目目录：`F:\shkj\Flame_RAG`。

## 当前状态

- 已使用本机 `D:\python\python.exe`（Python 3.12.10）创建独立的 `.venv`，符合项目 Python >= 3.11 的要求。`.python-version` 已固定为 `3.12`；安装命令禁止下载 Python。
- 已配置本地 `.env`，BGE-M3 路径为 `F:/shkj/Flame_RAG/models/BAAI/bge-m3`，设备使用 CPU。聊天服务使用本地填写的接口、密钥及 `OPENAI_MODEL=glm-5.3-flash`。
- 已设置 VS Code 默认解释器和环境变量文件。如果编辑器已有解释器选择，请手动选择 `.venv\Scripts\python.exe`；PyCharm 也使用此路径。
- 已检查商品表和 Markdown：均为 10 个商品。
- 已按 `uv.lock` 安装 237 个当前平台依赖，`uv pip check` 兼容性检查通过；PyTorch 2.14.0+cpu、Chroma 1.5.9、Sentence Transformers 6.1.0 均可导入。
- BGE-M3 已从 [BAAI 官方模型仓库](https://huggingface.co/BAAI/bge-m3/tree/main)下载，商品库已写入 71 个 1024 维向量。真实全库检索和商品过滤检索均为 11/11 通过。
- Redis 8.2.10 已安装到项目 `.cache/redis/`，使用 [redis-windows 社区便携版](https://github.com/redis-windows/redis-windows/releases/tag/8.2.10)，已核对发行版 SHA-256。仅监听 `127.0.0.1:6379`，启用密码及 AOF 持久化；认证连接、读写和拒绝未认证访问均已验证。未安装系统服务。
- 聊天 API 连接、模型工具调用和工具结果回传已通过 `scripts.check_chat_api` 验证。离线回归测试为 254 通过、6 跳过；另外启用真实依赖运行的 6 项集成测试全部通过，覆盖商品问答、引用、会话续聊、商品切换、缓存和真实 HTTP 接口。
- FastAPI 和 Streamlit 已在后台启动，本机健康检查均通过。访问 <http://127.0.0.1:8501> 使用聊天页面，访问 <http://127.0.0.1:8000/docs> 查看接口文档。

## 1. 安装锁定依赖

依赖已安装。后续需要重新同步时，在 PowerShell 中执行：

```powershell
Set-Location F:\shkj\Flame_RAG
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.cache\uv'
uv sync --locked --python D:\python\python.exe --no-python-downloads
.\.venv\Scripts\python.exe --version
```

后续命令使用 `uv run --no-sync`，直接运行已安装的环境。无需激活环境，也无需调整 PowerShell 执行策略。不要删除原有的 `venv`，本次创建的是独立的 `.venv`。

## 2. 下载 BGE-M3

模型已下载。若以后需要重新下载到 `.env` 中配置的位置，可执行（仅下载本项目使用的文本向量模型文件）：

```powershell
uv run --no-sync python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-m3', local_dir='models/BAAI/bge-m3', allow_patterns=['*.json','sentencepiece.bpe.model','pytorch_model.bin'])"
```

模型较大，下载需要联网和足够磁盘空间。若模型已放在其他目录，修改 `.env` 中的 `BGE_M3_PATH` 为实际绝对路径。

验证资料并构建商品库：

```powershell
uv run --no-sync python -m scripts.build_product_db --dry-run
uv run --no-sync python -m scripts.build_product_db --device cpu
uv run --no-sync python -m scripts.test_retrieval --device cpu
uv run --no-sync python -m scripts.test_retrieval --filter-products --device cpu
```

商品库使用 `chroma_db_products/` 的 `products` 集合，与 `.env` 的 `CHROMA_PERSIST_DIRECTORY` 不同，后者用于可选的 PDF 导入。

## 3. 配置聊天和 Redis

聊天 API 已配置。后续更换服务时，在本地 `.env` 更新 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL`；无需修改 Python 源码。模型必须支持工具调用，密钥只保留在本地。

可单独验证聊天 API，此命令发送两次简短请求，不加载 BGE：

```powershell
uv run --no-sync python -m scripts.check_chat_api
```

Redis 当前已启动。重启电脑后，在项目根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_redis.ps1
```

脚本在后台启动本项目的 Redis 并验证 `PONG`，不会修改系统执行策略，也不会创建开机自启服务。程序位于 `.cache/redis/Redis-8.2.10-Windows-x64-cygwin/`，本地配置位于 `.cache/redis/redis.conf`，持久化数据位于 `data/redis/`，日志位于 `logs/redis.log`。请保留这些本地目录。

`REDIS_URL` 已写入本地 `.env`，包含自动生成的密码。不要公开或提交 `.env` 和本地 Redis 配置。若修改 Redis 密码，需要同时更新上述两个文件。

停止本项目 Redis 时，在项目根目录执行（会保存数据）：

```powershell
uv run --no-sync python -c "from dotenv import dotenv_values; import redis; redis.Redis.from_url(dotenv_values('.env')['REDIS_URL']).shutdown(save=True)"
```

## 4. 启动与验证

本次已启动两个后台进程。重启电脑后，先按上文启动 Redis，再在项目根目录的一个终端启动 FastAPI：

```powershell
uv run --no-sync uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

在另一个 PowerShell 终端执行：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:8000/api/products
```

接口文档：<http://127.0.0.1:8000/docs>。上述接口不检查模型、Redis 或向量库。

模型、商品库、聊天配置和 Redis 均已就绪。可通过接口文档中的 `POST /api/chat` 测试，或在项目目录打开另一个终端启动聊天页面：

```powershell
uv run --no-sync streamlit run app.py --server.address 127.0.0.1 --server.headless true
```

聊天页面：<http://localhost:8501>。离线回归测试：

```powershell
uv run --no-sync pytest tests/ -q
```

需要重新执行真实服务验收时（会调用本地配置的聊天 API），在普通本机终端执行：

```powershell
$env:RUN_AGENT_INTEGRATION = '1'
try {
    uv run --no-sync pytest tests/test_agent.py tests/test_formal_api_integration.py -q
} finally {
    Remove-Item Env:RUN_AGENT_INTEGRATION -ErrorAction SilentlyContinue
}
```

后台进程的输出分别写入 `logs/api-stdout.log`、`logs/api-stderr.log`、`logs/streamlit-stdout.log` 和 `logs/streamlit-stderr.log`。需要停止后台 API 或页面时，按监听端口确认进程后停止相应进程；不要直接终止所有 Python 进程。使用上述终端命令启动时，按 `Ctrl+C` 停止。
