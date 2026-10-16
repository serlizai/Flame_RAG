"""提供 Agent 后端使用的 Redis 缓存接口。"""

import hashlib
import json

import redis
from langchain_community.chat_message_histories import RedisChatMessageHistory

from backend.prompts import PROMPT_VERSION

# 只保存无历史的首轮回答，不复用旧版本中可能依赖聊天历史的答案。
# 扩充到跨品类商品库后升级命名空间，避免复用旧商品集合生成的答案。
CACHE_KEY_PREFIX = f"flame:llm_cache_v3:{PROMPT_VERSION}:"
PRODUCT_CACHE_KEY_PREFIX = f"flame:llm_cache_product_v4:{PROMPT_VERSION}:"


class RedisCache:
    """供 Agent 后端使用的 Redis 缓存，指定商品时按商品隔离答案。

    make_cache_key 不包含会话历史，入口仅在该商品对话首轮使用此缓存。

    属性：
        redis_client：连接指定 Redis 服务的客户端。

    参数：
        redis_url：Redis 连接地址，例如 redis://localhost:6379/0。

    方法：
        make_cache_key(query, product_id)：生成与会话无关的 SHA-256 缓存键。
        get(key)：读取缓存，缺失时返回 None。
        set(key, value, ttl)：写入缓存，可指定以秒计的过期时间。
        get_chat_history(session_id)：通过 RedisChatMessageHistory 访问会话历史。

    说明：
        聊天历史按会话保存；商品问答缓存按商品和问题共享，隔离范围不同。
    """

    def __init__(self, redis_url):
        self.redis_url = redis_url
        self.redis_client = redis.Redis.from_url(redis_url)

    def make_cache_key(self, query, product_id=None):
        """生成按问题、商品区分的首轮缓存键，全库使用独立前缀。"""
        if product_id is None:
            return CACHE_KEY_PREFIX + hashlib.sha256(query.encode()).hexdigest()
        payload = json.dumps(
            {"query": query, "product_id": product_id},
            ensure_ascii=False,
            sort_keys=True,
        )
        return PRODUCT_CACHE_KEY_PREFIX + hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key):
        value = self.redis_client.get(key)
        return value.decode("utf-8") if value else None

    def set(self, key, value, ttl=None):
        if ttl:
            self.redis_client.setex(key, ttl, value)
        else:
            self.redis_client.set(key, value)

    def get_chat_history(self, session_id):
        return RedisChatMessageHistory(
            session_id=session_id, url=self.redis_url, key_prefix="flame:message_store:"
        )
