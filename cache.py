"""封装 Redis 问答缓存和聊天历史存储。"""

import redis
import hashlib
from langchain_community.chat_message_histories import RedisChatMessageHistory

CACHE_KEY_PREFIX = "flame:llm_cache_v1:"


class RedisCache:
    """通过 Redis 保存模型回答缓存与聊天历史。

    使用 LangChain 的 RedisChatMessageHistory 按会话持久化聊天记录。

    属性：
        redis_client：连接指定 Redis 服务的客户端。

    参数：
        redis_url：Redis 连接地址，例如 redis://localhost:6379/0。

    方法：
        make_cache_key(query, session_id)：按问题与会话生成 SHA-256 缓存键。
        get(key)：读取缓存字符串；键不存在时返回 None。
        set(key, value, ttl)：写入缓存，可设置以秒计的过期时间。
        get_chat_history(session_id)：获取指定会话的聊天历史对象。

    示例：
        >>> cache = RedisCache("redis://localhost:6379/0")
        >>> key = cache.make_cache_key("什么是人工智能？", "user123")
        >>> cache.set(key, "人工智能是一类计算机技术。")
        >>> cache.get(key)
        '人工智能是一类计算机技术。'

    说明：
        缓存键使用 SHA-256 摘要，支持长期保存或指定 TTL。
        聊天历史使用独立的 Flame 键前缀。
    """

    def __init__(self, redis_url):
        self.redis_url = redis_url
        self.redis_client = redis.Redis.from_url(redis_url)

    def make_cache_key(self, query, session_id):
        key_raw = f"{session_id}:{query}"
        return CACHE_KEY_PREFIX + hashlib.sha256(key_raw.encode()).hexdigest()

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
