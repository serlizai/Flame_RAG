"""提供 Agent 后端使用的 Redis 缓存接口。"""

import hashlib

from cache import RedisCache as _SessionScopedRedisCache
from cache import CACHE_KEY_PREFIX


class RedisCache(_SessionScopedRedisCache):
    """供 Agent 后端使用的 Redis 缓存，仅按问题文本生成缓存键。

    继承 cache.RedisCache 的连接管理、读写和聊天历史接口。
    make_cache_key 不包含 session_id，因此相同问题可以跨会话命中缓存。
    基础类则按 session_id 与问题文本共同区分缓存。

    属性：
        redis_client：连接指定 Redis 服务的客户端。

    参数：
        redis_url：Redis 连接地址，例如 redis://localhost:6379/0。

    方法：
        make_cache_key(query)：生成与会话无关的 SHA-256 缓存键。
        get(key)：读取缓存，缺失时返回 None。
        set(key, value, ttl)：写入缓存，可指定以秒计的过期时间。
        get_chat_history(session_id)：通过 RedisChatMessageHistory 访问会话历史。

    说明：
        聊天历史按会话保存；问答缓存按问题共享，两者的隔离范围不同。
    """

    def make_cache_key(self, query):
        return CACHE_KEY_PREFIX + hashlib.sha256(query.encode()).hexdigest()
