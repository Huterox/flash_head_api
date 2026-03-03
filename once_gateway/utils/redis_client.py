"""Redis 异步客户端封装"""
import logging
from typing import Optional

import redis.asyncio as aioredis

from once_gateway.config import get_config

logger = logging.getLogger(__name__)

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        cfg = get_config().redis
        _redis = aioredis.Redis(
            host=cfg.host, port=cfg.port,
            password=cfg.password or None,
            db=cfg.db, decode_responses=True,
        )
        await _redis.ping()
        logger.info("Redis 连接成功: %s:%d db=%d", cfg.host, cfg.port, cfg.db)
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None
        logger.info("Redis 连接已关闭")
