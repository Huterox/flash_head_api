"""Redis 同步客户端，用于任务队列和进度存储"""
import json
import redis
from typing import Optional

from config import get_config
from loguru import logger


class RedisClient:
    PREFIX = "flashhead:"

    def __init__(self):
        cfg = get_config().redis
        self.client = redis.Redis(
            host=cfg.host, port=cfg.port,
            password=cfg.password or None,
            db=cfg.db, decode_responses=True, max_connections=50,
        )
        logger.info(f"Redis客户端初始化: {cfg.host}:{cfg.port}")

    def test_connection(self) -> bool:
        try:
            self.client.ping()
            logger.info("Redis连接成功")
            return True
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            return False

    def _queue_key(self, node_id: str) -> str:
        return f"{self.PREFIX}queue:{node_id}:synthesize"

    def _progress_key(self, task_id: str) -> str:
        return f"{self.PREFIX}task:{task_id}:progress"

    def push_task(self, node_id: str, task_id: str) -> bool:
        try:
            queue_key = self._queue_key(node_id)
            members = self.client.lrange(queue_key, 0, -1)
            if task_id in members:
                return True
            self.client.rpush(queue_key, task_id)
            return True
        except Exception as e:
            logger.error(f"推送任务失败: {e}")
            return False

    def pop_task(self, node_id: str) -> Optional[str]:
        try:
            return self.client.lpop(self._queue_key(node_id))
        except Exception as e:
            logger.error(f"弹出任务失败: {e}")
            return None

    def get_queue_size(self, node_id: str) -> int:
        try:
            return self.client.llen(self._queue_key(node_id))
        except Exception:
            return 0

    def set_progress(self, task_id: str, progress: dict, ttl: int = 86400):
        try:
            self.client.set(self._progress_key(task_id), json.dumps(progress), ex=ttl)
        except Exception as e:
            logger.error(f"设置进度失败: {e}")

    def get_progress(self, task_id: str) -> Optional[dict]:
        try:
            data = self.client.get(self._progress_key(task_id))
            return json.loads(data) if data else None
        except Exception:
            return None

    def delete_progress(self, task_id: str):
        try:
            self.client.delete(self._progress_key(task_id))
        except Exception as e:
            logger.error(f"删除进度失败: {e}")

    def remove_from_queue(self, node_id: str, task_id: str):
        try:
            self.client.lrem(self._queue_key(node_id), 0, task_id)
        except Exception as e:
            logger.error(f"从队列移除任务失败: {e}")

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


_redis_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client
