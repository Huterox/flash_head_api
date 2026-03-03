"""节点注册表：基于 Redis Hash 存储，管理所有节点的注册信息和实时状态"""
import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel
from once_gateway.config.schema import NodeConfig

logger = logging.getLogger(__name__)

NODE_HASH = "flashhead_gateway:nodes"
SHANGHAI_TZ = timezone(timedelta(hours=8))


class NodeStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class NodeInfo(BaseModel):
    id: str
    url: str
    api_key: str = ""
    name: str = ""
    status: NodeStatus = NodeStatus.OFFLINE
    last_health_check: Optional[str] = None
    fail_count: int = 0
    queue_size: int = 0
    active_task: Optional[str] = None


class NodeRegistry:
    """节点注册表，基于 Redis Hash 存储"""

    def __init__(self):
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            from once_gateway.utils.redis_client import get_redis
            self._redis = await get_redis()
        return self._redis

    async def register(self, node_cfg: NodeConfig) -> NodeInfo:
        r = await self._get_redis()
        node = NodeInfo(id=node_cfg.id, url=node_cfg.url,
                        api_key=node_cfg.api_key, name=node_cfg.name)
        await r.hset(NODE_HASH, node.id, node.model_dump_json())
        logger.info("节点已注册: %s (%s)", node.id, node.url)
        return node

    async def unregister(self, node_id: str) -> Optional[NodeInfo]:
        r = await self._get_redis()
        data = await r.hget(NODE_HASH, node_id)
        if not data:
            return None
        await r.hdel(NODE_HASH, node_id)
        logger.info("节点已移除: %s", node_id)
        return NodeInfo.model_validate_json(data)

    async def get(self, node_id: str) -> Optional[NodeInfo]:
        r = await self._get_redis()
        data = await r.hget(NODE_HASH, node_id)
        if not data:
            return None
        return NodeInfo.model_validate_json(data)

    async def save(self, node: NodeInfo) -> None:
        r = await self._get_redis()
        await r.hset(NODE_HASH, node.id, node.model_dump_json())

    async def all_nodes(self) -> List[NodeInfo]:
        r = await self._get_redis()
        all_data = await r.hgetall(NODE_HASH)
        return [NodeInfo.model_validate_json(v) for v in all_data.values()]

    async def healthy_nodes(self) -> List[NodeInfo]:
        nodes = await self.all_nodes()
        return [n for n in nodes if n.status == NodeStatus.HEALTHY]

    async def update_health(self, node_id: str, healthy: bool) -> None:
        r = await self._get_redis()
        data = await r.hget(NODE_HASH, node_id)
        if not data:
            return
        node = NodeInfo.model_validate_json(data)
        node.last_health_check = datetime.now(SHANGHAI_TZ).isoformat()
        if healthy:
            node.status = NodeStatus.HEALTHY
            node.fail_count = 0
        else:
            node.fail_count += 1
            from once_gateway.config import get_config
            threshold = get_config().health_check.unhealthy_threshold
            if node.fail_count >= threshold:
                node.status = NodeStatus.UNHEALTHY
                logger.warning("节点 %s 连续 %d 次健康检查失败，标记为 unhealthy",
                               node_id, node.fail_count)
        await r.hset(NODE_HASH, node_id, node.model_dump_json())

    async def update_scheduler(self, node_id: str, queue_size: int = 0,
                               active_task: Optional[str] = None) -> None:
        r = await self._get_redis()
        data = await r.hget(NODE_HASH, node_id)
        if not data:
            return
        node = NodeInfo.model_validate_json(data)
        node.queue_size = queue_size
        node.active_task = active_task
        await r.hset(NODE_HASH, node_id, node.model_dump_json())


_registry: Optional[NodeRegistry] = None


def get_node_registry() -> NodeRegistry:
    global _registry
    if _registry is None:
        _registry = NodeRegistry()
    return _registry
