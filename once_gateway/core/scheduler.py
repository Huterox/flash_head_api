"""调度器：选择合适的节点（最少连接数策略）"""
import asyncio
import logging
from typing import Optional

from once_gateway.core.node_registry import NodeInfo, get_node_registry

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._counter: dict[str, int] = {}

    async def select_node(self) -> Optional[NodeInfo]:
        async with self._lock:
            registry = get_node_registry()
            candidates = await registry.healthy_nodes()
            if not candidates:
                logger.warning("无可用健康节点")
                return None
            selected = min(candidates,
                           key=lambda n: self._counter.get(n.id, 0))
            self._counter[selected.id] = self._counter.get(selected.id, 0) + 1
            return selected


_scheduler: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler()
    return _scheduler
