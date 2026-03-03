"""健康检查：后台定时任务"""
import asyncio
import logging
from typing import Optional

import httpx

from once_gateway.config import get_config
from once_gateway.core.node_registry import get_node_registry

logger = logging.getLogger(__name__)

_task: Optional[asyncio.Task] = None


async def _check_single_node(client: httpx.AsyncClient, node_id: str,
                              node_url: str, api_key: str, timeout: int):
    registry = get_node_registry()
    base = node_url.rstrip('/')
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        resp = await client.get(f"{base}/api/system/health", headers=headers, timeout=timeout)
        data = resp.json()
        if data.get("code") == 200:
            await registry.update_health(node_id, healthy=True)
            # 采集调度器信息
            try:
                sr = await client.get(f"{base}/api/system/scheduler", headers=headers, timeout=timeout)
                sd = sr.json()
                if sd.get("code") == 200:
                    await registry.update_scheduler(
                        node_id,
                        queue_size=sd["data"].get("queue_size", 0),
                        active_task=sd["data"].get("active_task")
                    )
            except Exception:
                pass
        else:
            await registry.update_health(node_id, healthy=False)
    except Exception as e:
        logger.debug("节点 %s 健康检查失败: %s", node_id, e)
        await registry.update_health(node_id, healthy=False)


async def _health_check_loop():
    cfg = get_config()
    interval = cfg.health_check.interval
    timeout = cfg.health_check.timeout

    async with httpx.AsyncClient() as client:
        while True:
            registry = get_node_registry()
            nodes = await registry.all_nodes()
            tasks = [
                _check_single_node(client, n.id, n.url, n.api_key, timeout)
                for n in nodes
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(interval)


def start_health_checker():
    global _task
    if _task is None or _task.done():
        _task = asyncio.create_task(_health_check_loop())
        logger.info("健康检查后台任务已启动")


def stop_health_checker():
    global _task
    if _task and not _task.done():
        _task.cancel()
        _task = None
        logger.info("健康检查后台任务已停止")
