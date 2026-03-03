"""任务查询 API（网关直连数据库 + Redis 读进度）"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from once_gateway.utils.result import R
from once_gateway.state.db_operations import TaskDB
from once_gateway.core.node_registry import get_node_registry

router = APIRouter(prefix="/api/tasks", tags=["任务查询"])

PROGRESS_PREFIX = "flashhead:task:"


async def _get_progress(task_id: str):
    import json
    from once_gateway.utils.redis_client import get_redis
    r = await get_redis()
    data = await r.get(f"{PROGRESS_PREFIX}{task_id}:progress")
    return json.loads(data) if data else None


@router.get("/cluster/stats")
async def cluster_stats():
    """集群任务统计"""
    stats = TaskDB.get_stats()
    registry = get_node_registry()
    nodes = await registry.all_nodes()
    total_queue = sum(n.queue_size for n in nodes)
    active_nodes = sum(1 for n in nodes if n.active_task)
    stats["total_queue"] = total_queue
    stats["active_nodes"] = active_nodes
    stats["total_nodes"] = len(nodes)
    stats["healthy_nodes"] = sum(1 for n in nodes if n.status.value == "healthy")
    return JSONResponse(R.ok().data(stats))


@router.get("/list")
async def list_tasks(node_id: str = None, status: str = None, page: int = 1, page_size: int = 20):
    result = TaskDB.list_tasks(node_id=node_id, status=status, page=page, page_size=page_size)
    registry = get_node_registry()
    for item in result["items"]:
        node = await registry.get(item["node_id"])
        item["node_name"] = node.name if node else item["node_id"]
        if item["status"] in ("pending", "running"):
            item["progress"] = await _get_progress(item["task_id"])
    return JSONResponse(R.ok().data(result))


@router.get("/{task_id}")
async def get_task(task_id: str):
    task_dict = TaskDB.get_task(task_id)
    if not task_dict:
        return JSONResponse(R.error("任务不存在"))
    registry = get_node_registry()
    node = await registry.get(task_dict["node_id"])
    task_dict["node_name"] = node.name if node else task_dict["node_id"]
    task_dict["progress"] = await _get_progress(task_id)
    return JSONResponse(R.ok().data(task_dict))
