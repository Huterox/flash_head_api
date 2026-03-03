"""数据库 CRUD 操作（网关侧）"""
from typing import Optional
from datetime import timezone, timedelta
from sqlalchemy import select, func, desc
from once_gateway.state.db_engine import get_sync_session
from once_gateway.state.db_models import Task

_SHANGHAI_TZ = timezone(timedelta(hours=8))


def _fmt_time(dt) -> Optional[str]:
    """将 naive datetime 标记为上海时区后输出 ISO 格式"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_SHANGHAI_TZ)
    return dt.isoformat()


class TaskDB:
    @staticmethod
    def list_tasks(node_id: str = None, status: str = None, page: int = 1, page_size: int = 20) -> dict:
        """分页查询任务列表，node_id 可选（不传则查全部节点）"""
        with get_sync_session() as session:
            base = select(Task)
            count_base = select(func.count(Task.task_id))

            if node_id:
                base = base.where(Task.node_id == node_id)
                count_base = count_base.where(Task.node_id == node_id)
            if status:
                base = base.where(Task.status == status)
                count_base = count_base.where(Task.status == status)

            total = session.execute(count_base).scalar() or 0
            stmt = base.order_by(desc(Task.created_at)).offset((page - 1) * page_size).limit(page_size)
            tasks = session.execute(stmt).scalars().all()

            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "items": [{
                    "task_id": t.task_id, "node_id": t.node_id,
                    "status": t.status, "config": t.config,
                    "result": t.result, "error_message": t.error_message,
                    "created_at": _fmt_time(t.created_at),
                    "updated_at": _fmt_time(t.updated_at),
                } for t in tasks],
            }

    @staticmethod
    def get_task(task_id: str) -> Optional[dict]:
        with get_sync_session() as session:
            task = session.get(Task, task_id)
            if not task:
                return None
            return {
                "task_id": task.task_id, "node_id": task.node_id,
                "status": task.status, "config": task.config,
                "result": task.result, "error_message": task.error_message,
                "created_at": _fmt_time(task.created_at),
                "updated_at": _fmt_time(task.updated_at),
            }

    @staticmethod
    def get_stats() -> dict:
        """集群任务统计"""
        with get_sync_session() as session:
            total = session.execute(select(func.count(Task.task_id))).scalar() or 0
            rows = session.execute(
                select(Task.status, func.count(Task.task_id)).group_by(Task.status)
            ).all()
            by_status = {r[0]: r[1] for r in rows}
            return {
                "total": total,
                "pending": by_status.get("pending", 0),
                "running": by_status.get("running", 0),
                "completed": by_status.get("completed", 0),
                "failed": by_status.get("failed", 0),
            }
