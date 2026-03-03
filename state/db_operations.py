"""数据库 CRUD 操作"""
from typing import Optional
from datetime import timezone, timedelta
from sqlalchemy import select, update
from state.db_engine import get_sync_session
from state.db_models import Task, UploadedFile
from loguru import logger

_UNSET = object()
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
    def create_task(task_id: str, node_id: str, config: dict) -> dict:
        with get_sync_session() as session:
            task = Task(task_id=task_id, node_id=node_id, status='pending', config=config)
            session.add(task)
            session.flush()
            return {"task_id": task.task_id, "status": task.status}

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
    def update_task_status(task_id: str, status: str, result=_UNSET, error_message=_UNSET):
        with get_sync_session() as session:
            values = {"status": status}
            if result is not _UNSET:
                values["result"] = result
            if error_message is not _UNSET:
                values["error_message"] = error_message
            session.execute(update(Task).where(Task.task_id == task_id).values(**values))

    @staticmethod
    def get_pending_tasks(node_id: str) -> list:
        with get_sync_session() as session:
            stmt = select(Task).where(Task.node_id == node_id, Task.status.in_(['pending', 'running']))
            tasks = session.execute(stmt).scalars().all()
            return [{"task_id": t.task_id, "status": t.status, "config": t.config} for t in tasks]

    @staticmethod
    def delete_task(task_id: str) -> bool:
        from sqlalchemy import delete as sql_delete
        with get_sync_session() as session:
            result = session.execute(sql_delete(Task).where(Task.task_id == task_id))
            return result.rowcount > 0

    @staticmethod
    def list_tasks(node_id: str, status: str = None, keyword: str = None, page: int = 1, page_size: int = 20) -> dict:
        """分页查询任务列表，支持按状态筛选和关键词搜索"""
        from sqlalchemy import func, desc
        with get_sync_session() as session:
            base = select(Task).where(Task.node_id == node_id)
            count_base = select(func.count(Task.task_id)).where(Task.node_id == node_id)
            if status:
                base = base.where(Task.status == status)
                count_base = count_base.where(Task.status == status)
            if keyword:
                base = base.where(Task.task_id.ilike(f"%{keyword}%"))
                count_base = count_base.where(Task.task_id.ilike(f"%{keyword}%"))

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


class FileDB:
    @staticmethod
    def create_file(file_id: str, filename: str, stored_path: str, file_size: int, file_type: str, expires_at=None):
        with get_sync_session() as session:
            f = UploadedFile(file_id=file_id, filename=filename, stored_path=stored_path,
                             file_size=file_size, file_type=file_type, expires_at=expires_at)
            session.add(f)

    @staticmethod
    def get_file(file_id: str) -> Optional[dict]:
        with get_sync_session() as session:
            f = session.get(UploadedFile, file_id)
            if not f:
                return None
            return {
                "file_id": f.file_id, "filename": f.filename,
                "stored_path": f.stored_path, "file_size": f.file_size,
                "file_type": f.file_type,
            }
