"""单线程池任务调度器（GPU 推理串行）"""
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from config import get_config
from state.redis_client import get_redis_client
from state.db_operations import TaskDB, FileDB
from cores.pipeline_adapter import synthesize
from schema.enums import TaskStatus
from loguru import logger


class TaskScheduler:
    def __init__(self):
        cfg = get_config()
        self.node_id = cfg.node.id
        self.max_workers = cfg.server.thread_pool.max_workers
        self.out_dir = cfg.out_dir
        self.executor = None
        self.is_running = False
        self._poll_thread = None
        self._stop_event = threading.Event()
        self.active_task_id = None

    def start(self):
        if self.is_running:
            return
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="Synth-Worker")
        self.is_running = True
        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info(f"调度器启动: max_workers={self.max_workers}")

    def stop(self):
        self.is_running = False
        self._stop_event.set()
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        logger.info("调度器已停止")

    def _poll_loop(self):
        redis_client = get_redis_client()
        while not self._stop_event.is_set():
            try:
                task_id = redis_client.pop_task(self.node_id)
                if task_id:
                    self.executor.submit(self._execute_task, task_id)
                else:
                    self._stop_event.wait(timeout=1.0)
            except Exception as e:
                logger.error(f"轮询异常: {e}")
                self._stop_event.wait(timeout=2.0)

    def _execute_task(self, task_id: str):
        redis_client = get_redis_client()
        try:
            task_dict = TaskDB.get_task(task_id)
            if not task_dict:
                logger.warning(f"任务不存在: {task_id}")
                return
            if task_dict["status"] not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                return

            self.active_task_id = task_id
            TaskDB.update_task_status(task_id, TaskStatus.RUNNING)
            redis_client.set_progress(task_id, {"status": "running", "chunk": 0, "total": 0})

            config = task_dict["config"]
            image_file = FileDB.get_file(config["image_file_id"])
            audio_file = FileDB.get_file(config["audio_file_id"])
            if not image_file or not audio_file:
                raise ValueError("文件不存在")

            output_path = os.path.join(self.out_dir, task_id, f"{task_id}.mp4")

            def on_progress(current, total):
                redis_client.set_progress(task_id, {
                    "status": "running", "chunk": current, "total": total,
                    "percent": round(current / total * 100, 1) if total > 0 else 0,
                })

            result_path = synthesize(
                image_path=image_file["stored_path"],
                audio_path=audio_file["stored_path"],
                output_path=output_path,
                crop_region=config.get("crop_region"),
                progress_callback=on_progress,
            )

            TaskDB.update_task_status(task_id, TaskStatus.COMPLETED, result={"video_path": result_path})
            redis_client.set_progress(task_id, {"status": "completed", "percent": 100})
            logger.info(f"任务完成: {task_id}")

        except Exception as e:
            logger.error(f"任务失败 {task_id}: {e}\n{traceback.format_exc()}")
            TaskDB.update_task_status(task_id, TaskStatus.FAILED, error_message=str(e))
            redis_client.set_progress(task_id, {"status": "failed", "error": str(e)})
        finally:
            self.active_task_id = None

    def submit_task(self, task_id: str):
        get_redis_client().push_task(self.node_id, task_id)

    def get_status(self) -> dict:
        return {
            "node_id": self.node_id,
            "is_running": self.is_running,
            "active_task": self.active_task_id,
            "queue_size": get_redis_client().get_queue_size(self.node_id),
        }


scheduler = TaskScheduler()
