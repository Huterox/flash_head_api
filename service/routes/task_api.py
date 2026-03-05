"""任务 API 路由"""
import uuid
import os
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, FileResponse

from service.dependencies import verify_api_key
from utils.result import R
from schema.request_entities import SynthesizeRequest
from state.db_operations import TaskDB, FileDB
from state.scheduler import scheduler
from state.redis_client import get_redis_client
from config import get_config
from loguru import logger

router = APIRouter(prefix="/api/tasks", tags=["任务管理"])


@router.post("/synthesize", dependencies=[Depends(verify_api_key)])
def create_synthesize_task(req: SynthesizeRequest):
    image_file = FileDB.get_file(req.image_file_id)
    if not image_file:
        return JSONResponse(content=R.error("图片文件不存在"))
    audio_file = FileDB.get_file(req.audio_file_id)
    if not audio_file:
        return JSONResponse(content=R.error("音频文件不存在"))

    task_id = str(uuid.uuid4())
    node_id = get_config().node.id
    config = {
        "image_file_id": req.image_file_id,
        "audio_file_id": req.audio_file_id,
        "crop_region": req.crop_region,
        "restore_to_original": req.restore_to_original,
        "bg_remove": req.bg_remove,
        "bg_color": req.bg_color,
    }
    TaskDB.create_task(task_id, node_id, config)
    scheduler.submit_task(task_id)
    logger.info(f"创建任务: {task_id} | 配置: restore_to_original={req.restore_to_original}, bg_remove={req.bg_remove}, bg_color={req.bg_color}")
    return JSONResponse(content=R.ok().data({"task_id": task_id, "status": "pending"}))


@router.get("/list", dependencies=[Depends(verify_api_key)])
def list_tasks(status: str = None, keyword: str = None, page: int = 1, page_size: int = 20):
    """分页查询任务列表，支持按状态筛选和关键词搜索"""
    node_id = get_config().node.id
    result = TaskDB.list_tasks(node_id, status=status, keyword=keyword, page=page, page_size=page_size)
    # 附加实时进度
    redis_client = get_redis_client()
    for item in result["items"]:
        if item["status"] in ("pending", "running"):
            item["progress"] = redis_client.get_progress(item["task_id"])
    return JSONResponse(content=R.ok().data(result))


@router.get("/{task_id}", dependencies=[Depends(verify_api_key)])
def get_task(task_id: str):
    task_dict = TaskDB.get_task(task_id)
    if not task_dict:
        return JSONResponse(content=R.error("任务不存在"))
    task_dict["progress"] = get_redis_client().get_progress(task_id)
    return JSONResponse(content=R.ok().data(task_dict))


@router.get("/{task_id}/download")
def download_task_result(task_id: str, key: str = None):
    """下载视频 - 支持 query param ?key=xxx 认证（浏览器直接打开）"""
    expected_key = get_config().server.api_key
    if expected_key and key != expected_key:
        return JSONResponse(content=R.fail(401, "Invalid API Key"))
    task_dict = TaskDB.get_task(task_id)
    if not task_dict:
        return JSONResponse(content=R.error("任务不存在"))
    if task_dict["status"] != "completed":
        return JSONResponse(content=R.error("任务未完成"))
    result = task_dict.get("result", {})
    video_path = result.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return JSONResponse(content=R.error("视频文件不存在"))
    return FileResponse(video_path, media_type="video/mp4", filename=f"{task_id}.mp4")


@router.get("/{task_id}/preview")
def preview_task_video(task_id: str, key: str = None):
    """视频预览流 - 支持 query param 认证"""
    expected_key = get_config().server.api_key
    if expected_key and key != expected_key:
        return JSONResponse(content=R.fail(401, "Invalid API Key"))
    task_dict = TaskDB.get_task(task_id)
    if not task_dict:
        return JSONResponse(content=R.error("任务不存在"))
    if task_dict["status"] != "completed":
        return JSONResponse(content=R.error("任务未完成"))
    result = task_dict.get("result", {})
    video_path = result.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return JSONResponse(content=R.error("视频文件不存在"))
    from fastapi.responses import StreamingResponse
    def iter_file():
        with open(video_path, "rb") as f:
            while chunk := f.read(65536):
                yield chunk
    file_size = os.path.getsize(video_path)
    return StreamingResponse(iter_file(), media_type="video/mp4",
                             headers={"Content-Length": str(file_size), "Accept-Ranges": "bytes"})


@router.delete("/{task_id}", dependencies=[Depends(verify_api_key)])
def delete_task(task_id: str):
    """删除任务（running 状态不允许删除）"""
    task_dict = TaskDB.get_task(task_id)
    if not task_dict:
        return JSONResponse(content=R.error("任务不存在"))
    if task_dict["status"] == "running":
        return JSONResponse(content=R.error("运行中的任务不允许删除"))

    node_id = get_config().node.id
    redis_client = get_redis_client()

    # 清理 Redis：进度 + 队列
    redis_client.delete_progress(task_id)
    redis_client.remove_from_queue(node_id, task_id)

    # 清理输出文件
    out_dir = os.path.join(get_config().out_dir, task_id)
    if os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)

    # 删除 DB 记录
    TaskDB.delete_task(task_id)
    logger.info(f"删除任务: {task_id}")
    return JSONResponse(content=R.ok().data({"task_id": task_id}))


@router.post("/{task_id}/retry", dependencies=[Depends(verify_api_key)])
def retry_task(task_id: str):
    """重试失败的任务"""
    task_dict = TaskDB.get_task(task_id)
    if not task_dict:
        return JSONResponse(content=R.error("任务不存在"))
    if task_dict["status"] != "failed":
        return JSONResponse(content=R.error("仅失败的任务可以重试"))

    # 清理旧的输出文件
    out_dir = os.path.join(get_config().out_dir, task_id)
    if os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)

    # 重置状态并重新入队
    TaskDB.update_task_status(task_id, "pending", result=None, error_message=None)
    redis_client = get_redis_client()
    redis_client.delete_progress(task_id)
    scheduler.submit_task(task_id)
    logger.info(f"重试任务: {task_id}")
    return JSONResponse(content=R.ok().data({"task_id": task_id, "status": "pending"}))
