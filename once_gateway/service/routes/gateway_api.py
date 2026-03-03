"""核心转发 API + 节点自动注册 + 体验中心"""
import logging

from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse

from once_gateway.config.schema import NodeConfig
from once_gateway.core.node_registry import get_node_registry
from once_gateway.core.scheduler import get_scheduler
from once_gateway.utils.result import R
from once_gateway.utils.http_client import forward_request

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== 节点自动注册 ====================

@router.post("/api/node/register")
async def node_register(request: Request):
    """节点启动时自动注册（公开端点，不需要认证）"""
    body = await request.json()
    node_id = body.get("node_id", "")
    node_name = body.get("node_name", "")
    node_url = body.get("node_url", "")
    api_key = body.get("api_key", "")

    if not node_id or not node_url:
        return JSONResponse(R.error("node_id 和 node_url 为必填项"))

    registry = get_node_registry()
    node_cfg = NodeConfig(id=node_id, url=node_url, api_key=api_key, name=node_name)
    node = await registry.register(node_cfg)
    return JSONResponse(R.ok().data({"id": node.id, "url": node.url}))


# ==================== 代理转发 ====================

async def _get_node_or_error(node_id: str):
    registry = get_node_registry()
    node = await registry.get(node_id)
    if not node:
        return None, JSONResponse(R.error(f"节点 {node_id} 不存在"))
    if node.status.value != "healthy":
        return None, JSONResponse(R.error(f"节点 {node_id} 不健康"))
    return node, None


@router.post("/api/proxy/{node_id}/files/upload")
async def proxy_upload_file(node_id: str, file: UploadFile = File(...)):
    node, err = await _get_node_or_error(node_id)
    if err:
        return err
    file_content = await file.read()
    resp = await forward_request(
        "POST", node.url, "/api/files/upload",
        api_key=node.api_key,
        files={"file": (file.filename, file_content, file.content_type)},
    )
    return JSONResponse(content=resp.json())


@router.post("/api/proxy/{node_id}/tasks/synthesize")
async def proxy_create_task(node_id: str, request: Request):
    node, err = await _get_node_or_error(node_id)
    if err:
        return err
    body = await request.json()
    resp = await forward_request(
        "POST", node.url, "/api/tasks/synthesize",
        api_key=node.api_key, json=body,
    )
    return JSONResponse(content=resp.json())


@router.get("/api/proxy/{node_id}/tasks/{task_id}/preview")
async def proxy_preview_video(node_id: str, task_id: str):
    node, err = await _get_node_or_error(node_id)
    if err:
        return err
    resp = await forward_request(
        "GET", node.url, f"/api/tasks/{task_id}/preview?key={node.api_key}",
    )
    ct = resp.headers.get("content-type", "")
    if resp.status_code != 200 or ct.startswith("application/json"):
        return JSONResponse(content=resp.json())
    return StreamingResponse(
        iter([resp.content]), media_type="video/mp4",
        headers={"Content-Length": resp.headers.get("content-length", "")},
    )


@router.get("/api/proxy/{node_id}/tasks/{task_id}/download")
async def proxy_download_video(node_id: str, task_id: str):
    node, err = await _get_node_or_error(node_id)
    if err:
        return err
    resp = await forward_request(
        "GET", node.url, f"/api/tasks/{task_id}/download?key={node.api_key}",
    )
    ct = resp.headers.get("content-type", "")
    if resp.status_code != 200 or ct.startswith("application/json"):
        return JSONResponse(content=resp.json())
    return StreamingResponse(
        iter([resp.content]), media_type="video/mp4",
        headers={
            "Content-Length": resp.headers.get("content-length", ""),
            "Content-Disposition": f'attachment; filename="{task_id}.mp4"',
        },
    )


# ==================== 体验中心（网关自动调度） ====================

@router.post("/api/experience/upload")
async def experience_upload(file: UploadFile = File(...), node_id: str = None):
    """体验中心上传文件，首次自动调度，后续可指定 node_id 复用同一节点"""
    registry = get_node_registry()
    if node_id:
        node = await registry.get(node_id)
        if not node:
            return JSONResponse(R.error(f"节点 {node_id} 不存在"))
    else:
        scheduler = get_scheduler()
        node = await scheduler.select_node()
        if not node:
            return JSONResponse(R.error("无可用节点"))
    file_content = await file.read()
    resp = await forward_request(
        "POST", node.url, "/api/files/upload",
        api_key=node.api_key,
        files={"file": (file.filename, file_content, file.content_type)},
    )
    data = resp.json()
    if data.get("code") == 200:
        data["data"]["node_id"] = node.id
    return JSONResponse(content=data)


@router.post("/api/experience/synthesize")
async def experience_synthesize(request: Request):
    """体验中心创建合成任务，转发到指定节点"""
    body = await request.json()
    node_id = body.pop("node_id", None)
    if not node_id:
        return JSONResponse(R.error("缺少 node_id"))
    registry = get_node_registry()
    node = await registry.get(node_id)
    if not node:
        return JSONResponse(R.error(f"节点 {node_id} 不存在"))
    resp = await forward_request(
        "POST", node.url, "/api/tasks/synthesize",
        api_key=node.api_key, json=body,
    )
    data = resp.json()
    if data.get("code") == 200:
        data["data"]["node_id"] = node_id
    return JSONResponse(content=data)
