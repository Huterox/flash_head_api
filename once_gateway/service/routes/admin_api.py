"""管理面板 API"""
import logging
import secrets

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from once_gateway.config import get_config
from once_gateway.config.schema import NodeConfig
from once_gateway.core.node_registry import get_node_registry
from once_gateway.service.middleware import store_admin_token, delete_admin_token
from once_gateway.utils.result import R

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin")


@router.post("/login")
async def admin_login(request: Request):
    body = await request.json()
    username = body.get("username", "")
    password = body.get("password", "")

    cfg = get_config()
    if username != cfg.gateway.admin_username or password != cfg.gateway.admin_password:
        return JSONResponse(R.fail(401, "用户名或密码错误"))

    token = secrets.token_hex(32)
    await store_admin_token(token, username)
    return JSONResponse(R.ok().data({"token": token, "username": username}))


@router.get("/me")
async def admin_me():
    return JSONResponse(R.ok().data({"username": get_config().gateway.admin_username}))


@router.post("/logout")
async def admin_logout(request: Request):
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        await delete_admin_token(token)
    return JSONResponse(R.ok().data({"message": "已登出"}))


# ==================== 节点管理 ====================

@router.get("/nodes")
async def list_nodes():
    registry = get_node_registry()
    nodes = []
    for n in await registry.all_nodes():
        nodes.append({
            "id": n.id, "name": n.name, "url": n.url,
            "status": n.status.value,
            "last_health_check": n.last_health_check,
            "fail_count": n.fail_count,
        })
    return JSONResponse(R.ok().data({"nodes": nodes, "total": len(nodes)}))


@router.post("/nodes")
async def add_node(request: Request):
    body = await request.json()
    node_id = body.get("id", "")
    node_url = body.get("url", "")
    api_key = body.get("api_key", "")
    name = body.get("name", "")

    if not node_id or not node_url:
        return JSONResponse(R.error("id 和 url 为必填项"))

    registry = get_node_registry()
    if await registry.get(node_id):
        return JSONResponse(R.error(f"节点 {node_id} 已存在"))

    node_cfg = NodeConfig(id=node_id, url=node_url, api_key=api_key, name=name)
    await registry.register(node_cfg)
    return JSONResponse(R.ok().data({"id": node_id, "url": node_url}))


@router.delete("/nodes/{node_id}")
async def remove_node(node_id: str):
    registry = get_node_registry()
    node = await registry.unregister(node_id)
    if not node:
        return JSONResponse(R.error(f"节点 {node_id} 不存在"))
    return JSONResponse(R.ok().data({"id": node_id}))
