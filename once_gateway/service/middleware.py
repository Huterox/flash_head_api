"""认证中间件（纯 ASGI，兼容文件上传）"""
import logging

from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.requests import Request
from starlette.responses import JSONResponse

from once_gateway.config import get_config
from once_gateway.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

_PUBLIC_PATHS = {"/docs", "/openapi.json", "/"}
_PUBLIC_PREFIXES = ("/static/", "/admin/login", "/api/node/register")

ADMIN_TOKEN_PREFIX = "flashhead_gateway:admin_token:"
ADMIN_TOKEN_TTL = 86400


async def verify_admin_token(token: str) -> bool:
    if not token:
        return False
    r = await get_redis()
    val = await r.get(f"{ADMIN_TOKEN_PREFIX}{token}")
    return val is not None


async def store_admin_token(token: str, username: str) -> None:
    r = await get_redis()
    await r.set(f"{ADMIN_TOKEN_PREFIX}{token}", username, ex=ADMIN_TOKEN_TTL)


async def delete_admin_token(token: str) -> None:
    r = await get_redis()
    await r.delete(f"{ADMIN_TOKEN_PREFIX}{token}")


class AuthMiddleware:
    """纯 ASGI 中间件，不消费 request body，兼容文件上传"""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        path = request.url.path

        # 公开路径直接放行
        if path in _PUBLIC_PATHS or any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            await self.app(scope, receive, send)
            return

        cfg = get_config()

        # 管理 API 认证（token 方式）
        if path.startswith("/admin/"):
            auth_header = request.headers.get("Authorization", "")
            token = ""
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
            if not token:
                token = request.query_params.get("token", "")
            if not await verify_admin_token(token):
                resp = JSONResponse({"code": 401, "message": "管理员认证失败，请重新登录"})
                await resp(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return

        # 用户 API 认证（api_key 方式，同时支持 admin token）
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            bearer_token = auth_header[7:]
            if await verify_admin_token(bearer_token):
                await self.app(scope, receive, send)
                return

        api_key = cfg.gateway.api_key
        if not api_key:
            await self.app(scope, receive, send)
            return

        provided = request.headers.get("X-API-Key", "")
        if not provided:
            provided = request.query_params.get("api_key", "")
        if provided != api_key:
            resp = JSONResponse({"code": 401, "message": "API Key 认证失败"})
            await resp(scope, receive, send)
            return

        await self.app(scope, receive, send)
