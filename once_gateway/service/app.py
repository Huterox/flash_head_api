"""网关 FastAPI 应用"""
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from once_gateway.config import get_config
from once_gateway.core.health_checker import start_health_checker, stop_health_checker
from once_gateway.core.node_registry import get_node_registry
from once_gateway.service.middleware import AuthMiddleware
from once_gateway.service.routes import admin_api, task_api, gateway_api
from once_gateway.state.db_engine import init_sync_engine, close_sync_engine
from once_gateway.utils.http_client import close_http_client
from once_gateway.utils.redis_client import get_redis, close_redis

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    logger.info("网关启动: port=%d", cfg.gateway.port)

    # 初始化
    init_sync_engine()
    await get_redis()

    registry = get_node_registry()
    for node_cfg in cfg.nodes:
        await registry.register(node_cfg)
    logger.info("已注册 %d 个配置节点", len(cfg.nodes))

    start_health_checker()
    logger.info("网关启动完成")

    yield

    # 关闭
    stop_health_checker()
    await close_http_client()
    await close_redis()
    close_sync_engine()
    logger.info("网关已关闭")


app = FastAPI(title="Once FlashHead Gateway", lifespan=lifespan)
app.add_middleware(AuthMiddleware)

app.include_router(admin_api.router)
app.include_router(task_api.router)
app.include_router(gateway_api.router)


@app.get("/", response_class=HTMLResponse)
def index():
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    if template_path.exists():
        return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>FlashHead Gateway</h1>")
