"""FastAPI 应用创建 + 生命周期"""
import os

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError

from config import get_config
from utils.result import R
from state.db_engine import init_sync_engine, close_sync_engine
from state.redis_client import get_redis_client
from state.scheduler import scheduler
from cores.pipeline_adapter import init_pipeline
from loguru import logger

from service.routes import task_api, file_api, system_api

app = FastAPI(
    title="Once FlashHead API",
    description="FlashHead 头部视频生成服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=200, content=R.fail(exc.status_code, exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    msgs = [f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
    return JSONResponse(status_code=200, content=R.error("请求参数错误: " + "; ".join(msgs)))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"未处理异常: {exc}")
    import traceback
    logger.error(traceback.format_exc())
    return JSONResponse(status_code=200, content=R.fail(500, str(exc)))


# 注册路由
app.include_router(task_api.router)
app.include_router(file_api.router)
app.include_router(system_api.router)


# 管理面板
@app.get("/", response_class=HTMLResponse)
def index():
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>FlashHead API</h1><p><a href='/docs'>API Docs</a></p>")


@app.on_event("startup")
def on_startup():
    cfg = get_config()
    logger.info(f"节点启动: {cfg.node.id} ({cfg.node.name})")

    # 创建必要目录
    for d in [cfg.cache_dir, cfg.out_dir, cfg.server.file_upload.upload_dir]:
        os.makedirs(d, exist_ok=True)

    # 初始化基础设施
    init_sync_engine()
    get_redis_client().test_connection()

    # 初始化 FlashHead pipeline
    init_pipeline()

    # 恢复未完成任务
    from state.db_operations import TaskDB
    pending = TaskDB.get_pending_tasks(cfg.node.id)
    redis_client = get_redis_client()
    for t in pending:
        redis_client.push_task(cfg.node.id, t["task_id"])
        if t["status"] == "running":
            TaskDB.update_task_status(t["task_id"], "pending")
    if pending:
        logger.info(f"恢复 {len(pending)} 个未完成任务")

    scheduler.start()
    logger.info("服务启动完成")

    # 向网关注册 + 启动心跳
    _register_to_gateway(cfg)


def _register_to_gateway(cfg):
    """向网关注册节点并启动心跳线程"""
    if not cfg.gateway.enabled:
        logger.info("网关自动注册未启用，跳过")
        return

    gw_url = cfg.gateway.url
    if not gw_url:
        logger.info("未配置网关地址，跳过注册")
        return

    import threading
    import time
    import requests as req_lib

    # 优先使用 gateway 配置中的字段，留空则回退到 node/server 配置
    node_id = cfg.gateway.node_id or cfg.node.id
    node_name = cfg.gateway.node_name or cfg.node.name
    node_url = cfg.gateway.node_url or f"http://127.0.0.1:{cfg.server.port}"
    api_key = cfg.gateway.api_key or cfg.server.api_key

    payload = {
        "node_id": node_id,
        "node_name": node_name,
        "node_url": node_url,
        "api_key": api_key,
    }

    # 注册
    try:
        resp = req_lib.post(f"{gw_url.rstrip('/')}/api/node/register", json=payload, timeout=5)
        if resp.status_code == 200:
            logger.info(f"已向网关注册: {gw_url}")
        else:
            logger.warning(f"网关注册失败: {resp.text}")
    except Exception as e:
        logger.warning(f"网关不可达，跳过注册: {e}")

    # 心跳线程（定期重新注册，保持节点信息最新）
    def heartbeat_loop():
        interval = cfg.gateway.heartbeat_interval
        while True:
            time.sleep(interval)
            try:
                req_lib.post(
                    f"{gw_url.rstrip('/')}/api/node/register",
                    json=payload, timeout=5,
                )
            except Exception:
                pass

    t = threading.Thread(target=heartbeat_loop, daemon=True)
    t.start()
    logger.info(f"心跳线程已启动 (间隔={cfg.gateway.heartbeat_interval}s)")


@app.on_event("shutdown")
def on_shutdown():
    scheduler.stop()
    get_redis_client().close()
    close_sync_engine()
    logger.info("服务已关闭")


def main_service():
    import uvicorn
    cfg = get_config()

    uvicorn_kwargs = {
        "app": "service.app:app",
        "host": cfg.server.host,
        "port": cfg.server.port,
        "log_level": "info",
        "workers": 1,
    }

    if cfg.server.ssl_enabled:
        uvicorn_kwargs["ssl_certfile"] = cfg.server.ssl.certfile
        uvicorn_kwargs["ssl_keyfile"] = cfg.server.ssl.keyfile

    uvicorn.run(**uvicorn_kwargs)
