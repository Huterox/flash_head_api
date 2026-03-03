"""系统 API 路由"""
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from utils.result import R
from service.dependencies import verify_api_key
from state.scheduler import scheduler
from config import get_config

router = APIRouter(prefix="/api/system", tags=["系统"])


@router.get("/health", dependencies=[Depends(verify_api_key)])
def health_check():
    cfg = get_config()
    return JSONResponse(content=R.ok().data({
        "node_id": cfg.node.id,
        "node_name": cfg.node.name,
        "status": "healthy",
    }))


@router.get("/scheduler", dependencies=[Depends(verify_api_key)])
def scheduler_status():
    return JSONResponse(content=R.ok().data(scheduler.get_status()))
