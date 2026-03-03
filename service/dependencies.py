"""API 认证依赖"""
from fastapi import Header, HTTPException, status
from config import get_config


def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    expected_key = get_config().server.api_key
    if not expected_key:
        return True
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key header")
    if x_api_key != expected_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return True
