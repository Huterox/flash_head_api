"""httpx 异步 HTTP 客户端封装"""
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200),
        )
    return _client


async def close_http_client() -> None:
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


async def forward_request(
    method: str,
    node_url: str,
    path: str,
    api_key: str = "",
    **kwargs: Any,
) -> httpx.Response:
    """转发 HTTP 请求到算法节点"""
    client = await get_http_client()
    url = f"{node_url.rstrip('/')}{path}"
    headers = kwargs.pop("headers", {})
    if api_key:
        headers["X-API-Key"] = api_key
    return await client.request(method, url, headers=headers, **kwargs)
