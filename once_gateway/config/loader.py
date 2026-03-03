"""配置加载器：从 gateway_config.yml 加载配置，提供单例访问"""
import logging
from pathlib import Path
from typing import Optional

import yaml
from .schema import AppConfig

logger = logging.getLogger(__name__)

_config: Optional[AppConfig] = None
_CONFIG_PATH = Path(__file__).parent / "gateway_config.yml"


def _load_yaml() -> dict:
    if not _CONFIG_PATH.exists():
        logger.warning("配置文件不存在: %s，使用默认配置", _CONFIG_PATH)
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def get_config() -> AppConfig:
    global _config
    if _config is None:
        data = _load_yaml()
        _config = AppConfig(**data)
        logger.info("配置加载完成，网关端口: %d", _config.gateway.port)
    return _config
