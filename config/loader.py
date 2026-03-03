"""配置加载 + 单例"""
import os
import yaml
from .schema import AppConfig

_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yml")
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        _config = AppConfig(**raw)

        # 环境变量覆盖
        if "NODE_ID" in os.environ:
            _config.node.id = os.environ["NODE_ID"]
        if "NODE_NAME" in os.environ:
            _config.node.name = os.environ["NODE_NAME"]

    return _config
