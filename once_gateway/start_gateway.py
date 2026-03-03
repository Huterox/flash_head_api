"""Once FlashHead Gateway 入口"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from once_gateway.config import get_config


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = get_config()

    uvicorn_kwargs = {
        "app": "once_gateway.service.app:app",
        "host": cfg.gateway.host,
        "port": cfg.gateway.port,
        "log_level": "info",
    }

    if cfg.gateway.ssl_enabled:
        ssl_dir = os.path.dirname(os.path.abspath(__file__))
        uvicorn_kwargs["ssl_certfile"] = os.path.join(ssl_dir, cfg.gateway.ssl.certfile)
        uvicorn_kwargs["ssl_keyfile"] = os.path.join(ssl_dir, cfg.gateway.ssl.keyfile)

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
