"""SQLAlchemy 数据库引擎（网关复用节点同一个 PostgreSQL）"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from once_gateway.config import get_config
from loguru import logger

_sync_engine = None
_sync_session_factory = None


def init_sync_engine():
    global _sync_engine, _sync_session_factory
    cfg = get_config().database
    url = f"postgresql://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.database}"
    _sync_engine = create_engine(
        url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        connect_args={"options": "-c timezone=Asia/Shanghai"},
    )
    _sync_session_factory = sessionmaker(bind=_sync_engine)
    logger.info(f"数据库连接成功: {cfg.host}:{cfg.port}/{cfg.database}")


@contextmanager
def get_sync_session() -> Session:
    session = _sync_session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def close_sync_engine():
    global _sync_engine
    if _sync_engine:
        _sync_engine.dispose()
        logger.info("数据库连接已关闭")
