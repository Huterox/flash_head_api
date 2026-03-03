"""SQLAlchemy 引擎和会话管理（同步模式）"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import threading

from config import get_config
from loguru import logger

_sync_engine = None
_sync_session_factory = None
_engine_lock = threading.Lock()


def _get_database_url() -> str:
    cfg = get_config().database
    return f"postgresql+psycopg2://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.database}"


def init_sync_engine():
    global _sync_engine, _sync_session_factory
    with _engine_lock:
        if _sync_engine is not None:
            return
        db_url = _get_database_url()
        logger.info(f"初始化数据库引擎: {db_url.split('@')[-1]}")
        _sync_engine = create_engine(db_url, echo=False, pool_size=10, max_overflow=20, pool_pre_ping=True)

        @event.listens_for(_sync_engine, "connect")
        def set_timezone(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("SET timezone='Asia/Shanghai'")
            cursor.close()

        _sync_session_factory = sessionmaker(_sync_engine, class_=Session, expire_on_commit=False)

        from state.db_models import Base
        Base.metadata.create_all(_sync_engine)
        logger.info("数据库引擎初始化成功（表已自动创建）")


@contextmanager
def get_sync_session():
    if _sync_session_factory is None:
        init_sync_engine()
    session = _sync_session_factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def close_sync_engine():
    global _sync_engine, _sync_session_factory
    with _engine_lock:
        if _sync_engine:
            _sync_engine.dispose()
            _sync_engine = None
            _sync_session_factory = None
            logger.info("数据库引擎已关闭")
