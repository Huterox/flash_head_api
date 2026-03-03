"""SQLAlchemy 数据库模型（网关复用节点的 tasks 表，只读）"""
from sqlalchemy import Column, String, Integer, Text, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone, timedelta

Base = declarative_base()
SHANGHAI_TZ = timezone(timedelta(hours=8))


def now_shanghai():
    return datetime.now(SHANGHAI_TZ)


class Task(Base):
    __tablename__ = 'tasks'

    task_id = Column(String(36), primary_key=True)
    node_id = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)
    config = Column(JSONB, nullable=True)
    result = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=now_shanghai)
    updated_at = Column(DateTime, default=now_shanghai, onupdate=now_shanghai)

    __table_args__ = (
        Index('idx_tasks_node_status', 'node_id', 'status'),
    )
