"""SQLAlchemy 数据库模型"""
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

    task_id = Column(String(36), primary_key=True, comment='任务ID')
    node_id = Column(String(50), nullable=False, comment='节点ID')
    status = Column(String(20), nullable=False, comment='任务状态')
    config = Column(JSONB, nullable=True, comment='任务配置')
    result = Column(JSONB, nullable=True, comment='任务结果')
    error_message = Column(Text, nullable=True, comment='错误信息')
    created_at = Column(DateTime, default=now_shanghai, comment='创建时间')
    updated_at = Column(DateTime, default=now_shanghai, onupdate=now_shanghai, comment='更新时间')

    __table_args__ = (
        Index('idx_tasks_node_status', 'node_id', 'status'),
    )


class UploadedFile(Base):
    __tablename__ = 'uploaded_files'

    file_id = Column(String(36), primary_key=True, comment='文件ID')
    filename = Column(String(255), nullable=False, comment='原始文件名')
    stored_path = Column(String(500), nullable=False, comment='存储路径')
    file_size = Column(Integer, nullable=False, comment='文件大小(字节)')
    file_type = Column(String(20), nullable=False, comment='文件类型')
    upload_time = Column(DateTime, default=now_shanghai, comment='上传时间')
    expires_at = Column(DateTime, nullable=True, comment='过期时间')
    created_at = Column(DateTime, default=now_shanghai, comment='创建时间')

    __table_args__ = (
        Index('idx_uploaded_files_expires_at', 'expires_at'),
    )
