"""文件管理：上传、存储、清理"""
import os
import uuid
from datetime import datetime, timedelta, timezone

from config import get_config
from loguru import logger

SHANGHAI_TZ = timezone(timedelta(hours=8))


class FileManager:
    def __init__(self):
        cfg = get_config().server.file_upload
        self.upload_dir = cfg.upload_dir
        self.max_file_size = cfg.max_file_size * 1024 * 1024
        self.allowed_extensions = cfg.allowed_extensions
        self.retain_hours = cfg.retain_hours
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_upload(self, filename: str, content: bytes) -> dict:
        ext = os.path.splitext(filename)[1].lower()
        if self.allowed_extensions and ext not in self.allowed_extensions:
            raise ValueError(f"不支持的文件类型: {ext}")
        if len(content) > self.max_file_size:
            raise ValueError(f"文件过大，最大 {self.max_file_size // (1024*1024)}MB")

        file_id = str(uuid.uuid4())
        now = datetime.now(SHANGHAI_TZ)
        dir_path = os.path.join(self.upload_dir, now.strftime("%Y/%m/%d"), file_id)
        os.makedirs(dir_path, exist_ok=True)
        stored_path = os.path.join(dir_path, f"{file_id}{ext}")
        with open(stored_path, "wb") as f:
            f.write(content)

        file_type = "image" if ext in [".png", ".jpg", ".jpeg"] else "audio"
        expires_at = now + timedelta(hours=self.retain_hours)

        return {
            "file_id": file_id, "filename": filename,
            "stored_path": stored_path, "file_size": len(content),
            "file_type": file_type, "expires_at": expires_at,
        }


_file_manager = None


def get_file_manager() -> FileManager:
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager
