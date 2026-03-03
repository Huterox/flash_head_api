"""Pydantic 配置模型"""
from typing import List, Optional
from pydantic import BaseModel


class NodeConfig(BaseModel):
    id: str = "node_flashhead_01"
    name: str = "FlashHead节点"


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "flash_head_api"
    user: str = "postgres"
    password: str = ""


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0


class FlashHeadConfig(BaseModel):
    mode: str = "lite"                    # lite 或 pro
    ckpt_dir: str = "checkpoint/SoulX-FlashHead-1_3B"
    wav2vec_dir: str = "checkpoint/wav2vec2-base-960h"
    face_ratio: float = 2.0
    pro_device_ids: str = "0,1"           # pro 模式双卡，sequence parallel 需要两张 GPU
    torch_compile: bool = True            # 是否启用 torch.compile 加速（双卡 pro 模式建议关闭）


class FileUploadConfig(BaseModel):
    upload_dir: str = "cache/uploads"
    max_file_size: int = 100  # MB
    allowed_extensions: List[str] = [".png", ".jpg", ".jpeg", ".wav", ".mp3", ".m4a"]
    retain_hours: int = 720
    cleanup_interval: int = 3600


class ThreadPoolConfig(BaseModel):
    max_workers: int = 1
    queue_size: int = 200


class SSLConfig(BaseModel):
    certfile: str = "ssl/certificate.crt"
    keyfile: str = "ssl/private.key"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8100
    api_key: str = ""
    ssl_enabled: bool = False
    ssl: SSLConfig = SSLConfig()
    thread_pool: ThreadPoolConfig = ThreadPoolConfig()
    file_upload: FileUploadConfig = FileUploadConfig()


class GatewayConfig(BaseModel):
    enabled: bool = False
    url: str = ""
    api_key: str = ""
    node_id: str = ""
    node_name: str = ""
    node_url: str = ""
    heartbeat_interval: int = 30


class AppConfig(BaseModel):
    node: NodeConfig = NodeConfig()
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    flashhead: FlashHeadConfig = FlashHeadConfig()
    server: ServerConfig = ServerConfig()
    gateway: GatewayConfig = GatewayConfig()
    device_ids: str = "0"
    ffmpeg_path: str = "libs/ffmpeg.exe"
    cache_dir: str = "cache"
    out_dir: str = "cache/out"
