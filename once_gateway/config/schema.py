"""网关配置 Pydantic 模型"""
from typing import List
from pydantic import BaseModel, Field


class SSLConfig(BaseModel):
    certfile: str = "ssl/certificate.crt"
    keyfile: str = "ssl/private.key"


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8200
    api_key: str = ""
    admin_username: str = "admin"
    admin_password: str = "admin123"
    ssl_enabled: bool = False
    ssl: SSLConfig = SSLConfig()


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "flash_head_api"
    user: str = "postgres"
    password: str = ""


class NodeConfig(BaseModel):
    id: str
    url: str
    api_key: str = ""
    name: str = ""


class HealthCheckConfig(BaseModel):
    interval: int = 30
    timeout: int = 10
    unhealthy_threshold: int = 3


class RedisConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6379
    password: str = ""
    db: int = 0


class AppConfig(BaseModel):
    gateway: GatewayConfig = GatewayConfig()
    database: DatabaseConfig = DatabaseConfig()
    nodes: List[NodeConfig] = Field(default_factory=list)
    health_check: HealthCheckConfig = HealthCheckConfig()
    redis: RedisConfig = RedisConfig()
