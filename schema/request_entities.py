"""请求/响应实体"""
from typing import Optional, List
from pydantic import BaseModel, Field


class SynthesizeRequest(BaseModel):
    image_file_id: str = Field(..., description="上传的图片文件ID")
    audio_file_id: str = Field(..., description="上传的音频文件ID")
    crop_region: Optional[List[int]] = Field(None, description="裁剪区域 [x1, y1, x2, y2]，不传则自动人脸检测")
    restore_to_original: bool = Field(False, description="是否将生成的头部视频贴回原图")
    bg_remove: bool = Field(False, description="是否移除背景（替换为绿幕）")
    bg_color: Optional[str] = Field("#00FF00", description="背景颜色（十六进制格式，如 #00FF00 绿色）")


class TaskResponse(BaseModel):
    task_id: str
    status: str
    config: Optional[dict] = None
    result: Optional[dict] = None
    error_message: Optional[str] = None
    progress: Optional[dict] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
