"""文件上传/下载 API"""
from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import JSONResponse

from service.dependencies import verify_api_key
from utils.result import R
from utils.file_manager import get_file_manager
from state.db_operations import FileDB
from loguru import logger

router = APIRouter(prefix="/api/files", tags=["文件管理"])


@router.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = get_file_manager().save_upload(file.filename, content)
        FileDB.create_file(
            file_id=result["file_id"], filename=result["filename"],
            stored_path=result["stored_path"], file_size=result["file_size"],
            file_type=result["file_type"], expires_at=result["expires_at"],
        )
        logger.info(f"文件上传: {result['file_id']} ({result['filename']}, {result['file_size']} bytes)")
        return JSONResponse(content=R.ok().data({
            "file_id": result["file_id"],
            "filename": result["filename"],
            "file_size": result["file_size"],
            "file_type": result["file_type"],
        }))
    except ValueError as e:
        return JSONResponse(content=R.error(str(e)))
    except Exception as e:
        logger.error(f"上传失败: {e}")
        return JSONResponse(content=R.fail(500, f"上传失败: {e}"))
