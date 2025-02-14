from fastapi import APIRouter, File, UploadFile, HTTPException
from services.image_service import process_image

router = APIRouter()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return await process_image(file)