from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.xray import XrayPredictionResponse
from app.services.xray_service import predict_xray, is_ready

router = APIRouter()


@router.post("", response_model=XrayPredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not is_ready():
        raise HTTPException(status_code=503, detail="X-ray model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    result = predict_xray(image_bytes)

    if not result["valid_input"]:
        raise HTTPException(
            status_code=422,
            detail="Uploaded image does not appear to be a valid chest X-ray (color/saturation check failed).",
        )

    return XrayPredictionResponse(**result)
