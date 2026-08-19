from fastapi import APIRouter

from app.services.prediction_service import is_ready

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": is_ready(),
    }