from fastapi import APIRouter

from app.api.v1.endpoints import health, diabetes, cardio

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(diabetes.router, prefix="/predict/diabetes", tags=["diabetes"])
api_router.include_router(cardio.router, prefix="/predict/cardio", tags=["cardio"])