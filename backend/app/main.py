from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.router import api_router
from app.services.prediction_service import load_models
from app.db.session import Base, engine
from app.db import models  # noqa: F401 — import needed so SQLAlchemy registers the Prediction table


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create DB tables if they don't exist yet, then load ML models.
    print("Creating database tables (if not already present)...")
    Base.metadata.create_all(bind=engine)

    print("Loading ML models...")
    load_models()

    yield

    print("Shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def root():
    return {"message": settings.PROJECT_NAME, "docs": "/docs"}