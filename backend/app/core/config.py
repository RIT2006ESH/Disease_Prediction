import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings

# In Docker, ML_DIR is set via environment variable and points to a mounted
# volume. Locally, it falls back to the relative path on disk.
ML_DIR = os.environ.get("ML_DIR", str(Path(__file__).resolve().parents[3] / "ml"))
sys.path.append(ML_DIR)


class Settings(BaseSettings):
    PROJECT_NAME: str = "Chronic Disease Risk Prediction API"
    API_V1_PREFIX: str = "/api/v1"

    DATABASE_URL: str = "mysql+pymysql://appuser:apppass@localhost:3306/disease_prediction"

    ML_MODELS_DIR: Path = Path(ML_DIR) / "models"

    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"


settings = Settings()