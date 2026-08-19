import sys
sys.path.append(str(Path(__file__).resolve().parents[3] / "ml"))
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    PROJECT_NAME: str = "Chronic Disease Risk Prediction API"
    API_V1_PREFIX: str = "/api/v1"

    DATABASE_URL: str = "mysql+pymysql://appuser:apppass@localhost:3306/disease_prediction"

    # Points to the ml/models directory — shared artifact location between
    # the ML training pipeline and this backend, per the "keep ML training
    # code separate from production API code" requirement.
    ML_MODELS_DIR: Path = Path(__file__).resolve().parents[3] / "ml" / "models"

    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"


settings = Settings()