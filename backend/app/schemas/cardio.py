from pydantic import BaseModel, Field
from typing import Literal

from app.schemas.diabetes import FeatureContribution


class CardioPredictionRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: Literal[0, 1]  # 0 = female, 1 = male (matches UCI encoding)
    cp: Literal[0, 1, 2, 3]  # chest pain type
    trestbps: float = Field(..., ge=60, le=260, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1]  # fasting blood sugar > 120 mg/dl
    restecg: Literal[0, 1, 2]  # resting ECG results
    thalach: float = Field(..., ge=60, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1]  # exercise-induced angina
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[0, 1, 2]  # slope of peak exercise ST segment
    ca: Literal[0, 1, 2, 3]  # number of major vessels colored by fluoroscopy
    thal: Literal[3, 6, 7]  # 3=normal, 6=fixed defect, 7=reversible defect

    class Config:
        json_schema_extra = {
            "example": {
                "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 246,
                "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
                "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 3,
            }
        }


class CardioPredictionResponse(BaseModel):
    risk_label: Literal["Low Risk", "High Risk"]
    probability: float
    top_features: list[FeatureContribution]
    model_version: str
    disclaimer: str = (
        "This tool provides a statistical risk estimate based on a machine "
        "learning model and is NOT a medical diagnosis. Please consult a "
        "qualified healthcare professional for any medical concerns."
    )