from pydantic import BaseModel, Field
from typing import Literal


class DiabetesPredictionRequest(BaseModel):
    gender: Literal["Male", "Female", "Other"]
    age: float = Field(..., ge=0, le=120)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    smoking_history: Literal["never", "former", "current", "not current", "ever", "No Info"]
    bmi: float = Field(..., ge=10, le=80)
    hba1c_level: float = Field(..., ge=3, le=20)
    blood_glucose_level: float = Field(..., ge=50, le=400)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "age": 45,
                "hypertension": 0,
                "heart_disease": 0,
                "smoking_history": "never",
                "bmi": 27.3,
                "hba1c_level": 6.1,
                "blood_glucose_level": 130,
            }
        }


class FeatureContribution(BaseModel):
    feature: str
    impact: float


class DiabetesPredictionResponse(BaseModel):
    risk_label: Literal["Low Risk", "High Risk"]
    probability: float
    top_features: list[FeatureContribution]
    model_version: str
    disclaimer: str = (
        "This tool provides a statistical risk estimate based on a machine "
        "learning model and is NOT a medical diagnosis. Please consult a "
        "qualified healthcare professional for any medical concerns."
    )