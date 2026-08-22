from pydantic import BaseModel
from typing import Literal


class XrayPredictionResponse(BaseModel):
    risk_label: Literal["Low Risk", "High Risk", "Invalid Input"]
    probability: float
    model_version: str
    valid_input: bool = True
    disclaimer: str = (
        "This tool provides a statistical estimate from a machine learning "
        "model trained on a limited dataset and is NOT a medical diagnosis. "
        "Please consult a qualified healthcare professional for any medical concerns."
    )
