from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.schemas.diabetes import DiabetesPredictionRequest, DiabetesPredictionResponse
from app.services.prediction_service import predict_diabetes
from app.services.explanation_service import explain_diabetes_prediction
from app.db.session import get_db
from app.db.models import Prediction

router = APIRouter()


@router.post("", response_model=DiabetesPredictionResponse)
def predict(request: DiabetesPredictionRequest, db: Session = Depends(get_db)):
    request_data = request.model_dump()

    result = predict_diabetes(request_data)
    top_features = explain_diabetes_prediction(result["input_df"])

    response = DiabetesPredictionResponse(
        risk_label=result["risk_label"],
        probability=result["probability"],
        top_features=top_features,
        model_version=result["model_version"],
    )

    # Log to DB — doesn't block or affect the response if something's odd here later
    db_entry = Prediction(
        disease_type="diabetes",
        input_data=request_data,
        risk_label=response.risk_label,
        probability=response.probability,
        top_features=[f.model_dump() for f in response.top_features],
        model_version=response.model_version,
    )
    db.add(db_entry)
    db.commit()

    return response