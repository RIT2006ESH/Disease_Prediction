"""
Loads production model artifacts once at startup and runs inference.
"""

import joblib
import pandas as pd

from app.core.config import settings

_models: dict = {}
_manifests: dict = {}

DIABETES_FIELDS = [
    "age", "bmi", "hba1c_level", "blood_glucose_level",
    "gender", "smoking_history", "hypertension", "heart_disease",
]

CARDIO_FIELDS = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca",
    "cp", "restecg", "slope", "thal",
    "sex", "fbs", "exang",
]


def load_models():
    import json
    for disease in ("diabetes", "cardio"):
        model_path = settings.ML_MODELS_DIR / disease / "production_model.pkl"
        manifest_path = settings.ML_MODELS_DIR / disease / "production_manifest.json"

        if not model_path.exists():
            print(f"[prediction_service] WARNING: no production model for '{disease}'")
            continue

        _models[disease] = joblib.load(model_path)

        if manifest_path.exists():
            with open(manifest_path) as f:
                _manifests[disease] = json.load(f)

        print(f"[prediction_service] Loaded {disease} model")


def is_ready() -> bool:
    return "diabetes" in _models and "cardio" in _models


def get_model(disease: str):
    if disease not in _models:
        raise RuntimeError(f"Model for '{disease}' is not loaded")
    return _models[disease]


def _get_model_version(disease: str) -> str:
    manifest = _manifests.get(disease, {})
    model_name = manifest.get("selected_model", "unknown")
    return f"{disease}_{model_name}_v1"


def predict_diabetes(request_data: dict) -> dict:
    model = _models["diabetes"]
    input_df = pd.DataFrame([{k: request_data[k] for k in DIABETES_FIELDS}])

    proba = model.predict_proba(input_df)[0][1]
    risk_label = "High Risk" if proba >= 0.5 else "Low Risk"

    return {
        "probability": float(proba),
        "risk_label": risk_label,
        "model_version": _get_model_version("diabetes"),
        "input_df": input_df,
    }


def predict_cardio(request_data: dict) -> dict:
    model = _models["cardio"]
    input_df = pd.DataFrame([{k: request_data[k] for k in CARDIO_FIELDS}])

    proba = model.predict_proba(input_df)[0][1]
    risk_label = "High Risk" if proba >= 0.5 else "Low Risk"

    return {
        "probability": float(proba),
        "risk_label": risk_label,
        "model_version": _get_model_version("cardio"),
        "input_df": input_df,
    }