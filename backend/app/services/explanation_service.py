"""
Wraps shap_utils.py for the two disease-specific inference paths.
Uses a small cached background sample per disease (loaded once) rather
than reloading training data on every request.
"""

import pandas as pd
from pathlib import Path

from app.core.config import settings
from app.services.prediction_service import _models

_background_cache: dict = {}


def _get_background(disease: str, fields: list[str]) -> pd.DataFrame:
    """
    SHAP needs a small background sample to compare against.
    For simplicity, this samples from the training data on first call
    and caches it in memory. Loaded once, not per-request.
    """
    if disease in _background_cache:
        return _background_cache[disease]

    from src.data.load_data import load_diabetes_data, load_cardio_data  # ml pipeline reused here

    if disease == "diabetes":
        df = load_diabetes_data()
    else:
        df = load_cardio_data().drop(columns=["source"])

    sample = df[fields].sample(n=min(100, len(df)), random_state=42)
    _background_cache[disease] = sample
    return sample


def _explain(disease: str, fields: list[str], input_df: pd.DataFrame) -> list[dict]:
    import shap

    pipeline = _models[disease]
    clf = pipeline.named_steps["clf"]
    preprocessor = pipeline.named_steps["preprocessor"]

    background = _get_background(disease, fields)
    background_transformed = preprocessor.transform(background)
    input_transformed = preprocessor.transform(input_df)

    tree_based = clf.__class__.__name__ in (
        "DecisionTreeClassifier", "RandomForestClassifier", "XGBClassifier",
    )

    if tree_based:
        explainer = shap.TreeExplainer(clf)
    else:
        bg_sample = shap.sample(background_transformed, 50)
        explainer = shap.KernelExplainer(clf.predict_proba, bg_sample)

    shap_values = explainer.shap_values(input_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    feature_names = preprocessor.get_feature_names_out()
    contributions = dict(zip(feature_names, shap_values[0]))
    top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]

    return [{"feature": name, "impact": float(value)} for name, value in top]


def explain_diabetes_prediction(input_df: pd.DataFrame) -> list[dict]:
    from app.services.prediction_service import DIABETES_FIELDS
    return _explain("diabetes", DIABETES_FIELDS, input_df)


def explain_cardio_prediction(input_df: pd.DataFrame) -> list[dict]:
    from app.services.prediction_service import CARDIO_FIELDS
    return _explain("cardio", CARDIO_FIELDS, input_df)