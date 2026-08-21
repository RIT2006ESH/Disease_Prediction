"""
Wraps SHAP explainability for the two disease-specific inference paths.
Uses a small cached background sample per disease (loaded once) rather
than reloading training data on every request.
"""

import numpy as np
import pandas as pd

_background_cache: dict = {}


def _get_background(disease: str, fields: list[str]) -> pd.DataFrame:
    """
    SHAP needs a small background sample to compare against.
    Sampled from the training data on first call and cached in memory —
    loaded once per disease, not per-request.
    """
    if disease in _background_cache:
        return _background_cache[disease]

    from src.data.load_data import load_diabetes_data, load_cardio_data

    if disease == "diabetes":
        df = load_diabetes_data()
    else:
        df = load_cardio_data().drop(columns=["source"])

    sample = df[fields].sample(n=min(100, len(df)), random_state=42)
    _background_cache[disease] = sample
    return sample


def _normalize_shap_values(shap_values):
    """
    SHAP's return shape varies across versions and explainer types:
      - list of arrays (one per class)              -> take class 1 (positive class)
      - 3D array (n_samples, n_features, n_classes)  -> take class 1
      - 2D array (n_samples, n_features)             -> already single-output, use as-is
    Normalizing here means the rest of the code never needs to care
    which SHAP version or explainer produced the values.
    """
    if isinstance(shap_values, list):
        return shap_values[1]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def _explain(disease: str, fields: list[str], input_df: pd.DataFrame) -> list[dict]:
    import shap
    from app.services.prediction_service import get_model

    pipeline = get_model(disease)
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

    raw_shap_values = explainer.shap_values(input_transformed)
    shap_values = _normalize_shap_values(raw_shap_values)

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