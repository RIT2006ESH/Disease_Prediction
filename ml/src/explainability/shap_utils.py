"""
SHAP explainability utilities.
Used both offline (generating report visuals) and, via the same
normalization logic mirrored in the backend, at inference time.
"""

from pathlib import Path
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def load_production_artifacts(disease_name: str):
    """Loads the promoted production model (a full pipeline: preprocessor + SMOTE + clf)."""
    path = MODELS_DIR / disease_name / "production_model.pkl"
    return joblib.load(path)


def _get_explainer(pipeline, X_background):
    """
    Builds a SHAP explainer. TreeExplainer is used for tree-based models
    (fast, exact) — for Logistic Regression / SVM, falls back to
    KernelExplainer (model-agnostic, slower, needs a background sample).
    """
    clf = pipeline.named_steps["clf"]
    preprocessor = pipeline.named_steps["preprocessor"]
    X_background_transformed = preprocessor.transform(X_background)

    tree_based = clf.__class__.__name__ in (
        "DecisionTreeClassifier", "RandomForestClassifier", "XGBClassifier",
    )

    if tree_based:
        explainer = shap.TreeExplainer(clf)
    else:
        background_sample = shap.sample(X_background_transformed, 100)
        explainer = shap.KernelExplainer(clf.predict_proba, background_sample)

    return explainer, preprocessor


def _normalize_shap_values(shap_values):
    """
    SHAP's return shape varies across versions and explainer types:
      - list of arrays (one per class)              -> take class 1 (positive class)
      - 3D array (n_samples, n_features, n_classes)  -> take class 1
      - 2D array (n_samples, n_features)             -> already single-output, use as-is
    """
    if isinstance(shap_values, list):
        return shap_values[1]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def explain_single_prediction(disease_name: str, X_background, single_input_df) -> dict:
    """
    Returns top contributing features for a single prediction.
    """
    pipeline = load_production_artifacts(disease_name)
    explainer, preprocessor = _get_explainer(pipeline, X_background)

    X_transformed = preprocessor.transform(single_input_df)
    raw_shap_values = explainer.shap_values(X_transformed)
    shap_values = _normalize_shap_values(raw_shap_values)

    feature_names = preprocessor.get_feature_names_out()
    contributions = dict(zip(feature_names, shap_values[0]))

    top_features = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]

    return {
        "top_features": [
            {"feature": name, "impact": float(value)} for name, value in top_features
        ]
    }


def plot_global_shap_summary(disease_name: str, X_background):
    """Report visual: global feature importance across the whole test set."""
    pipeline = load_production_artifacts(disease_name)
    explainer, preprocessor = _get_explainer(pipeline, X_background)

    X_transformed = preprocessor.transform(X_background)
    raw_shap_values = explainer.shap_values(X_transformed)
    shap_values = _normalize_shap_values(raw_shap_values)

    feature_names = preprocessor.get_feature_names_out()
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)

    out_path = Path(__file__).resolve().parents[2] / "experiments" / f"{disease_name}_shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")