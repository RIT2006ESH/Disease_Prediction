"""
End-to-end entrypoint: load -> preprocess -> train -> evaluate -> explain -> promote.
Run per disease: python -m src.run_pipeline diabetes
                 python -m src.run_pipeline cardio
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_diabetes_data, load_cardio_data
from src.data.preprocess import (
    get_diabetes_pipeline, split_diabetes_data,
    get_cardio_pipeline, split_cardio_data,
)
from src.models.train import train_all_models
from src.models.evaluate import (
    plot_confusion_matrices, plot_roc_pr_curves, plot_model_comparison_bar,
)
from src.models.registry import promote_best_model
from src.explainability.shap_utils import plot_global_shap_summary


def run_diabetes_pipeline():
    df = load_diabetes_data()
    X_train, X_test, y_train, y_test = split_diabetes_data(df)
    preprocessor = get_diabetes_pipeline()

    train_all_models(X_train, y_train, X_test, y_test, preprocessor, "diabetes")
    plot_confusion_matrices("diabetes", X_test, y_test)
    plot_roc_pr_curves("diabetes", X_test, y_test)
    plot_model_comparison_bar("diabetes")
    promote_best_model("diabetes")
    plot_global_shap_summary("diabetes", X_train)


def run_cardio_pipeline():
    df = load_cardio_data()
    df = df.drop(columns=["source"])
    X_train, X_test, y_train, y_test = split_cardio_data(df)
    preprocessor = get_cardio_pipeline()

    train_all_models(X_train, y_train, X_test, y_test, preprocessor, "cardio")
    plot_confusion_matrices("cardio", X_test, y_test)
    plot_roc_pr_curves("cardio", X_test, y_test)
    plot_model_comparison_bar("cardio")
    promote_best_model("cardio")
    plot_global_shap_summary("cardio", X_train)


if __name__ == "__main__":
    disease = sys.argv[1] if len(sys.argv) > 1 else None

    if disease == "diabetes":
        run_diabetes_pipeline()
    elif disease == "cardio":
        run_cardio_pipeline()
    else:
        print("Usage: python -m src.run_pipeline [diabetes|cardio]")
        sys.exit(1)