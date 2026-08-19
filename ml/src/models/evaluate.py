"""
Evaluation visuals for the report — confusion matrices, ROC/PR curves,
and a cross-model comparison chart. Run AFTER train.py has populated
metrics_comparison.json for a disease.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "experiments"


def plot_confusion_matrices(disease_name: str, X_test, y_test):
    disease_dir = MODELS_DIR / disease_name
    model_names = ["logistic_regression", "svm", "decision_tree", "random_forest", "xgboost"]

    fig, axes = plt.subplots(1, len(model_names), figsize=(20, 4))
    for ax, name in zip(axes, model_names):
        model = joblib.load(disease_dir / f"{name}.pkl")
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, colorbar=False)
        ax.set_title(name)

    plt.tight_layout()
    out_path = REPORTS_DIR / f"{disease_name}_confusion_matrices.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_roc_pr_curves(disease_name: str, X_test, y_test):
    disease_dir = MODELS_DIR / disease_name
    model_names = ["logistic_regression", "svm", "decision_tree", "random_forest", "xgboost"]

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for name in model_names:
        model = joblib.load(disease_dir / f"{name}.pkl")
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc, name=name)
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax_pr, name=name)

    ax_roc.set_title(f"{disease_name}: ROC Curves")
    ax_pr.set_title(f"{disease_name}: Precision-Recall Curves")
    plt.tight_layout()
    out_path = REPORTS_DIR / f"{disease_name}_roc_pr_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_model_comparison_bar(disease_name: str):
    """Bar chart comparing F1, ROC-AUC, PR-AUC across all 5 models — good summary slide."""
    disease_dir = MODELS_DIR / disease_name
    with open(disease_dir / "metrics_comparison.json") as f:
        results = json.load(f)

    metrics_to_plot = ["f1", "roc_auc", "pr_auc"]
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics_to_plot):
        values = [results[m][metric] for m in model_names]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_title(f"{disease_name}: Model Comparison")
    ax.legend()
    plt.tight_layout()
    out_path = REPORTS_DIR / f"{disease_name}_model_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")