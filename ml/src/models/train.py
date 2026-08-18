"""
Model training with cross-validation, SMOTE (inside CV folds only),
and hyperparameter tuning via GridSearchCV.
Trains all 5 models per disease, saves the best per model type,
and logs comparison metrics for registry.py to pick the final winner.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"

MODEL_GRID = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "svm": (
        SVC(probability=True, random_state=42),
        {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"]},
    ),
    "decision_tree": (
        DecisionTreeClassifier(random_state=42),
        {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 5, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [5, 10, None]},
    ),
    "xgboost": (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5, 7], "clf__learning_rate": [0.01, 0.1]},
    ),
}


def evaluate_predictions(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_all_models(
    X_train, y_train, X_test, y_test,
    preprocessor, disease_name: str,
) -> dict:
    """
    Trains all 5 model types with SMOTE + GridSearchCV, evaluates on held-out
    test set, saves each best model, and returns a metrics summary dict.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    disease_dir = MODELS_DIR / disease_name
    disease_dir.mkdir(parents=True, exist_ok=True)

    for name, (estimator, param_grid) in MODEL_GRID.items():
        print(f"\n[{disease_name}] Training {name}...")

        # SMOTE is INSIDE the pipeline: applied only to training folds
        # during CV, never to validation/test folds. Prevents leakage
        # and prevents synthetic samples from inflating test metrics.
        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("clf", estimator),
        ])

        grid = GridSearchCV(
            pipeline, param_grid, cv=cv,
            scoring="f1",  # F1 chosen over accuracy given class imbalance
            n_jobs=-1, verbose=0,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = evaluate_predictions(y_test, y_pred, y_proba)
        metrics["best_params"] = grid.best_params_
        results[name] = metrics

        joblib.dump(best_model, disease_dir / f"{name}.pkl")
        print(f"  Best params: {grid.best_params_}")
        print(f"  F1={metrics['f1']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}")

    # Save comparison metrics for the report / registry.py to consume
    with open(disease_dir / "metrics_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return results