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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MAX_SVM_SAMPLES = 10000  # kernel SVM doesn't scale past ~10-20k rows

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
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    disease_dir = MODELS_DIR / disease_name
    disease_dir.mkdir(parents=True, exist_ok=True)

    for name, (estimator, param_grid) in MODEL_GRID.items():
        print(f"\n[{disease_name}] Training {name}...")

        if name == "svm" and len(X_train) > MAX_SVM_SAMPLES:
            X_train_use, _, y_train_use, _ = train_test_split(
                X_train, y_train, train_size=MAX_SVM_SAMPLES,
                stratify=y_train, random_state=42,
            )
        else:
            X_train_use, y_train_use = X_train, y_train

        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("clf", estimator),
        ])

        n_jobs = 1 if name == "svm" else -1
        grid = GridSearchCV(
            pipeline, param_grid, cv=cv,
            scoring="f1", n_jobs=n_jobs, verbose=1,
        )
        grid.fit(X_train_use, y_train_use)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = evaluate_predictions(y_test, y_pred, y_proba)
        metrics["best_params"] = grid.best_params_
        results[name] = metrics

        joblib.dump(best_model, disease_dir / f"{name}.pkl")
        print(f"  Best params: {grid.best_params_}")
        print(f"  F1={metrics['f1']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}")

    with open(disease_dir / "metrics_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
