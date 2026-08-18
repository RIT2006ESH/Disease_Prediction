"""
Model registry: selects the best model per disease from metrics_comparison.json
and promotes it to a stable artifact name for the backend to consume.

This is the one place that decides "what is currently in production."
Re-running training does NOT automatically change production — you must
re-run promote_best_model() explicitly. That's deliberate: it mirrors
real MLOps practice where retraining and deployment are separate steps.
"""

import json
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"

# Primary metric used to rank models. F1 chosen for consistency with
# training-time model selection (see train.py) — both stages optimize
# for the same objective given class imbalance.
SELECTION_METRIC = "f1"


def select_best_model(disease_name: str, metric: str = SELECTION_METRIC) -> dict:
    """
    Reads metrics_comparison.json for a disease and returns
    {model_name, metrics} for the best-performing model.
    """
    disease_dir = MODELS_DIR / disease_name
    metrics_path = disease_dir / "metrics_comparison.json"

    with open(metrics_path) as f:
        results = json.load(f)

    best_name = max(results, key=lambda name: results[name][metric])
    return {"model_name": best_name, "metrics": results[best_name]}


def promote_best_model(disease_name: str, metric: str = SELECTION_METRIC) -> Path:
    """
    Copies the winning model file to a stable name: production_model.pkl
    This is the ONLY file the FastAPI backend should ever load.
    """
    disease_dir = MODELS_DIR / disease_name
    best = select_best_model(disease_name, metric)

    source = disease_dir / f"{best['model_name']}.pkl"
    destination = disease_dir / "production_model.pkl"
    shutil.copy(source, destination)

    # Record which model + version is in production, with its metrics —
    # useful both for the backend's /health endpoint and your report.
    manifest = {
        "disease": disease_name,
        "selected_model": best["model_name"],
        "selection_metric": metric,
        "metrics": best["metrics"],
    }
    with open(disease_dir / "production_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[{disease_name}] Promoted '{best['model_name']}' to production "
          f"({metric}={best['metrics'][metric]:.3f})")
    return destination


if __name__ == "__main__":
    promote_best_model("diabetes")
    promote_best_model("cardio")