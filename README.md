# Early-Stage Chronic Disease Risk Prediction Using Machine Learning and Explainable AI

A full-stack system for early-stage risk screening of diabetes and cardiovascular disease, with an exploratory third module for pneumonia detection from chest X-rays. Built as a final-year Computer Engineering project.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Datasets](#3-datasets)
4. [ML Methodology](#4-ml-methodology)
5. [Model Results](#5-model-results)
6. [Explainable AI (SHAP)](#6-explainable-ai-shap)
7. [Backend API Reference](#7-backend-api-reference)
8. [Database Schema](#8-database-schema)
9. [Frontend](#9-frontend)
10. [Bonus Module: X-Ray Pneumonia Classification](#10-bonus-module-x-ray-pneumonia-classification)
11. [Deployment (Docker)](#11-deployment-docker)
12. [Project Structure](#12-project-structure)
13. [Running the Project Locally](#13-running-the-project-locally)
14. [Known Limitations](#14-known-limitations)
15. [Future Work](#15-future-work)

---

## 1. Project Overview

### Problem Statement

Chronic diseases such as diabetes and cardiovascular disease are most treatable when caught early, but early-stage risk is often invisible without deliberate screening. This project builds a system that takes routine clinical markers — the kind already captured in a basic checkup — and produces a statistical risk estimate, along with a plain explanation of *why* the model reached that estimate.

### Objectives

- Predict **diabetes risk** and **cardiovascular disease risk** from separate public clinical datasets
- Compare five ML model families (Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost) per disease, selecting the best performer systematically rather than by default
- Apply rigorous ML practice: proper preprocessing, class-imbalance handling, cross-validation, hyperparameter tuning, and multi-metric evaluation
- Make predictions explainable via SHAP, not just a black-box score
- Deliver a real, deployable full-stack application — not a notebook — with a clean separation between ML training code and production serving code
- (Exploratory extension) Demonstrate the same explainability-first philosophy applied to a different data modality: image-based pneumonia classification from chest X-rays

### Tech Stack

| Layer | Technology |
|---|---|
| ML Training | Python, scikit-learn, XGBoost, imbalanced-learn, SHAP |
| Backend API | FastAPI, SQLAlchemy, Pydantic |
| Database | MySQL 8 |
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| Deployment | Docker, Docker Compose |
| Bonus Module | PyTorch, torchvision (ResNet18 transfer learning) |

---

## 2. System Architecture

The project is organized into three independently deployable layers:

```
ml/          — training pipeline (offline). Produces versioned model artifacts.
backend/     — FastAPI service (serving only). Loads artifacts, never trains.
frontend/    — React + TypeScript UI. Talks to backend over REST.
```

This mirrors standard MLOps practice: **train/serve separation**. Retraining a model does not require redeploying the API; redeploying the API does not require retraining a model. The `ml/models/<disease>/production_model.pkl` file is the single artifact the backend ever reads — everything upstream of that (raw data, preprocessing, the four other candidate models, tuning logs) stays entirely within `ml/` and never touches production code.

### Request Flow

```
React form (clinical inputs)
    │
    ▼
POST /api/v1/predict/{diabetes|cardio}   (JSON body)
POST /api/v1/predict/xray                (multipart/form-data, image upload)
    │
    ▼
FastAPI validates request via Pydantic schema
    │
    ▼
prediction_service: loads production_model.pkl
    (loaded once at application startup, not per-request)
    runs .predict_proba()
    │
    ▼
explanation_service: SHAP top-5 contributing features (tabular models only)
    │
    ▼
Response assembled: risk_label, probability, top_features,
                     disclaimer, model_version
    │
    ├──▶ Logged to MySQL (fire-and-forget relative to the response)
    │
    ▼
JSON returned to React, rendered as a result card
```

### Why Two Separate Prediction Endpoints Per Disease, Not One Generic Endpoint

Diabetes and cardiovascular disease have genuinely different input schemas (8 fields vs. 13 fields, entirely different feature semantics). A single generic `/predict` endpoint with a `disease_type` flag would mean loose, weakly-typed request validation. Separate endpoints mean separate Pydantic schemas, which FastAPI uses to auto-generate accurate OpenAPI documentation and enforce real input validation (e.g., rejecting `age: -5` at the schema level before any application code runs).

---

## 3. Datasets

### Diabetes

**Primary dataset:** ["Diabetes Prediction Dataset"](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) (Kaggle), 100,000 rows.

Chosen over the classic PIMA Indians Diabetes dataset as the primary source because PIMA is small (768 rows), covers a single demographic (Pima Indian women only), and has known missing-value encoding issues (zeros used as placeholders for missing glucose/BMI/skin-thickness values, which are not medically valid zero readings). The larger, more general dataset better supports a system intended to generalize across a broader population.

**Secondary/benchmark reference:** [PIMA Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — retained for literature-review comparison only, not used in the production pipeline.

**Features used:** age, gender, hypertension, heart disease (history), smoking history, BMI, HbA1c level, blood glucose level.

### Cardiovascular Disease

**Dataset:** [UCI Heart Disease Repository](https://archive.ics.uci.edu/dataset/45/heart+disease), combined across all four original collection sites (Cleveland, Hungarian, Switzerland, VA — approximately 920 rows total after combining).

Chosen over a larger but shallower alternative (a 70,000-row Kaggle cardiovascular dataset with only demographic/vitals features) because **feature richness mattered more than raw volume** for this project's goals. This dataset includes clinically diagnostic features — chest pain type, resting ECG results, ST depression induced by exercise, thalassemia, number of major vessels — that give SHAP explainability something clinically meaningful to explain, rather than only generic demographic risk factors.

**Features used:** age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate achieved, exercise-induced angina, ST depression (oldpeak), ST slope, number of major vessels, thalassemia.

### X-Ray (Bonus Module)

**Dataset:** [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle), ~5,800 labeled images (Normal / Pneumonia).

---

## 4. ML Methodology

### Preprocessing

- **Missing values:** median imputation (numeric features), most-frequent imputation (categorical/binary features) — applied via `sklearn.ColumnTransformer` inside the training pipeline, fit *only* on training folds during cross-validation, to prevent data leakage.
- The UCI cardiovascular data encodes missing values as the string `?` rather than a proper null — this is explicitly converted to `NaN` at data-loading time so imputation handles it correctly rather than treating it as a valid category.
- **Categorical encoding:** one-hot encoding for nominal features (smoking history, chest pain type, etc.); binary features (already 0/1) are passed through unchanged rather than redundantly re-encoded.
- **Feature scaling:** `StandardScaler`, fit only on training folds.
- **Target binarization (cardio only):** the original UCI target is a 0–4 severity scale; this is collapsed to binary (0 = no disease, 1 = disease present) for a clean classification task.

### Class Imbalance Handling

Both datasets are imbalanced (diabetes: roughly 8–9% positive class). **SMOTE** (Synthetic Minority Oversampling Technique) is applied *inside* the cross-validation pipeline using `imblearn.Pipeline`, meaning it is fit and applied only to training folds — never to validation or test folds. This is a deliberate methodological choice: applying SMOTE before the train/test split (a common mistake) would let synthetic samples leak influence into evaluation data, artificially inflating reported metrics.

### Model Comparison

Five model families were trained and compared per disease:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree (CART)
4. Random Forest
5. XGBoost

Each was tuned via `GridSearchCV` with 5-fold stratified cross-validation, optimizing for **F1 score** rather than accuracy. Under class imbalance, a trivial model that always predicts "no disease" would score roughly 91% accuracy on the diabetes dataset while being clinically useless — F1 forces the tuning process to genuinely value catching positive cases.

**Practical note on SVM:** kernel SVM training scales poorly (roughly O(n²)–O(n³)), which made it computationally infeasible to train on the full ~96,000-row diabetes training set on consumer hardware. SVM training was subsampled to a maximum of 10,000 rows specifically for this model family — documented explicitly here as a deliberate, reported limitation rather than a silent shortcut.

### Evaluation Metrics

Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix are reported for every model. **PR-AUC is reported alongside ROC-AUC** specifically because ROC-AUC can look deceptively strong under class imbalance (it accounts for the large true-negative rate), while PR-AUC focuses specifically on positive-class performance, which is more informative for this problem.

### Model Registry and Promotion

A dedicated promotion step (`ml/src/models/registry.py`) selects the best-performing model per disease (by F1 score on the held-out test set) and copies it to a stable filename: `production_model.pkl`. This is the *only* artifact the backend ever loads. Promotion is a deliberate, explicit, version-pinned step separate from training — retraining a model does not automatically change what is live in production without this explicit promotion step, which mirrors real-world MLOps deployment gating.

---

## 5. Model Results

Metrics from the final held-out test set, for the promoted production model per disease:

| Disease | Model Selected | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Diabetes | Decision Tree (CART) | 1.000 | 0.678 | 0.808 | 0.905 | 0.734 |
| Cardiovascular | Random Forest | 0.865 | 0.882 | 0.874 | 0.923 | 0.927 |

**Interpretation:**

- The diabetes model's **perfect precision with moderate recall** reflects the near-diagnostic nature of HbA1c and blood glucose level for diabetes classification — these two features alone create very clean separation between classes. The model is conservative: it only flags "high risk" when very confident, which explains the recall gap (some true positive cases with less extreme values are missed).
- The cardiovascular model shows a **more balanced profile** across precision and recall, consistent with cardiovascular risk being determined by a more diffuse combination of interacting factors (no single dominant feature), which suits an ensemble method like Random Forest well.

All five models per disease, with full hyperparameter search results, are saved in `ml/models/<disease>/metrics_comparison.json` for complete transparency and reproducibility.

---

## 6. Explainable AI (SHAP)

Every prediction is accompanied by the **top 5 contributing features**, computed via SHAP (SHapley Additive exPlanations) — a game-theoretic approach that attributes each feature's contribution to a specific prediction.

**Explainer selection is model-family-aware:**
- `TreeExplainer` — used for Decision Tree, Random Forest, and XGBoost. Exact and computationally fast, since it exploits the tree structure directly.
- `KernelExplainer` — used as a fallback for Logistic Regression and SVM. Model-agnostic but slower, and requires a background sample of representative data to approximate the explanation.

A positive SHAP value for a feature means that feature pushed the prediction *toward* the positive class (disease present); a negative value means it pushed *away* from it. This sign convention is surfaced directly in the UI (red for positive/risk-increasing, green for negative/risk-decreasing).

---

## 7. Backend API Reference

Base URL: `http://localhost:8000/api/v1`

Interactive API documentation (Swagger UI) is auto-generated by FastAPI and available at `http://localhost:8000/docs`.

### `GET /health`

Returns service and model-loading status.

```json
{ "status": "ok", "models_loaded": true }
```

### `POST /predict/diabetes`

**Request body:**
```json
{
  "gender": "Female",
  "age": 45,
  "hypertension": 0,
  "heart_disease": 0,
  "smoking_history": "never",
  "bmi": 27.3,
  "hba1c_level": 6.1,
  "blood_glucose_level": 130
}
```

**Response:**
```json
{
  "risk_label": "Low Risk",
  "probability": 0.0817,
  "top_features": [
    { "feature": "hba1c_level", "impact": -0.185 },
    { "feature": "age", "impact": -0.128 }
  ],
  "model_version": "diabetes_decision_tree_v1",
  "disclaimer": "This tool provides a statistical risk estimate..."
}
```

### `POST /predict/cardio`

**Request body:**
```json
{
  "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 246,
  "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
  "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 3
}
```

**Response:** same shape as `/predict/diabetes`, with `model_version: "cardio_random_forest_v1"`.

### `POST /predict/xray`

**Request:** `multipart/form-data`, field name `file`, an image file.

**Response (valid X-ray input):**
```json
{
  "risk_label": "High Risk",
  "probability": 0.601,
  "model_version": "pneumonia_resnet18_v1",
  "valid_input": true,
  "disclaimer": "This tool provides a statistical estimate..."
}
```

**Response (invalid/non-X-ray input):** HTTP 422
```json
{ "detail": "Uploaded image does not appear to be a valid chest X-ray (color/saturation check failed)." }
```

---

## 8. Database Schema

```sql
CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    disease_type VARCHAR(20) NOT NULL,
    input_data JSON NOT NULL,
    risk_label VARCHAR(20) NOT NULL,
    probability FLOAT NOT NULL,
    top_features JSON NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Why `JSON` columns for `input_data` and `top_features`, rather than rigid per-field columns:** the different prediction modalities have completely different input shapes (8 fields for diabetes, 13 for cardio, an image for X-ray). A single wide table with 20+ mostly-NULL columns per row is worse relational design than a queryable JSON blob (MySQL supports querying into JSON fields directly, e.g. `input_data->>'$.age'`), and avoids a schema migration every time a new prediction modality is added.

**Why `model_version` is stored per row:** if a model is retrained and re-promoted later, historical predictions remain traceable to the exact model that actually produced them, rather than silently being reattributed to whatever is currently live — an auditability property that matters for any system making health-adjacent predictions.

---

## 9. Frontend

Built with React + TypeScript + Vite, styled with Tailwind CSS using a custom clinical/lab-report-inspired design system (monospace data readouts, a restrained teal/alert/signal color palette, and an ECG-style pulse-line motif — a deliberate stylistic choice grounded in the subject matter rather than a generic dashboard template).

**Pages:**
- `/` — landing page with links to each assessment
- `/diabetes` — diabetes risk form
- `/cardio` — cardiovascular risk form
- `/xray` — X-ray upload and pneumonia risk assessment

Each result view shows the risk label, probability (with a visual bar), the top contributing features (tabular modalities only), the model version that produced the prediction, and a persistent medical disclaimer.

---

## 10. Bonus Module: X-Ray Pneumonia Classification

Added as a time-boxed exploratory extension to demonstrate the same explainability-first philosophy applied to a different data modality (images, not tabular clinical data).

**Architecture:** ResNet18 with a frozen pretrained backbone (ImageNet weights) — only the final classification layer is fine-tuned. This is a deliberate scope decision: full fine-tuning of a CNN is significantly more computationally expensive, and freezing the backbone keeps training tractable on CPU within a constrained timeframe while still leveraging genuinely useful pretrained visual features.

**Training scope:** capped at 1,000 training images (500 per class) and 2 epochs, given time constraints. Achieved **79.97% accuracy** on the full, uncapped held-out test set (624 images).

**Out-of-distribution input handling:** initial testing revealed the model would confidently misclassify clearly unrelated images (e.g., a UI screenshot) as high pneumonia risk — a well-known limitation of narrow image classifiers, which have no innate concept of "this input doesn't belong to my domain" unless explicitly given one. A lightweight heuristic guard was added: chest X-rays are near-uniformly grayscale, while photos, screenshots, and diagrams contain a meaningful fraction of genuinely colored pixels even when their *average* saturation looks low (e.g., a mostly-white UI with a few colored buttons). The guard computes the fraction of pixels with meaningful per-pixel R/G/B channel divergence and rejects inputs above a threshold, returning an explicit HTTP 422 rather than a silent, misleading prediction.

**Explicitly out of scope for this module (see [Limitations](#14-known-limitations)):** true pixel-level segmentation (this module performs classification, not segmentation), Grad-CAM visual explainability (the natural image-domain analogue to SHAP), and training on the full dataset rather than a capped subset.

---

## 11. Deployment (Docker)

The full system runs as three containers via Docker Compose:

```yaml
services:
  db:        # MySQL 8
  backend:   # FastAPI + PyTorch (CPU-only)
  frontend:  # React build, served via nginx with SPA fallback routing
```

**Key design decision:** the `ml/` folder (trained model artifacts and datasets) is mounted into the backend container as a **read-only volume**, rather than baked into the Docker image at build time. This keeps the backend image lean and means retraining and promoting a new model only requires a container restart — not a full image rebuild — directly extending the train/serve separation principle from the codebase into the deployment architecture.

**Run the full stack:**
```bash
docker compose up -d --build
```

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- Backend docs: `http://localhost:8000/docs`

---

## 12. Project Structure

```
disease-risk-prediction/
├── ml/                          # Training pipeline (offline, independent)
│   ├── data/raw/{diabetes,cardio}/
│   ├── src/
│   │   ├── data/                # loading, preprocessing
│   │   ├── models/              # train, tune, evaluate, registry
│   │   └── explainability/      # SHAP utilities
│   ├── models/<disease>/        # saved artifacts + production_model.pkl
│   └── experiments/             # evaluation plots, comparison charts
│
├── backend/                     # FastAPI (serving only)
│   ├── app/
│   │   ├── api/v1/endpoints/    # diabetes, cardio, xray, health
│   │   ├── core/config.py
│   │   ├── db/                  # SQLAlchemy models, session
│   │   ├── schemas/             # Pydantic request/response models
│   │   └── services/            # prediction + explanation logic
│   └── Dockerfile
│
├── frontend/                    # React + TypeScript
│   ├── src/
│   │   ├── pages/                # DiabetesForm, CardioForm, XrayForm, Home
│   │   ├── components/           # ResultDisplay, PulseLine
│   │   ├── api/client.ts
│   │   └── types/prediction.ts
│   └── Dockerfile
│
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   └── viva-notes.md
│
└── docker-compose.yml
```

---

## 13. Running the Project Locally

### Option A — Docker (recommended, single command)

```bash
docker compose up -d --build
```

### Option B — Manual (for development)

**ML pipeline** (run once to generate models):
```bash
cd ml
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
python -m src.run_pipeline diabetes
python -m src.run_pipeline cardio
```

**Backend:**
```bash
cd backend
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## 14. Known Limitations

- **SVM training was subsampled** on the larger diabetes dataset for computational feasibility; this is a documented trade-off, not an oversight.
- **The X-ray module is a scoped proof-of-concept**, not a fully realized third pillar of the system: frozen backbone, capped training data, no Grad-CAM explainability, and a heuristic (not learned) out-of-distribution guard that will not catch every possible invalid input (e.g., a different medical imaging modality like an MRI would likely pass the grayscale check despite being the wrong input type).
- **No authentication or authorization** is implemented on any endpoint — acceptable for an academic demonstration, but a hard requirement before any real-world deployment.
- **SHAP explainer logic is duplicated** between the offline ML pipeline (`ml/src/explainability/shap_utils.py`) and the backend's inference-time service (`backend/app/services/explanation_service.py`) — a known code duplication, acceptable given project time constraints, and a clear candidate for refactoring into a shared module.

## 15. Future Work

- Full fine-tuning (not just the final layer) of the X-ray classifier on the complete dataset
- Grad-CAM explainability for the X-ray module, to match the explainability standard already met by the tabular models
- True pixel-level segmentation for abnormality localization (would require mask-annotated data and a U-Net-style architecture — a substantial follow-on project in its own right)
- A learned, rather than heuristic, out-of-distribution detector for the X-ray module
- Authentication and role-based access control for a clinical deployment context
- A prediction history view in the frontend, surfacing the data already being logged to MySQL
