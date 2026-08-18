from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


# ---------------------------------------------------------------------------
# Diabetes preprocessing
# ---------------------------------------------------------------------------

DIABETES_NUMERIC = ["age", "bmi", "hba1c_level", "blood_glucose_level"]
DIABETES_CATEGORICAL = ["gender", "smoking_history"]
DIABETES_BINARY = ["hypertension", "heart_disease"]  # already 0/1, no encoding needed
DIABETES_TARGET = "diabetes"


def get_diabetes_pipeline() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, DIABETES_NUMERIC),
        ("cat", categorical_pipeline, DIABETES_CATEGORICAL),
        ("bin", "passthrough", DIABETES_BINARY),
    ])
    return preprocessor


def split_diabetes_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[DIABETES_NUMERIC + DIABETES_CATEGORICAL + DIABETES_BINARY]
    y = df[DIABETES_TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ---------------------------------------------------------------------------
# Cardiovascular preprocessing
# ---------------------------------------------------------------------------

CARDIO_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CARDIO_CATEGORICAL = ["cp", "restecg", "slope", "thal"]
CARDIO_BINARY = ["sex", "fbs", "exang"]  # already 0/1
CARDIO_TARGET = "target"


def get_cardio_pipeline() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, CARDIO_NUMERIC),
        ("cat", categorical_pipeline, CARDIO_CATEGORICAL),
        ("bin", "passthrough", CARDIO_BINARY),
    ])
    return preprocessor


def split_cardio_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[CARDIO_NUMERIC + CARDIO_CATEGORICAL + CARDIO_BINARY]
    y = df[CARDIO_TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )