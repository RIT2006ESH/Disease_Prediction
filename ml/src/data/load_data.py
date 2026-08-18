"""
Responsible ONLY for reading raw files 
No cleaning logic
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

# ---------------------------------------------------------------------------
# Diabetes dataset (Kaggle: iammustafatz/diabetes-prediction-dataset)
# ---------------------------------------------------------------------------

DIABETES_FILE = RAW_DIR / "diabetes" / "diabetes_prediction_dataset.csv"


def load_diabetes_data(path: Path = DIABETES_FILE) -> pd.DataFrame:
    """
    Loads the raw diabetes dataset.
    Expected columns: gender, age, hypertension, heart_disease,
    smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes
    """
    df = pd.read_csv(path)

    
    df.columns = [c.strip().lower() for c in df.columns]

    # Drop exact duplicate rows — known issue in this dataset.
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"[load_diabetes_data] Dropped {dropped} duplicate rows")

    return df

# ---------------------------------------------------------------------------
# PIMA Indians Diabetes (secondary — literature-review benchmark only,
# not part of the main training pipeline)
# ---------------------------------------------------------------------------

PIMA_FILE = RAW_DIR / "diabetes" / "pima-indians-diabetes.csv"

PIMA_COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree_function", "age", "outcome",
]


def load_pima_data(path: Path = PIMA_FILE) -> pd.DataFrame:
    """
    Loads the PIMA Indians Diabetes dataset (no header row in source file).
    Used only as a benchmark reference in the report/literature review —
    not part of the primary training pipeline.
    """
    df = pd.read_csv(path, header=None, names=PIMA_COLUMNS)

    # Known issue: 0s in these columns are missing-data placeholders,
    # not real physiological zeros. Convert to NaN so it's handled
    # explicitly rather than silently treated as valid data.
    zero_as_missing = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
    df[zero_as_missing] = df[zero_as_missing].replace(0, pd.NA)

    return df

# ---------------------------------------------------------------------------
# Cardiovascular dataset (UCI Heart Disease: Cleveland/Hungarian/Switzerland/VA)
# ---------------------------------------------------------------------------

CARDIO_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

CARDIO_SOURCES = {
    "cleveland": RAW_DIR / "cardio" / "processed.cleveland.data",
    "hungarian": RAW_DIR / "cardio" / "processed.hungarian.data",
    "switzerland": RAW_DIR / "cardio" / "processed.switzerland.data",
    "va": RAW_DIR / "cardio" / "processed.va.data",
}


def _load_single_cardio_source(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=CARDIO_COLUMNS, na_values="?")
    df["source"] = source_name  # keep provenance — useful for missingness analysis
    return df


def load_cardio_data(sources: dict = CARDIO_SOURCES) -> pd.DataFrame:
    """
    Loads and combines all 4 UCI Heart Disease sources.
    Binarizes the target: 0 = no disease, 1 = disease present (was 0-4 severity).
    """
    frames = []
    for name, path in sources.items():
        if not path.exists():
            print(f"[load_cardio_data] WARNING: missing file for source '{name}': {path}")
            continue
        frames.append(_load_single_cardio_source(path, name))

    if not frames:
        raise FileNotFoundError(
            "No cardio source files found. Check ml/data/raw/cardio/ "
            "and CARDIO_SOURCES paths."
        )

    df = pd.concat(frames, ignore_index=True)

    # Binarize target: original values are 0 (no disease) to 4 (severity levels)
    df["target"] = (df["target"] > 0).astype(int)

    return df


if __name__ == "__main__":
    diabetes_df = load_diabetes_data()
    print("Diabetes dataset:", diabetes_df.shape)
    print(diabetes_df["diabetes"].value_counts(normalize=True))

    cardio_df = load_cardio_data()
    print("\nCardio dataset:", cardio_df.shape)
    print(cardio_df["target"].value_counts(normalize=True))
    print(cardio_df.groupby("source").apply(lambda g: g.isna().mean()))