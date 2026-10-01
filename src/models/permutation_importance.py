"""
Calculate permutation importance for the trained XGBoost RUL model.

Permutation importance measures how much validation performance
changes when individual features are randomly shuffled.

The analysis uses the exact engine-level train/validation split
defined in src.models.split_data.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_rul_model.joblib"
FEATURE_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "train_features.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "xgboost_permutation_importance.csv"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

TARGET_COLUMN = "RUL"

EXCLUDED_COLUMNS = {
    "unit_id",
    "cycle",
    TARGET_COLUMN,
}

RANDOM_STATE = 42
N_REPEATS = 5


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load the trained XGBoost pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

def load_features():
    """Load the final engineered feature dataset."""

    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_DATA_PATH}"
        )

    return pd.read_csv(FEATURE_DATA_PATH)


# ============================================================
# RECREATE VALIDATION SPLIT
# ============================================================

def create_validation_split(data):
    """
    Recreate the exact Day 4 engine-level validation split.
    """

    from src.models.split_data import split_by_engine

    _, validation = split_by_engine(data)

    return validation


# ============================================================
# PREPARE VALIDATION DATA
# ============================================================

def prepare_validation_data(validation):
    """Separate validation features and target."""

    feature_columns = [
        column
        for column in validation.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X_validation = validation[feature_columns]
    y_validation = validation[TARGET_COLUMN]

    return X_validation, y_validation, feature_columns


# ============================================================
# CALCULATE PERMUTATION IMPORTANCE
# ============================================================

def calculate_importance(
    model,
    X_validation,
    y_validation,
    feature_columns,
):
    """Calculate permutation importance using validation MAE."""

    result = permutation_importance(
        model,
        X_validation,
        y_validation,
        scoring="neg_mean_absolute_error",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.insert(
        0,
        "rank",
        range(1, len(importance_df) + 1),
    )

    return importance_df


# ============================================================
# MAIN
# ============================================================

def main():
    """Run permutation importance analysis."""

    print("Loading trained XGBoost model...")
    model = load_model()

    print("Loading engineered feature dataset...")
    data = load_features()

    print("Recreating Day 4 engine-level validation split...")
    validation = create_validation_split(data)

    print(
        f"Validation engines: "
        f"{validation['unit_id'].nunique()}"
    )

    print(
        f"Validation rows: "
        f"{len(validation)}"
    )

    X_validation, y_validation, feature_columns = (
        prepare_validation_data(validation)
    )

    print(
        f"Validation features: "
        f"{len(feature_columns)}"
    )

    print(
        f"\nCalculating permutation importance "
        f"with {N_REPEATS} repeats..."
    )

    importance_df = calculate_importance(
        model,
        X_validation,
        y_validation,
        feature_columns,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nTop 20 features by permutation importance:")
    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    print(
        f"\nSaved report to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Total features analyzed: "
        f"{len(importance_df)}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()