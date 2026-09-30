"""
Extract feature importance from the trained XGBoost RUL model.

The script loads the trained XGBoost pipeline, maps feature importance
to the corresponding feature names, ranks the features, and saves
the results as a CSV report.
"""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_rul_model.joblib"
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "xgboost_feature_importance.csv"

EXCLUDED_COLUMNS = {"unit_id", "cycle", "RUL"}


def load_model():
    """Load the trained XGBoost pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_feature_names():
    """Load feature names used by the model."""

    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_DATA_PATH}"
        )

    df = pd.read_csv(FEATURE_DATA_PATH)

    feature_names = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    return feature_names


def extract_feature_importance(model, feature_names):
    """Extract and rank XGBoost feature importance."""

    # The saved pipeline contains:
    # 1. imputer
    # 2. regressor
    xgb_model = model.named_steps["regressor"]

    importances = xgb_model.feature_importances_

    if len(importances) != len(feature_names):
        raise ValueError(
            "Number of feature importances does not match "
            "number of model features."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.insert(
        0,
        "rank",
        range(1, len(importance_df) + 1),
    )

    return importance_df


def main():
    """Run feature importance analysis."""

    print("Loading trained XGBoost model...")
    model = load_model()

    print("Loading model feature names...")
    feature_names = load_feature_names()

    print(f"Model features: {len(feature_names)}")

    importance_df = extract_feature_importance(
        model,
        feature_names,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nTop 20 features:")
    print(
        importance_df.head(20).to_string(index=False)
    )

    print(
        f"\nSaved report to: {OUTPUT_PATH}"
    )

    print(
        f"Total features analyzed: {len(importance_df)}"
    )


if __name__ == "__main__":
    main()