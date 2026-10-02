from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAP_PATH = (
    PROJECT_ROOT
    / "reports"
    / "shap"
    / "validation_shap_values.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "shap"
    / "global_shap_importance.csv"
)


def load_shap_data():
    """Load validation SHAP values."""
    if not SHAP_PATH.exists():
        raise FileNotFoundError(
            f"SHAP values not found: {SHAP_PATH}"
        )

    return pd.read_csv(SHAP_PATH)


def calculate_global_importance(df):
    """Calculate global feature importance from SHAP values."""

    metadata_columns = {
        "unit_id",
        "cycle",
        "RUL",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    shap_matrix = df[feature_columns]

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap": shap_matrix.abs().mean().values,
            "mean_shap": shap_matrix.mean().values,
            "shap_std": shap_matrix.std().values,
        }
    )

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    importance["rank"] = (
        np.arange(len(importance)) + 1
    )

    return importance


def validate_importance(importance):
    """Validate the global SHAP importance report."""

    if importance.empty:
        raise ValueError(
            "Global SHAP importance report is empty."
        )

    required_columns = {
        "feature",
        "mean_abs_shap",
        "mean_shap",
        "shap_std",
        "rank",
    }

    if set(importance.columns) != required_columns:
        raise ValueError(
            "Unexpected global SHAP report columns."
        )

    if importance["feature"].duplicated().any():
        raise ValueError(
            "Duplicate features found."
        )

    if importance["mean_abs_shap"].isna().any():
        raise ValueError(
            "NaN values found in SHAP importance."
        )

    if not importance["mean_abs_shap"].is_monotonic_decreasing:
        raise ValueError(
            "Features are not sorted by SHAP importance."
        )

    if importance["rank"].iloc[0] != 1:
        raise ValueError(
            "Ranking does not start at 1."
        )


def main():
    print("=" * 70)
    print("GLOBAL SHAP FEATURE IMPORTANCE")
    print("=" * 70)

    print("\n[1/4] Loading SHAP values...")

    df = load_shap_data()

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\n[2/4] Calculating global SHAP importance...")

    importance = calculate_global_importance(df)

    print(
        f"Features analyzed: {len(importance)}"
    )

    print("\n[3/4] Validating results...")

    validate_importance(importance)

    print("Validation passed.")

    print("\n[4/4] Saving report...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print("\nTop 20 SHAP features:")
    print(
        importance.head(20).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()