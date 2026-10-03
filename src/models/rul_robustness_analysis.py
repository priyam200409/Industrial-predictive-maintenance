from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "xgboost_validation_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "rul_robustness_by_lifecycle.csv"
)


RUL_BINS = [
    (0, 20, "0-20"),
    (21, 50, "21-50"),
    (51, 100, "51-100"),
    (101, 150, "101-150"),
    (151, 200, "151-200"),
    (201, 250, "201-250"),
    (251, float("inf"), "251+"),
]


def load_predictions():
    """Load validated XGBoost validation predictions."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction report not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = {
        "unit_id",
        "cycle",
        "RUL",
        "predicted_RUL",
        "error",
        "absolute_error",
        "squared_error",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def assign_lifecycle_stage(rul):
    """Assign each observation to an RUL lifecycle stage."""

    for lower, upper, label in RUL_BINS:
        if lower <= rul <= upper:
            return label

    raise ValueError(
        f"RUL value outside expected ranges: {rul}"
    )


def add_lifecycle_stage(df):
    """Add lifecycle stage to validation predictions."""

    result = df.copy()

    result["lifecycle_stage"] = (
        result["RUL"]
        .apply(assign_lifecycle_stage)
    )

    return result


def calculate_robustness_profile(df):
    """Calculate robustness metrics by lifecycle stage."""

    rows = []

    for _, _, label in RUL_BINS:

        group = df[
            df["lifecycle_stage"] == label
        ]

        if group.empty:
            continue

        over_prediction_rate = (
            group["error"] > 0
        ).mean()

        under_prediction_rate = (
            group["error"] < 0
        ).mean()

        rows.append(
            {
                "RUL_range": label,
                "samples": len(group),
                "engines": group["unit_id"].nunique(),
                "MAE": group["absolute_error"].mean(),
                "RMSE": np.sqrt(
                    group["squared_error"].mean()
                ),
                "mean_error": group["error"].mean(),
                "median_absolute_error": (
                    group["absolute_error"].median()
                ),
                "max_absolute_error": (
                    group["absolute_error"].max()
                ),
                "over_prediction_rate": (
                    over_prediction_rate
                ),
                "under_prediction_rate": (
                    under_prediction_rate
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_output(result):
    """Validate robustness report."""

    if result.empty:
        raise ValueError(
            "Robustness report is empty."
        )

    required_columns = {
        "RUL_range",
        "samples",
        "engines",
        "MAE",
        "RMSE",
        "mean_error",
        "median_absolute_error",
        "max_absolute_error",
        "over_prediction_rate",
        "under_prediction_rate",
    }

    missing = required_columns - set(
        result.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if result.isna().any().any():
        raise ValueError(
            "Robustness report contains NaN values."
        )

    if (result["samples"] <= 0).any():
        raise ValueError(
            "Lifecycle stage contains zero samples."
        )

    if (
        result["over_prediction_rate"] < 0
    ).any() or (
        result["over_prediction_rate"] > 1
    ).any():
        raise ValueError(
            "Invalid over-prediction rates."
        )

    if (
        result["under_prediction_rate"] < 0
    ).any() or (
        result["under_prediction_rate"] > 1
    ).any():
        raise ValueError(
            "Invalid under-prediction rates."
        )


def main():

    print("=" * 70)
    print("RUL LIFECYCLE ROBUSTNESS ANALYSIS")
    print("=" * 70)

    print("\n[1/4] Loading validation predictions...")

    df = load_predictions()

    print(
        f"Validation rows: {len(df)}"
    )

    print(
        f"Validation engines: "
        f"{df['unit_id'].nunique()}"
    )

    print("\n[2/4] Assigning lifecycle stages...")

    df = add_lifecycle_stage(df)

    print(
        "Lifecycle stages:",
        df["lifecycle_stage"]
        .unique()
        .tolist(),
    )

    print("\n[3/4] Calculating robustness profile...")

    result = calculate_robustness_profile(
        df
    )

    validate_output(result)

    print(
        f"Lifecycle groups: {len(result)}"
    )

    print("\n[4/4] Saving report...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print("\nLifecycle robustness profile:")
    print(
        result.to_string(index=False)
    )

    print(
        "\nRUL lifecycle robustness analysis "
        "completed successfully."
    )


if __name__ == "__main__":
    main()