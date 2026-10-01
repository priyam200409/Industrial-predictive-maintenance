"""
Analyze XGBoost feature importance by feature category.

Categories:
- Original sensor features
- Rolling mean features
- Rolling standard deviation features
- Delta features
- Slope features
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "reports" / "xgboost_feature_importance.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "xgboost_feature_category_importance.csv"


def classify_feature(feature_name):
    """Classify a feature based on its naming pattern."""

    if "_rolling_mean_" in feature_name:
        return "Rolling Mean"

    if "_rolling_std_" in feature_name:
        return "Rolling Standard Deviation"

    if "_delta_" in feature_name:
        return "Delta"

    if "_slope_" in feature_name:
        return "Slope"

    if feature_name.startswith("sensor_"):
        return "Original Sensor"

    return "Other"


def main():
    """Run feature category importance analysis."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature importance report not found: {INPUT_PATH}"
        )

    importance_df = pd.read_csv(INPUT_PATH)

    importance_df["category"] = importance_df["feature"].apply(
        classify_feature
    )

    category_df = (
        importance_df.groupby("category", as_index=False)
        .agg(
            feature_count=("feature", "count"),
            total_importance=("importance", "sum"),
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
        )
        .sort_values(
            by="total_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    category_df["importance_percentage"] = (
        category_df["total_importance"]
        / category_df["total_importance"].sum()
        * 100
    )

    category_df.insert(
        0,
        "rank",
        range(1, len(category_df) + 1),
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    category_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nFeature importance by category:")
    print(category_df.to_string(index=False))

    print(
        f"\nSaved report to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()