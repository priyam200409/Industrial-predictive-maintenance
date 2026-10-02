from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GLOBAL_SHAP_PATH = (
    PROJECT_ROOT
    / "reports"
    / "shap"
    / "global_shap_importance.csv"
)

ENGINE_SHAP_PATH = (
    PROJECT_ROOT
    / "reports"
    / "shap"
    / "engine_shap_explanations.csv"
)

PREDICTION_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "shap"
    / "shap_prediction_summary.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day6_explainability_summary.csv"
)


def load_reports():
    """Load Day 6 explainability reports."""

    for path in [
        GLOBAL_SHAP_PATH,
        ENGINE_SHAP_PATH,
        PREDICTION_SUMMARY_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required report not found: {path}"
            )

    global_shap = pd.read_csv(
        GLOBAL_SHAP_PATH
    )

    engine_shap = pd.read_csv(
        ENGINE_SHAP_PATH
    )

    prediction_summary = pd.read_csv(
        PREDICTION_SUMMARY_PATH
    )

    return (
        global_shap,
        engine_shap,
        prediction_summary,
    )


def get_metric(summary, metric):
    """Retrieve a metric from the prediction summary."""

    match = summary.loc[
        summary["metric"] == metric,
        "value",
    ]

    if match.empty:
        raise ValueError(
            f"Metric not found: {metric}"
        )

    return float(match.iloc[0])


def build_summary(
    global_shap,
    engine_shap,
    prediction_summary,
):
    """Build consolidated Day 6 summary."""

    top_feature = (
        global_shap.iloc[0]["feature"]
    )

    top_feature_importance = (
        global_shap.iloc[0]["mean_abs_shap"]
    )

    second_feature = (
        global_shap.iloc[1]["feature"]
    )

    third_feature = (
        global_shap.iloc[2]["feature"]
    )

    validation_rows = get_metric(
        prediction_summary,
        "validation_rows",
    )

    mae = get_metric(
        prediction_summary,
        "mean_absolute_error",
    )

    mean_prediction_error = get_metric(
        prediction_summary,
        "mean_prediction_error",
    )

    mean_abs_shap = get_metric(
        prediction_summary,
        "mean_absolute_shap_contribution",
    )

    over_predictions = get_metric(
        prediction_summary,
        "over_prediction_rows",
    )

    under_predictions = get_metric(
        prediction_summary,
        "under_prediction_rows",
    )

    error_abs_shap_correlation = get_metric(
        prediction_summary,
        "error_abs_shap_correlation",
    )

    error_positive_shap_correlation = get_metric(
        prediction_summary,
        "error_positive_shap_correlation",
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "model",
                "validation_rows",
                "model_features",
                "engine_explanations",
                "explained_engines",
                "mean_absolute_error",
                "mean_prediction_error",
                "mean_absolute_shap_contribution",
                "over_prediction_rows",
                "under_prediction_rows",
                "top_global_shap_feature",
                "top_global_shap_mean_abs_value",
                "second_global_shap_feature",
                "third_global_shap_feature",
                "error_absolute_shap_correlation",
                "error_positive_shap_correlation",
            ],
            "value": [
                "XGBoost",
                int(validation_rows),
                len(global_shap),
                len(engine_shap),
                engine_shap["unit_id"].nunique(),
                mae,
                mean_prediction_error,
                mean_abs_shap,
                int(over_predictions),
                int(under_predictions),
                top_feature,
                top_feature_importance,
                second_feature,
                third_feature,
                error_abs_shap_correlation,
                error_positive_shap_correlation,
            ],
        }
    )

    return summary


def validate_summary(summary):
    """Validate the consolidated Day 6 report."""

    if summary.empty:
        raise ValueError(
            "Day 6 summary is empty."
        )

    if summary.isna().any().any():
        raise ValueError(
            "Day 6 summary contains NaN values."
        )

    required_metrics = {
        "model",
        "validation_rows",
        "model_features",
        "engine_explanations",
        "explained_engines",
        "mean_absolute_error",
        "mean_prediction_error",
        "mean_absolute_shap_contribution",
        "over_prediction_rows",
        "under_prediction_rows",
        "top_global_shap_feature",
        "top_global_shap_mean_abs_value",
        "second_global_shap_feature",
        "third_global_shap_feature",
        "error_absolute_shap_correlation",
        "error_positive_shap_correlation",
    }

    actual_metrics = set(
        summary["metric"]
    )

    missing = (
        required_metrics - actual_metrics
    )

    if missing:
        raise ValueError(
            f"Missing metrics: {missing}"
        )


def main():

    print("=" * 70)
    print("DAY 6 EXPLAINABILITY SUMMARY")
    print("=" * 70)

    print("\n[1/3] Loading explainability reports...")

    (
        global_shap,
        engine_shap,
        prediction_summary,
    ) = load_reports()

    print(
        f"Global SHAP features: {len(global_shap)}"
    )

    print(
        f"Engine explanation rows: {len(engine_shap)}"
    )

    print(
        f"Prediction metrics: {len(prediction_summary)}"
    )

    print("\n[2/3] Building consolidated summary...")

    summary = build_summary(
        global_shap,
        engine_shap,
        prediction_summary,
    )

    validate_summary(summary)

    print(
        f"Summary metrics: {len(summary)}"
    )

    print("\n[3/3] Saving report...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print("\nDay 6 summary:")
    print(
        summary.to_string(index=False)
    )

    print(
        "\nDay 6 explainability summary completed successfully."
    )


if __name__ == "__main__":
    main()
    