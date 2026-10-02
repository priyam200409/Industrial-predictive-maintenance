from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "train_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "models" / "xgboost_rul_model.joblib"
)

SPLIT_PATH = (
    PROJECT_ROOT / "reports" / "train_validation_split.csv"
)

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
    / "shap_prediction_analysis.csv"
)


def load_inputs():
    """Load validation data, model and SHAP values."""

    for path in [
        FEATURE_DATA_PATH,
        MODEL_PATH,
        SPLIT_PATH,
        SHAP_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    features = pd.read_csv(FEATURE_DATA_PATH)
    split = pd.read_csv(SPLIT_PATH)
    shap_df = pd.read_csv(SHAP_PATH)

    return features, split, shap_df


def prepare_validation(features, split):
    """Prepare the exact Day 4 validation dataset."""

    validation_engines = split.loc[
        split["split"] == "validation",
        "unit_id",
    ].tolist()

    validation = features[
        features["unit_id"].isin(validation_engines)
    ].copy()

    validation = validation.reset_index(drop=True)

    feature_columns = [
        column
        for column in validation.columns
        if column not in {
            "unit_id",
            "cycle",
            "RUL",
        }
    ]

    return validation, feature_columns


def load_model():
    """Load the existing XGBoost pipeline."""

    import joblib

    return joblib.load(MODEL_PATH)


def calculate_prediction_analysis(
    validation,
    feature_columns,
    shap_df,
    model,
):
    """Calculate prediction and SHAP contribution diagnostics."""

    X = validation[feature_columns]

    predictions = np.asarray(
        model.predict(X)
    )

    shap_matrix = shap_df[
        feature_columns
    ].to_numpy()

    if len(predictions) != len(validation):
        raise ValueError(
            "Prediction count does not match validation rows."
        )

    if shap_matrix.shape != (
        len(validation),
        len(feature_columns),
    ):
        raise ValueError(
            "SHAP matrix shape does not match validation data."
        )

    actual_rul = validation["RUL"].to_numpy()

    prediction_error = (
        predictions - actual_rul
    )

    positive_shap = np.where(
        shap_matrix > 0,
        shap_matrix,
        0,
    ).sum(axis=1)

    negative_shap = np.where(
        shap_matrix < 0,
        shap_matrix,
        0,
    ).sum(axis=1)

    absolute_shap = np.abs(
        shap_matrix
    ).sum(axis=1)

    result = pd.DataFrame(
        {
            "unit_id": validation["unit_id"],
            "cycle": validation["cycle"],
            "actual_rul": actual_rul,
            "predicted_rul": predictions,
            "prediction_error": prediction_error,
            "absolute_error": np.abs(
                prediction_error
            ),
            "positive_shap_contribution": positive_shap,
            "negative_shap_contribution": negative_shap,
            "absolute_shap_contribution": absolute_shap,
        }
    )

    result["error_direction"] = np.where(
        result["prediction_error"] > 0,
        "over_prediction",
        np.where(
            result["prediction_error"] < 0,
            "under_prediction",
            "exact",
        ),
    )

    return result


def calculate_summary(result):
    """Calculate aggregate SHAP prediction diagnostics."""

    correlation_error_abs_shap = (
        result[
            [
                "absolute_error",
                "absolute_shap_contribution",
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    correlation_signed_error_shap = (
        result[
            [
                "prediction_error",
                "positive_shap_contribution",
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "validation_rows",
                "mean_absolute_error",
                "mean_prediction_error",
                "mean_absolute_shap_contribution",
                "mean_positive_shap_contribution",
                "mean_negative_shap_contribution",
                "over_prediction_rows",
                "under_prediction_rows",
                "exact_prediction_rows",
                "error_abs_shap_correlation",
                "error_positive_shap_correlation",
            ],
            "value": [
                len(result),
                result["absolute_error"].mean(),
                result["prediction_error"].mean(),
                result[
                    "absolute_shap_contribution"
                ].mean(),
                result[
                    "positive_shap_contribution"
                ].mean(),
                result[
                    "negative_shap_contribution"
                ].mean(),
                (
                    result["error_direction"]
                    == "over_prediction"
                ).sum(),
                (
                    result["error_direction"]
                    == "under_prediction"
                ).sum(),
                (
                    result["error_direction"]
                    == "exact"
                ).sum(),
                correlation_error_abs_shap,
                correlation_signed_error_shap,
            ],
        }
    )

    return summary


def validate_result(result, summary):
    """Validate generated analysis."""

    if result.empty:
        raise ValueError(
            "Prediction analysis is empty."
        )

    if result.isna().any().any():
        raise ValueError(
            "Prediction analysis contains NaN values."
        )

    required_columns = {
        "unit_id",
        "cycle",
        "actual_rul",
        "predicted_rul",
        "prediction_error",
        "absolute_error",
        "positive_shap_contribution",
        "negative_shap_contribution",
        "absolute_shap_contribution",
        "error_direction",
    }

    missing = required_columns - set(
        result.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if len(summary) != 11:
        raise ValueError(
            "Unexpected summary metric count."
        )


def main():

    print("=" * 70)
    print("SHAP-BASED RUL PREDICTION ANALYSIS")
    print("=" * 70)

    print("\n[1/5] Loading inputs...")

    features, split, shap_df = load_inputs()

    print(
        f"Feature data: {features.shape}"
    )

    print(
        f"SHAP data: {shap_df.shape}"
    )

    print("\n[2/5] Preparing validation data...")

    validation, feature_columns = (
        prepare_validation(
            features,
            split,
        )
    )

    print(
        f"Validation rows: {len(validation)}"
    )

    print(
        f"Model features: {len(feature_columns)}"
    )

    print("\n[3/5] Loading model...")

    model = load_model()

    print("Model loaded successfully.")

    print("\n[4/5] Calculating SHAP prediction analysis...")

    result = calculate_prediction_analysis(
        validation,
        feature_columns,
        shap_df,
        model,
    )

    summary = calculate_summary(
        result
    )

    validate_result(
        result,
        summary,
    )

    print("\n[5/5] Saving reports...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary_path = (
        OUTPUT_PATH.parent
        / "shap_prediction_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nDetailed report: {OUTPUT_PATH}"
    )

    print(
        f"Summary report: {summary_path}"
    )

    print("\nPrediction analysis summary:")

    print(
        summary.to_string(index=False)
    )

    print(
        "\nSHAP prediction analysis completed successfully."
    )


if __name__ == "__main__":
    main()