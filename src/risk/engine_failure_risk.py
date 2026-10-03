from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "xgboost_validation_predictions.csv"
)

THRESHOLDS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "failure_risk_thresholds.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "engine_failure_risk.csv"
)


def load_inputs():
    """Load validation predictions and risk thresholds."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction report not found: {PREDICTIONS_PATH}"
        )

    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Threshold report not found: {THRESHOLDS_PATH}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    thresholds = pd.read_csv(
        THRESHOLDS_PATH
    )

    return predictions, thresholds


def classify_risk(predicted_rul):
    """Classify predicted RUL into a project risk level."""

    if predicted_rul < 0:
        return "Critical"

    if predicted_rul <= 20:
        return "Critical"

    if predicted_rul <= 50:
        return "Warning"

    return "Normal"


def calculate_risk_score(risk_level):
    """Convert risk level into project-defined priority score."""

    scores = {
        "Normal": 0,
        "Warning": 50,
        "Critical": 100,
    }

    return scores[risk_level]


def build_risk_indicators(df):
    """Build observation-level failure-risk indicators."""

    result = df.copy()

    result["risk_level"] = (
        result["predicted_RUL"]
        .apply(classify_risk)
    )

    result["risk_score"] = (
        result["risk_level"]
        .apply(calculate_risk_score)
    )

    result["error_direction"] = np.where(
        result["error"] > 0,
        "over_prediction",
        np.where(
            result["error"] < 0,
            "under_prediction",
            "exact",
        ),
    )

    return result[
        [
            "unit_id",
            "cycle",
            "RUL",
            "predicted_RUL",
            "error",
            "absolute_error",
            "error_direction",
            "risk_level",
            "risk_score",
        ]
    ]


def validate_result(result):
    """Validate failure-risk indicators."""

    if result.empty:
        raise ValueError(
            "Failure-risk report is empty."
        )

    required_columns = {
        "unit_id",
        "cycle",
        "RUL",
        "predicted_RUL",
        "error",
        "absolute_error",
        "error_direction",
        "risk_level",
        "risk_score",
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
            "Failure-risk report contains NaN values."
        )

    valid_levels = {
        "Normal",
        "Warning",
        "Critical",
    }

    if not set(result["risk_level"]).issubset(
        valid_levels
    ):
        raise ValueError(
            "Invalid risk level detected."
        )

    expected_scores = {
        "Normal": 0,
        "Warning": 50,
        "Critical": 100,
    }

    for level, score in expected_scores.items():

        actual = result.loc[
            result["risk_level"] == level,
            "risk_score",
        ]

        if not (actual == score).all():
            raise ValueError(
                f"Incorrect risk score for {level}."
            )

    if len(result) != 4070:
        raise ValueError(
            f"Expected 4070 validation rows, got {len(result)}."
        )


def main():

    print("=" * 70)
    print("ENGINE-LEVEL FAILURE-RISK ANALYSIS")
    print("=" * 70)

    print("\n[1/4] Loading inputs...")

    predictions, thresholds = load_inputs()

    print(
        f"Validation rows: {len(predictions)}"
    )

    print(
        f"Threshold definitions: {len(thresholds)}"
    )

    print("\n[2/4] Calculating risk indicators...")

    result = build_risk_indicators(
        predictions
    )

    print(
        "Risk indicators calculated."
    )

    print("\n[3/4] Validating results...")

    validate_result(result)

    print(
        "Validation passed."
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

    print("\nRisk-level distribution:")

    print(
        result["risk_level"]
        .value_counts()
        .reindex(
            ["Normal", "Warning", "Critical"],
            fill_value=0,
        )
        .to_string()
    )

    print(
        "\nEngine-level failure-risk analysis "
        "completed successfully."
    )


if __name__ == "__main__":
    main()