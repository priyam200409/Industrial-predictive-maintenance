from pathlib import Path

import joblib
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
    / "engine_shap_explanations.csv"
)


def load_inputs():
    """Load feature data, model, split metadata and SHAP values."""

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
    model = joblib.load(MODEL_PATH)
    split = pd.read_csv(SPLIT_PATH)
    shap_df = pd.read_csv(SHAP_PATH)

    return features, model, split, shap_df


def prepare_validation_data(features, split):
    """Prepare validation rows using the deterministic Day 4 split."""

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
        if column not in {"unit_id", "cycle", "RUL"}
    ]

    X = validation[feature_columns]

    return validation, X, feature_columns


def calculate_predictions(model, X):
    """Generate RUL predictions using the existing model."""

    predictions = model.predict(X)

    return np.asarray(predictions)


def align_shap_values(validation, shap_df, feature_columns):
    """Align stored SHAP values with validation rows."""

    metadata = validation[
        ["unit_id", "cycle", "RUL"]
    ].reset_index(drop=True)

    stored_metadata = shap_df[
        ["unit_id", "cycle", "RUL"]
    ].reset_index(drop=True)

    if not metadata.equals(stored_metadata):
        raise ValueError(
            "SHAP metadata does not match validation data."
        )

    shap_matrix = shap_df[
        feature_columns
    ].to_numpy()

    if shap_matrix.shape != (
        len(validation),
        len(feature_columns),
    ):
        raise ValueError(
            "SHAP matrix shape does not match validation data."
        )

    return shap_matrix


def select_representative_engines(validation, predictions):
    """Select engines representing different prediction-error patterns."""

    results = validation[
        ["unit_id", "cycle", "RUL"]
    ].copy()

    results["predicted_rul"] = predictions

    results["prediction_error"] = (
        results["predicted_rul"] - results["RUL"]
    )

    results["absolute_error"] = (
        results["prediction_error"].abs()
    )

    engine_error = (
        results.groupby("unit_id")["absolute_error"]
        .mean()
        .sort_values()
    )

    selected = []

    # Lowest-error engine
    selected.append(engine_error.index[0])

    # Highest-error engine
    selected.append(engine_error.index[-1])

    # Median-error engine
    median_position = len(engine_error) // 2
    selected.append(
        engine_error.index[median_position]
    )

    # Remove duplicates while preserving order
    selected = list(dict.fromkeys(selected))

    return selected


def build_engine_explanations(
    validation,
    predictions,
    shap_matrix,
    feature_columns,
    selected_engines,
):
    """Build engine-level SHAP explanations."""

    rows = []

    for engine_id in selected_engines:

        engine_indices = validation.index[
            validation["unit_id"] == engine_id
        ].tolist()

        for idx in engine_indices:

            contributions = pd.Series(
                shap_matrix[idx],
                index=feature_columns,
            )

            top_positive = (
                contributions
                .sort_values(ascending=False)
                .head(5)
            )

            top_negative = (
                contributions
                .sort_values(ascending=True)
                .head(5)
            )

            top_overall = (
                contributions.abs()
                .sort_values(ascending=False)
                .head(10)
            )

            row = {
                "unit_id": int(
                    validation.loc[idx, "unit_id"]
                ),
                "cycle": int(
                    validation.loc[idx, "cycle"]
                ),
                "actual_rul": float(
                    validation.loc[idx, "RUL"]
                ),
                "predicted_rul": float(
                    predictions[idx]
                ),
                "prediction_error": float(
                    predictions[idx]
                    - validation.loc[idx, "RUL"]
                ),
            }

            for rank, (feature, value) in enumerate(
                top_positive.items(),
                start=1,
            ):
                row[
                    f"positive_feature_{rank}"
                ] = feature

                row[
                    f"positive_shap_{rank}"
                ] = float(value)

            for rank, (feature, value) in enumerate(
                top_negative.items(),
                start=1,
            ):
                row[
                    f"negative_feature_{rank}"
                ] = feature

                row[
                    f"negative_shap_{rank}"
                ] = float(value)

            for rank, (feature, value) in enumerate(
                top_overall.items(),
                start=1,
            ):
                row[
                    f"top_feature_{rank}"
                ] = feature

                row[
                    f"top_abs_shap_{rank}"
                ] = float(
                    contributions[feature]
                )

            rows.append(row)

    return pd.DataFrame(rows)


def validate_output(result):
    """Validate engine-level explanation output."""

    if result.empty:
        raise ValueError(
            "Engine SHAP explanation report is empty."
        )

    required_columns = {
        "unit_id",
        "cycle",
        "actual_rul",
        "predicted_rul",
        "prediction_error",
        "positive_feature_1",
        "positive_shap_1",
        "negative_feature_1",
        "negative_shap_1",
        "top_feature_1",
        "top_abs_shap_1",
    }

    missing = required_columns - set(
        result.columns
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if result.isna().any().any():
        raise ValueError(
            "Engine SHAP report contains NaN values."
        )

    if result["unit_id"].nunique() < 3:
        raise ValueError(
            "Expected explanations for at least 3 engines."
        )


def main():

    print("=" * 70)
    print("ENGINE-LEVEL SHAP ANALYSIS")
    print("=" * 70)

    print("\n[1/6] Loading inputs...")

    features, model, split, shap_df = load_inputs()

    print(
        f"Feature data: {features.shape}"
    )

    print(
        f"Stored SHAP data: {shap_df.shape}"
    )

    print("\n[2/6] Preparing validation data...")

    validation, X, feature_columns = (
        prepare_validation_data(
            features,
            split,
        )
    )

    print(
        f"Validation rows: {len(validation)}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    print("\n[3/6] Generating predictions...")

    predictions = calculate_predictions(
        model,
        X,
    )

    print(
        f"Predictions generated: {len(predictions)}"
    )

    print("\n[4/6] Aligning SHAP values...")

    shap_matrix = align_shap_values(
        validation,
        shap_df,
        feature_columns,
    )

    print(
        f"SHAP matrix: {shap_matrix.shape}"
    )

    print("\n[5/6] Selecting representative engines...")

    selected_engines = (
        select_representative_engines(
            validation,
            predictions,
        )
    )

    print(
        f"Selected engines: {selected_engines}"
    )

    print("\n[6/6] Building explanations...")

    result = build_engine_explanations(
        validation,
        predictions,
        shap_matrix,
        feature_columns,
        selected_engines,
    )

    validate_output(result)

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

    print(
        f"Explanation rows: {len(result)}"
    )

    print(
        f"Engines explained: "
        f"{result['unit_id'].nunique()}"
    )

    print("\nEngine-level analysis completed successfully.")


if __name__ == "__main__":
    main()