from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_rul_model.joblib"
SPLIT_PATH = PROJECT_ROOT / "reports" / "train_validation_split.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "shap"
OUTPUT_PATH = OUTPUT_DIR / "validation_shap_values.csv"


def load_feature_data():
    """Load the unified feature dataset."""
    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_DATA_PATH}"
        )

    return pd.read_csv(FEATURE_DATA_PATH)


def load_model():
    """Load the trained XGBoost pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_validation_engines():
    """Load the deterministic validation engine split."""
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Validation split not found: {SPLIT_PATH}"
        )

    split_df = pd.read_csv(SPLIT_PATH)

    validation_engines = split_df.loc[
        split_df["split"] == "validation",
        "unit_id",
    ].tolist()

    if not validation_engines:
        raise ValueError("No validation engines found.")

    return validation_engines


def prepare_validation_data(df, validation_engines):
    """Prepare validation features using the same feature selection as training."""

    validation_df = df[df["unit_id"].isin(validation_engines)].copy()

    if validation_df.empty:
        raise ValueError("Validation dataset is empty.")

    excluded_columns = {
        "unit_id",
        "cycle",
        "RUL",
    }

    feature_columns = [
        column
        for column in validation_df.columns
        if column not in excluded_columns
    ]

    X_validation = validation_df[feature_columns].copy()

    return validation_df, X_validation, feature_columns


def create_shap_values(model, X_validation):
    """Calculate SHAP values using the XGBoost regressor inside the pipeline."""

    if not hasattr(model, "named_steps"):
        raise TypeError("Expected a sklearn pipeline.")

    if "regressor" not in model.named_steps:
        raise KeyError(
            "Expected pipeline step named 'regressor'."
        )

    regressor = model.named_steps["regressor"]

    # Apply the same preprocessing used during model training.
    X_transformed = model.named_steps["imputer"].transform(X_validation)

    explainer = shap.TreeExplainer(regressor)

    shap_values = explainer.shap_values(X_transformed)

    return np.asarray(shap_values)


def build_shap_report(validation_df, X_validation, shap_values):
    """Build a row-level SHAP values dataframe."""

    shap_df = pd.DataFrame(
        shap_values,
        columns=X_validation.columns,
        index=X_validation.index,
    )

    metadata = validation_df[
        ["unit_id", "cycle", "RUL"]
    ].copy()

    metadata.index = X_validation.index

    result = pd.concat(
        [
            metadata,
            shap_df,
        ],
        axis=1,
    )

    return result.reset_index(drop=True)


def validate_output(result, feature_columns):
    """Validate SHAP output structure."""

    expected_rows = len(result)

    expected_columns = {
        "unit_id",
        "cycle",
        "RUL",
        *feature_columns,
    }

    actual_columns = set(result.columns)

    if actual_columns != expected_columns:
        missing = expected_columns - actual_columns
        extra = actual_columns - expected_columns

        raise ValueError(
            f"SHAP columns mismatch. Missing: {missing}, Extra: {extra}"
        )

    if expected_rows == 0:
        raise ValueError("SHAP output contains no rows.")

    if result[feature_columns].isna().any().any():
        raise ValueError("SHAP output contains NaN values.")

    return True


def main():
    print("=" * 70)
    print("SHAP EXPLAINABILITY PIPELINE")
    print("=" * 70)

    print("\n[1/6] Loading feature dataset...")
    df = load_feature_data()

    print(f"Feature dataset shape: {df.shape}")

    print("\n[2/6] Loading XGBoost model...")
    model = load_model()

    print("Model loaded successfully.")

    print("\n[3/6] Loading validation engine split...")
    validation_engines = load_validation_engines()

    print(
        f"Validation engines: {len(validation_engines)}"
    )

    print("\n[4/6] Preparing validation data...")
    validation_df, X_validation, feature_columns = (
        prepare_validation_data(
            df,
            validation_engines,
        )
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    print(
        f"Model features: {len(feature_columns)}"
    )

    print("\n[5/6] Calculating SHAP values...")

    shap_values = create_shap_values(
        model,
        X_validation,
    )

    print(
        f"SHAP matrix shape: {shap_values.shape}"
    )

    print("\n[6/6] Building and validating report...")

    result = build_shap_report(
        validation_df,
        X_validation,
        shap_values,
    )

    validate_output(
        result,
        feature_columns,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSHAP analysis completed successfully.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows: {len(result)}")
    print(f"Columns: {len(result.columns)}")

    print("\nValidation:")
    print("✓ Validation engines preserved")
    print("✓ Feature count validated")
    print("✓ SHAP values validated")
    print("✓ No NaN SHAP values")


if __name__ == "__main__":
    main()