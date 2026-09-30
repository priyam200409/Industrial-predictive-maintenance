from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_PATH = PROCESSED_DIR / "train_features.csv"
SPLIT_PATH = REPORTS_DIR / "train_validation_split.csv"

MODEL_PATH = MODELS_DIR / "xgboost_rul_model.joblib"

PREDICTIONS_PATH = REPORTS_DIR / "xgboost_validation_predictions.csv"
ERROR_SUMMARY_PATH = REPORTS_DIR / "xgboost_error_summary.csv"
RUL_RANGE_PATH = REPORTS_DIR / "xgboost_error_by_rul_range.csv"
ENGINE_ERROR_PATH = REPORTS_DIR / "xgboost_engine_error.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "RUL"

EXCLUDED_COLUMNS = {
    "unit_id",
    "cycle",
    TARGET_COLUMN,
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load engineered features, split metadata and model."""

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_PATH}"
        )

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split metadata not found: {SPLIT_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {MODEL_PATH}"
        )

    data = pd.read_csv(FEATURE_PATH)
    split_metadata = pd.read_csv(SPLIT_PATH)

    model = joblib.load(MODEL_PATH)

    return data, split_metadata, model


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

def create_validation_data(
    data: pd.DataFrame,
    split_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Select only engines assigned to validation."""

    validation_engines = set(
        split_metadata.loc[
            split_metadata["split"] == "validation",
            "unit_id",
        ]
    )

    validation = data[
        data["unit_id"].isin(validation_engines)
    ].copy()

    validation = validation.sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True)

    return validation


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    validation: pd.DataFrame,
):
    """Prepare validation features and target."""

    feature_columns = [
        column
        for column in validation.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X_validation = validation[
        feature_columns
    ].copy()

    y_validation = validation[
        TARGET_COLUMN
    ].copy()

    return X_validation, y_validation


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    X_validation: pd.DataFrame,
) -> np.ndarray:
    """Generate validation RUL predictions."""

    predictions = model.predict(
        X_validation
    )

    return predictions


# ============================================================
# CREATE PREDICTION TABLE
# ============================================================

def create_prediction_table(
    validation: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Create detailed prediction and error table."""

    results = validation[
        ["unit_id", "cycle", "RUL"]
    ].copy()

    results["predicted_RUL"] = predictions

    results["error"] = (
        results["predicted_RUL"]
        - results["RUL"]
    )

    results["absolute_error"] = (
        results["error"].abs()
    )

    results["squared_error"] = (
        results["error"] ** 2
    )

    return results


# ============================================================
# OVERALL ERROR SUMMARY
# ============================================================

def create_error_summary(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate overall validation error metrics."""

    actual = predictions_df["RUL"]
    predicted = predictions_df["predicted_RUL"]

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    summary = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "samples": len(predictions_df),
                "MAE": mae,
                "RMSE": rmse,
                "mean_error": predictions_df[
                    "error"
                ].mean(),
                "median_absolute_error": predictions_df[
                    "absolute_error"
                ].median(),
                "max_absolute_error": predictions_df[
                    "absolute_error"
                ].max(),
            }
        ]
    )

    return summary


# ============================================================
# ERROR BY RUL RANGE
# ============================================================

def create_rul_range_analysis(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze prediction error across RUL ranges."""

    bins = [
        -1,
        20,
        50,
        100,
        150,
        200,
        250,
        500,
    ]

    labels = [
        "0-20",
        "21-50",
        "51-100",
        "101-150",
        "151-200",
        "201-250",
        "251+",
    ]

    analysis = predictions_df.copy()

    analysis["RUL_range"] = pd.cut(
        analysis["RUL"],
        bins=bins,
        labels=labels,
    )

    grouped = (
        analysis
        .groupby(
            "RUL_range",
            observed=False,
        )
        .agg(
            samples=("RUL", "size"),
            MAE=("absolute_error", "mean"),
            RMSE=(
                "squared_error",
                lambda x: np.sqrt(x.mean()),
            ),
            mean_error=("error", "mean"),
        )
        .reset_index()
    )

    return grouped


# ============================================================
# ENGINE-LEVEL ERROR ANALYSIS
# ============================================================

def create_engine_analysis(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate error statistics for every validation engine."""

    engine_analysis = (
        predictions_df
        .groupby("unit_id")
        .agg(
            samples=("RUL", "size"),
            mean_actual_RUL=("RUL", "mean"),
            mean_predicted_RUL=(
                "predicted_RUL",
                "mean",
            ),
            MAE=("absolute_error", "mean"),
            RMSE=(
                "squared_error",
                lambda x: np.sqrt(x.mean()),
            ),
            max_absolute_error=(
                "absolute_error",
                "max",
            ),
        )
        .reset_index()
    )

    return engine_analysis.sort_values(
        "MAE",
        ascending=False,
    )


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    predictions_df: pd.DataFrame,
    error_summary: pd.DataFrame,
    rul_range_analysis: pd.DataFrame,
    engine_analysis: pd.DataFrame,
) -> None:
    """Save all error-analysis outputs."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    error_summary.to_csv(
        ERROR_SUMMARY_PATH,
        index=False,
    )

    rul_range_analysis.to_csv(
        RUL_RANGE_PATH,
        index=False,
    )

    engine_analysis.to_csv(
        ENGINE_ERROR_PATH,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """Run complete XGBoost error analysis."""

    print("\nXGBOOST RUL ERROR ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data, split_metadata, model = load_data()

    print(
        f"Total feature rows  : {len(data)}"
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    validation = create_validation_data(
        data,
        split_metadata,
    )

    print(
        f"Validation engines  : "
        f"{validation['unit_id'].nunique()}"
    )

    print(
        f"Validation rows     : "
        f"{len(validation)}"
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_validation, y_validation = prepare_features(
        validation
    )

    print(
        f"Model features      : "
        f"{X_validation.shape[1]}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = generate_predictions(
        model,
        X_validation,
    )

    # --------------------------------------------------------
    # Prediction table
    # --------------------------------------------------------

    predictions_df = create_prediction_table(
        validation,
        predictions,
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    error_summary = create_error_summary(
        predictions_df
    )

    # --------------------------------------------------------
    # RUL range analysis
    # --------------------------------------------------------

    rul_range_analysis = create_rul_range_analysis(
        predictions_df
    )

    # --------------------------------------------------------
    # Engine analysis
    # --------------------------------------------------------

    engine_analysis = create_engine_analysis(
        predictions_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_reports(
        predictions_df,
        error_summary,
        rul_range_analysis,
        engine_analysis,
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nOVERALL ERROR SUMMARY")
    print("-" * 70)

    print(
        f"MAE                : "
        f"{error_summary.loc[0, 'MAE']:.4f}"
    )

    print(
        f"RMSE               : "
        f"{error_summary.loc[0, 'RMSE']:.4f}"
    )

    print(
        f"Mean error         : "
        f"{error_summary.loc[0, 'mean_error']:.4f}"
    )

    print(
        f"Median abs. error  : "
        f"{error_summary.loc[0, 'median_absolute_error']:.4f}"
    )

    print(
        f"Maximum abs. error : "
        f"{error_summary.loc[0, 'max_absolute_error']:.4f}"
    )

    print("\nERROR BY RUL RANGE")
    print("-" * 70)
    print(
        rul_range_analysis.to_string(
            index=False
        )
    )

    print("\nWORST VALIDATION ENGINES")
    print("-" * 70)

    print(
        engine_analysis.head(10).to_string(
            index=False
        )
    )

    print("\nSaved reports:")

    print(
        f"- {PREDICTIONS_PATH}"
    )

    print(
        f"- {ERROR_SUMMARY_PATH}"
    )

    print(
        f"- {RUL_RANGE_PATH}"
    )

    print(
        f"- {ENGINE_ERROR_PATH}"
    )


if __name__ == "__main__":
    main()