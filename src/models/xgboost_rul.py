from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


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
METRICS_PATH = REPORTS_DIR / "xgboost_rul_metrics.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "RUL"

EXCLUDED_COLUMNS = {
    "unit_id",
    "cycle",
    TARGET_COLUMN,
}

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load engineered features and split metadata."""

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_PATH}"
        )

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split metadata not found: {SPLIT_PATH}"
        )

    data = pd.read_csv(FEATURE_PATH)
    split_metadata = pd.read_csv(SPLIT_PATH)

    return data, split_metadata


# ============================================================
# CREATE ENGINE-LEVEL SPLIT
# ============================================================

def create_split(
    data: pd.DataFrame,
    split_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the deterministic engine-level split."""

    train_engines = set(
        split_metadata.loc[
            split_metadata["split"] == "train",
            "unit_id",
        ]
    )

    validation_engines = set(
        split_metadata.loc[
            split_metadata["split"] == "validation",
            "unit_id",
        ]
    )

    train = data[
        data["unit_id"].isin(train_engines)
    ].copy()

    validation = data[
        data["unit_id"].isin(validation_engines)
    ].copy()

    return train, validation


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Prepare feature matrices and RUL targets."""

    feature_columns = [
        column
        for column in train.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X_train = train[feature_columns].copy()
    y_train = train[TARGET_COLUMN].copy()

    X_validation = validation[feature_columns].copy()
    y_validation = validation[TARGET_COLUMN].copy()

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


# ============================================================
# BUILD XGBOOST MODEL
# ============================================================

def build_model() -> Pipeline:
    """Build the XGBoost RUL regression pipeline."""

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict:
    """Calculate RUL regression metrics."""

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    return {
        "model": "XGBoost",
        "MAE": mae,
        "RMSE": rmse,
        "samples": len(y_true),
    }


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics: dict) -> None:
    """Save XGBoost metrics."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """Train and evaluate the XGBoost RUL model."""

    print("\nXGBOOST RUL MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data, split_metadata = load_data()

    print(
        f"Feature rows       : {len(data)}"
    )

    print(
        f"Feature columns    : {len(data.columns)}"
    )

    # --------------------------------------------------------
    # Recreate split
    # --------------------------------------------------------

    train, validation = create_split(
        data,
        split_metadata,
    )

    print(
        f"Training engines   : "
        f"{train['unit_id'].nunique()}"
    )

    print(
        f"Validation engines : "
        f"{validation['unit_id'].nunique()}"
    )

    print(
        f"Training rows      : {len(train)}"
    )

    print(
        f"Validation rows    : {len(validation)}"
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = prepare_features(
        train,
        validation,
    )

    print(
        f"\nModel features     : {X_train.shape[1]}"
    )

    print(
        f"Excluded columns   : "
        f"{sorted(EXCLUDED_COLUMNS)}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining XGBoost model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        X_validation
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_validation,
        predictions,
    )

    print("\nXGBOOST RESULTS")
    print("-" * 70)

    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"Samples: {metrics['samples']}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    save_metrics(
        metrics
    )

    print(
        f"\nSaved model:\n{MODEL_PATH}"
    )

    print(
        f"\nSaved metrics:\n{METRICS_PATH}"
    )


if __name__ == "__main__":
    main()