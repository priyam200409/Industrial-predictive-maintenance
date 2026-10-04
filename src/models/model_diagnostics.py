from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    predictions_path = REPORTS_DIR / "xgboost_validation_predictions.csv"
    predictions = pd.read_csv(predictions_path)

    required_columns = {
        "unit_id",
        "cycle",
        "RUL",
        "predicted_RUL",
        "error",
        "absolute_error",
    }

    missing = required_columns - set(predictions.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    predictions["bias_direction"] = predictions["error"].apply(
        lambda x: "Over-prediction" if x > 0 else "Under-prediction"
    )

    predictions["RUL_range"] = pd.cut(
        predictions["RUL"],
        bins=[-1, 20, 50, 100, 150, 200, 250, float("inf")],
        labels=[
            "0-20",
            "21-50",
            "51-100",
            "101-150",
            "151-200",
            "201-250",
            "251+",
        ],
    )

    lifecycle = (
        predictions.groupby("RUL_range", observed=False)
        .agg(
            samples=("RUL", "size"),
            engines=("unit_id", "nunique"),
            actual_mean=("RUL", "mean"),
            predicted_mean=("predicted_RUL", "mean"),
            MAE=("absolute_error", "mean"),
            mean_error=("error", "mean"),
            max_absolute_error=("absolute_error", "max"),
        )
        .reset_index()
    )

    lifecycle["bias"] = lifecycle["predicted_mean"] - lifecycle["actual_mean"]

    engine = (
        predictions.groupby("unit_id")
        .agg(
            samples=("RUL", "size"),
            MAE=("absolute_error", "mean"),
            RMSE=("error", lambda x: (x.pow(2).mean()) ** 0.5),
            mean_error=("error", "mean"),
            max_absolute_error=("absolute_error", "max"),
            mean_actual_RUL=("RUL", "mean"),
            mean_predicted_RUL=("predicted_RUL", "mean"),
        )
        .reset_index()
    )

    engine["bias"] = (
        engine["mean_predicted_RUL"] - engine["mean_actual_RUL"]
    )

    engine = engine.sort_values("MAE", ascending=False)

    lifecycle_path = REPORTS_DIR / "day8_lifecycle_diagnostics.csv"
    engine_path = REPORTS_DIR / "day8_engine_diagnostics.csv"

    lifecycle.to_csv(lifecycle_path, index=False)
    engine.to_csv(engine_path, index=False)

    print("=" * 70)
    print("DAY 8 MODEL DIAGNOSTICS")
    print("=" * 70)

    print("\nLifecycle diagnostics:")
    print(lifecycle.to_string(index=False))

    print("\nWorst engines by MAE:")
    print(engine.head(10).to_string(index=False))

    print("\nValidation checks:")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Validation engines: {predictions['unit_id'].nunique()}")
    print(f"Lifecycle rows: {len(lifecycle)}")
    print(f"Engine rows: {len(engine)}")
    print(
        f"Lifecycle NaNs: {lifecycle.isna().sum().sum()}"
    )
    print(
        f"Engine NaNs: {engine.isna().sum().sum()}"
    )

    if lifecycle.isna().sum().sum() != 0:
        raise ValueError("Lifecycle diagnostics contain NaN values.")

    if engine.isna().sum().sum() != 0:
        raise ValueError("Engine diagnostics contain NaN values.")

    if len(predictions) != 4070:
        raise ValueError(
            f"Expected 4070 validation rows, got {len(predictions)}."
        )

    if predictions["unit_id"].nunique() != 20:
        raise ValueError(
            "Expected 20 validation engines."
        )

    print("\nValidation passed.")


if __name__ == "__main__":
    main()