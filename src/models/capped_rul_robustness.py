from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

BINS = [-1, 20, 50, 100, 150, 200, 250, float("inf")]
LABELS = ["0-20", "21-50", "51-100", "101-150", "151-200", "201-250", "251+"]


def main():
    original = pd.read_csv(REPORTS / "xgboost_validation_predictions.csv")
    capped = pd.read_csv(REPORTS / "xgboost_capped_rul_predictions.csv")

    original["range"] = pd.cut(original.RUL, BINS, labels=LABELS)
    capped["range"] = pd.cut(capped.RUL, BINS, labels=LABELS)

    rows = []

    for label in LABELS:
        o = original[original["range"] == label]
        c = capped[capped["range"] == label]

        rows.append({
            "RUL_range": label,
            "samples": len(o),
            "original_MAE": mean_absolute_error(o.RUL, o.predicted_RUL),
            "capped_MAE": mean_absolute_error(c.RUL, c.predicted_RUL),
            "original_RMSE": mean_squared_error(
                o.RUL, o.predicted_RUL
            ) ** 0.5,
            "capped_RMSE": mean_squared_error(
                c.RUL, c.predicted_RUL
            ) ** 0.5,
            "original_bias": (o.predicted_RUL - o.RUL).mean(),
            "capped_bias": (c.predicted_RUL - c.RUL).mean()
        })

    result = pd.DataFrame(rows)
    result["MAE_change"] = result.original_MAE - result.capped_MAE

    result.to_csv(
        REPORTS / "day8_rul_robustness_comparison.csv",
        index=False
    )

    original_mae = mean_absolute_error(
        original.RUL, original.predicted_RUL
    )
    capped_mae = mean_absolute_error(
        capped.RUL, capped.predicted_RUL
    )

    original_rmse = mean_squared_error(
        original.RUL, original.predicted_RUL
    ) ** 0.5
    capped_rmse = mean_squared_error(
        capped.RUL, capped.predicted_RUL
    ) ** 0.5

    print(result.to_string(index=False))

    print("\nOverall comparison")
    print(f"Original MAE: {original_mae:.4f}")
    print(f"Capped MAE:   {capped_mae:.4f}")
    print(f"MAE change:   {original_mae - capped_mae:.4f}")
    print(f"Original RMSE: {original_rmse:.4f}")
    print(f"Capped RMSE:   {capped_rmse:.4f}")
    print(f"RMSE change:   {original_rmse - capped_rmse:.4f}")


if __name__ == "__main__":
    main()