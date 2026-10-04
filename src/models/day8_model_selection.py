from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def main():
    experiment = pd.read_csv(
        REPORTS / "day8_capped_rul_experiment.csv"
    )

    robustness = pd.read_csv(
        REPORTS / "day8_rul_robustness_comparison.csv"
    )

    original = experiment.iloc[0]

    capped = experiment.loc[
        experiment["RUL_cap"].astype(str) == "200"
    ].iloc[0]

    high_rul = robustness[
        robustness["RUL_range"].isin(["201-250", "251+"])
    ]

    result = pd.DataFrame([
        ["Original MAE", original["MAE"]],
        ["Cap-200 MAE", capped["MAE"]],
        ["MAE improvement", original["MAE"] - capped["MAE"]],
        ["Original RMSE", original["RMSE"]],
        ["Cap-200 RMSE", capped["RMSE"]],
        ["RMSE improvement", original["RMSE"] - capped["RMSE"]],
        ["Original high-RUL MAE", high_rul["original_MAE"].mean()],
        ["Cap-200 high-RUL MAE", high_rul["capped_MAE"].mean()],
        ["Selected model", "Original XGBoost"],
    ], columns=["metric", "value"])

    result.to_csv(
        REPORTS / "day8_model_selection.csv",
        index=False
    )

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()