from pathlib import Path

import pandas as pd

from src.data.load_data import load_train_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REPORTS_DIR / "sensor_rul_analysis.csv"


def create_training_rul(train: pd.DataFrame) -> pd.DataFrame:
    """Calculate RUL for every training observation."""

    train = train.copy()

    max_cycles = (
        train.groupby("unit_id")["cycle"]
        .max()
        .rename("max_cycle")
    )

    train = train.merge(
        max_cycles,
        on="unit_id",
        how="left",
    )

    train["RUL"] = (
        train["max_cycle"] - train["cycle"]
    )

    train = train.drop(
        columns=["max_cycle"]
    )

    return train


def analyze_sensor_rul_relationship(
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate Pearson and Spearman correlation between sensors and RUL."""

    sensors = [
        f"sensor_{i}"
        for i in range(1, 22)
    ]

    records = []

    for sensor in sensors:

        if train[sensor].nunique() <= 1:
            pearson = None
            spearman = None

        else:
            pearson = train[sensor].corr(
                train["RUL"],
                method="pearson",
            )

            spearman = train[sensor].corr(
                train["RUL"],
                method="spearman",
            )

        records.append(
            {
                "sensor": sensor,
                "pearson_rul_corr": pearson,
                "spearman_rul_corr": spearman,
                "absolute_pearson": (
                    abs(pearson)
                    if pearson is not None
                    else None
                ),
                "absolute_spearman": (
                    abs(spearman)
                    if spearman is not None
                    else None
                ),
            }
        )

    result = pd.DataFrame(records)

    result = result.sort_values(
        "absolute_spearman",
        ascending=False,
        na_position="last",
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    return result


def print_summary(result: pd.DataFrame) -> None:
    """Print strongest and weakest sensor-RUL relationships."""

    print("\nSENSOR-RUL RELATIONSHIP ANALYSIS")
    print("=" * 70)

    print(
        "\nSensors with strongest Spearman relationship to RUL:"
    )

    strongest = result.head(10)

    print(
        strongest[
            [
                "sensor",
                "pearson_rul_corr",
                "spearman_rul_corr",
            ]
        ].to_string(index=False)
    )

    print(
        "\nSensors with weakest Spearman relationship to RUL:"
    )

    weakest = (
        result
        .dropna(subset=["absolute_spearman"])
        .sort_values(
            "absolute_spearman",
            ascending=True,
        )
        .head(10)
    )

    print(
        weakest[
            [
                "sensor",
                "pearson_rul_corr",
                "spearman_rul_corr",
            ]
        ].to_string(index=False)
    )

    print("\nReport created:")
    print(OUTPUT_PATH)


def main():
    """Run sensor-RUL analysis."""

    train = load_train_data()

    train = create_training_rul(train)

    print(
        f"Training observations: {len(train)}"
    )

    print(
        f"Training engines: "
        f"{train['unit_id'].nunique()}"
    )

    print(
        f"RUL minimum: "
        f"{train['RUL'].min()}"
    )

    print(
        f"RUL maximum: "
        f"{train['RUL'].max()}"
    )

    result = analyze_sensor_rul_relationship(
        train
    )

    print_summary(result)


if __name__ == "__main__":
    main()