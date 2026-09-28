from pathlib import Path

import pandas as pd

from src.data.load_data import load_train_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REPORTS_DIR / "sensor_degradation_analysis.csv"


def analyze_sensor_trends(train: pd.DataFrame) -> pd.DataFrame:
    """Analyze sensor relationships with engine cycle."""

    sensors = [
        f"sensor_{i}"
        for i in range(1, 22)
    ]

    records = []

    for sensor in sensors:

        # Constant sensors cannot have a meaningful correlation.
        if train[sensor].nunique() <= 1:
            pearson = None
            spearman = None

        else:
            pearson = train[sensor].corr(
                train["cycle"],
                method="pearson",
            )

            spearman = train[sensor].corr(
                train["cycle"],
                method="spearman",
            )

        records.append(
            {
                "sensor": sensor,
                "pearson_cycle_corr": pearson,
                "spearman_cycle_corr": spearman,
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

    # Convert collected records into a DataFrame.
    result = pd.DataFrame(records)

    # Sort sensors by absolute Spearman correlation.
    result = result.sort_values(
        "absolute_spearman",
        ascending=False,
        na_position="last",
    )

    # Save report.
    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    return result


def print_summary(result: pd.DataFrame) -> None:
    """Print strongest and weakest sensor-cycle relationships."""

    print("\nSENSOR DEGRADATION ANALYSIS")
    print("=" * 70)

    print(
        "\nSensors with strongest Spearman relationship to cycle:"
    )

    strongest = result.head(10)

    print(
        strongest[
            [
                "sensor",
                "pearson_cycle_corr",
                "spearman_cycle_corr",
            ]
        ].to_string(index=False)
    )

    print(
        "\nSensors with weakest Spearman relationship to cycle:"
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
                "pearson_cycle_corr",
                "spearman_cycle_corr",
            ]
        ].to_string(index=False)
    )

    print("\nReport created:")
    print(OUTPUT_PATH)


def main():
    """Run sensor degradation analysis."""

    train = load_train_data()

    result = analyze_sensor_trends(train)

    print_summary(result)


if __name__ == "__main__":
    main()