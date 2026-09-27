from pathlib import Path

import pandas as pd

from src.data.load_data import load_train_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REPORTS_DIR / "sensor_quality.csv"


def analyze_sensor_quality(train: pd.DataFrame) -> pd.DataFrame:
    """Analyze variability and basic quality of every sensor."""

    sensors = [f"sensor_{i}" for i in range(1, 22)]

    records = []

    for sensor in sensors:
        series = train[sensor]

        records.append(
            {
                "sensor": sensor,
                "unique_values": series.nunique(),
                "missing_values": series.isnull().sum(),
                "mean": series.mean(),
                "std": series.std(),
                "variance": series.var(),
                "minimum": series.min(),
                "maximum": series.max(),
                "range": series.max() - series.min(),
                "zero_variance": series.var() == 0,
            }
        )

    result = pd.DataFrame(records)

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    return result


def print_summary(result: pd.DataFrame) -> None:
    """Print sensor-quality summary."""

    print("\nSENSOR QUALITY ANALYSIS")
    print("=" * 70)

    print("\nSensor statistics:")
    print(
        result[
            [
                "sensor",
                "unique_values",
                "mean",
                "std",
                "minimum",
                "maximum",
                "range",
                "zero_variance",
            ]
        ].to_string(index=False)
    )

    constant_sensors = result[
        result["zero_variance"]
    ]["sensor"].tolist()

    print("\nConstant sensors:")
    print(constant_sensors)

    print(
        f"\nNumber of constant sensors: "
        f"{len(constant_sensors)}"
    )

    print("\nLowest-variance sensors:")

    print(
        result.sort_values("variance")
        [
            [
                "sensor",
                "variance",
                "std",
                "unique_values",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nHighest-variance sensors:")

    print(
        result.sort_values(
            "variance",
            ascending=False,
        )
        [
            [
                "sensor",
                "variance",
                "std",
                "unique_values",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nReport created:")
    print(OUTPUT_PATH)


def main():
    train = load_train_data()

    result = analyze_sensor_quality(train)

    print_summary(result)


if __name__ == "__main__":
    main()
    