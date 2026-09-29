from pathlib import Path

import pandas as pd

from src.data.load_data import load_train_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "train_filtered_sensors.csv"

SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]


def identify_constant_sensors(
    train: pd.DataFrame,
) -> list[str]:
    """Identify sensors with no meaningful variation."""

    constant_sensors = []

    for sensor in SENSOR_COLUMNS:
        if train[sensor].nunique(dropna=False) <= 1:
            constant_sensors.append(sensor)

    return constant_sensors


def remove_constant_sensors(
    train: pd.DataFrame,
    constant_sensors: list[str],
) -> pd.DataFrame:
    """Remove constant sensors from the dataset."""

    data = train.copy()

    data = data.drop(
        columns=constant_sensors,
    )

    return data


def validate_sensor_filtering(
    original: pd.DataFrame,
    filtered: pd.DataFrame,
    removed_sensors: list[str],
) -> None:
    """Validate sensor filtering."""

    for sensor in removed_sensors:
        assert sensor not in filtered.columns

    assert len(filtered) == len(original)

    assert (
        filtered["unit_id"]
        .equals(original["unit_id"])
    )

    assert (
        filtered["cycle"]
        .equals(original["cycle"])
    )

    remaining_sensors = [
        sensor
        for sensor in SENSOR_COLUMNS
        if sensor not in removed_sensors
    ]

    for sensor in remaining_sensors:
        assert sensor in filtered.columns

        assert (
            filtered[sensor]
            .equals(original[sensor])
        )


def main():
    """Run sensor filtering pipeline."""

    train = load_train_data()

    constant_sensors = identify_constant_sensors(
        train,
    )

    filtered = remove_constant_sensors(
        train,
        constant_sensors,
    )

    validate_sensor_filtering(
        train,
        filtered,
        constant_sensors,
    )

    filtered.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSENSOR FILTERING")
    print("=" * 70)

    print(
        f"Original sensor count : "
        f"{len(SENSOR_COLUMNS)}"
    )

    print(
        f"Removed sensor count  : "
        f"{len(constant_sensors)}"
    )

    print("\nRemoved sensors:")

    for sensor in constant_sensors:
        print(f"- {sensor}")

    remaining = [
        sensor
        for sensor in SENSOR_COLUMNS
        if sensor not in constant_sensors
    ]

    print(
        f"\nRemaining sensor count : "
        f"{len(remaining)}"
    )

    print("\nValidation: PASSED")

    print(f"\nSaved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
    