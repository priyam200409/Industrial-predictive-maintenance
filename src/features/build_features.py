from pathlib import Path

import pandas as pd

from src.data.load_data import load_train_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "train_features.csv"
INVENTORY_PATH = REPORTS_DIR / "feature_inventory.csv"


CONSTANT_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

SENSORS = [
    f"sensor_{i}"
    for i in range(1, 22)
    if f"sensor_{i}" not in CONSTANT_SENSORS
]


def create_rul(data: pd.DataFrame) -> pd.DataFrame:
    """Create engine-specific RUL."""

    data = data.copy()

    max_cycles = (
        data.groupby("unit_id")["cycle"]
        .max()
        .rename("max_cycle")
    )

    data = data.merge(
        max_cycles,
        on="unit_id",
        how="left",
    )

    data["RUL"] = (
        data["max_cycle"] - data["cycle"]
    )

    data.drop(
        columns=["max_cycle"],
        inplace=True,
    )

    return data


def add_rolling_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe rolling statistics."""

    data = data.copy()

    data = (
        data
        .sort_values(["unit_id", "cycle"])
        .reset_index(drop=True)
    )

    feature_frames = []

    for sensor in SENSORS:

        previous = (
            data
            .groupby("unit_id")[sensor]
            .shift(1)
        )

        sensor_features = {}

        for window in [5, 10, 20]:

            rolling = (
                previous
                .groupby(data["unit_id"])
                .rolling(
                    window=window,
                    min_periods=1,
                )
            )

            sensor_features[
                f"{sensor}_rolling_mean_{window}"
            ] = (
                rolling.mean()
                .reset_index(
                    level=0,
                    drop=True,
                )
                .reset_index(drop=True)
            )

            sensor_features[
                f"{sensor}_rolling_std_{window}"
            ] = (
                rolling.std()
                .reset_index(
                    level=0,
                    drop=True,
                )
                .reset_index(drop=True)
            )

        feature_frames.append(
            pd.DataFrame(
                sensor_features,
                index=data.index,
            )
        )

    features = pd.concat(
        feature_frames,
        axis=1,
    )

    return pd.concat(
        [data, features],
        axis=1,
    )


def add_trend_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe delta features."""

    data = data.copy()

    feature_frames = []

    grouped = data.groupby(
        "unit_id",
        sort=False,
    )

    for sensor in SENSORS:

        features = pd.DataFrame(
            index=data.index
        )

        previous_1 = grouped[sensor].shift(1)
        previous_5 = grouped[sensor].shift(5)

        features[
            f"{sensor}_delta_1"
        ] = data[sensor] - previous_1

        features[
            f"{sensor}_delta_5"
        ] = data[sensor] - previous_5

        feature_frames.append(features)

    features = pd.concat(
        feature_frames,
        axis=1,
    )

    return pd.concat(
        [data, features],
        axis=1,
    )


def create_feature_inventory(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create metadata for every final feature."""

    records = []

    for column in data.columns:

        if column in [
            "unit_id",
            "cycle",
        ]:
            feature_type = "identifier"

        elif column == "RUL":
            feature_type = "target"

        elif column.startswith("sensor_"):
            feature_type = "sensor_or_engineered"

        else:
            feature_type = "other"

        records.append(
            {
                "feature": column,
                "dtype": str(
                    data[column].dtype
                ),
                "missing_values": int(
                    data[column].isna().sum()
                ),
                "feature_type": feature_type,
            }
        )

    inventory = pd.DataFrame(records)

    inventory.to_csv(
        INVENTORY_PATH,
        index=False,
    )

    return inventory


def validate_pipeline(
    data: pd.DataFrame,
) -> None:
    """Validate final feature dataset."""

    assert len(data) == 20631

    assert data["unit_id"].nunique() == 100

    assert data["RUL"].min() == 0

    assert data["RUL"].max() == 361

    for sensor in CONSTANT_SENSORS:
        assert sensor not in data.columns

    assert data["unit_id"].notna().all()

    assert data["cycle"].notna().all()

    assert data["RUL"].notna().all()

    assert (
        data.groupby("unit_id")["cycle"]
        .is_monotonic_increasing
        .all()
    )


def main():
    """Run the complete feature-engineering pipeline."""

    print("\nBUILDING FINAL FEATURE DATASET")
    print("=" * 70)

    data = load_train_data()

    print(
        f"Raw columns       : {len(data.columns)}"
    )

    data = create_rul(data)

    print(
        f"After RUL         : {len(data.columns)}"
    )

    data = data.drop(
        columns=CONSTANT_SENSORS
    )

    print(
        f"After sensor filter: {len(data.columns)}"
    )

    data = add_rolling_features(data)

    print(
        f"After rolling      : {len(data.columns)}"
    )

    data = add_trend_features(data)

    print(
        f"After trends       : {len(data.columns)}"
    )

    validate_pipeline(data)

    create_feature_inventory(data)

    data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nFINAL FEATURE DATASET")
    print("=" * 70)

    print(f"Rows    : {len(data)}")
    print(f"Columns : {len(data.columns)}")
    print(
        f"Engines : {data['unit_id'].nunique()}"
    )

    print("\nValidation: PASSED")

    print(
        f"\nDataset saved to:\n{OUTPUT_PATH}"
    )

    print(
        f"\nFeature inventory:\n{INVENTORY_PATH}"
    )


if __name__ == "__main__":
    main()