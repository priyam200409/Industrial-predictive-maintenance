from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_train_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "train_with_rul.csv"
OUTPUT_PATH = PROCESSED_DIR / "train_trend_features.csv"


SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
    if i not in [1, 5, 10, 16, 18, 19]
]


def load_training_data() -> pd.DataFrame:
    """Load training data with RUL."""

    if INPUT_PATH.exists():
        return pd.read_csv(INPUT_PATH)

    data = load_train_data()

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


def calculate_group_slope(
    group: pd.DataFrame,
    sensor: str,
    window: int,
) -> pd.Series:
    """
    Calculate rolling linear slope using only previous observations.

    The current observation is excluded using shift(1).
    """

    values = (
        group[sensor]
        .shift(1)
    )

    cycles = (
        group["cycle"]
        .shift(1)
    )

    slopes = np.full(
        len(group),
        np.nan,
        dtype=float,
    )

    for i in range(len(group)):

        start = max(
            0,
            i - window + 1,
        )

        y = values.iloc[start:i + 1].dropna()
        x = cycles.iloc[start:i + 1].dropna()

        valid = (
            y.index.intersection(x.index)
        )

        if len(valid) < 2:
            continue

        x_values = x.loc[valid].to_numpy(
            dtype=float
        )

        y_values = y.loc[valid].to_numpy(
            dtype=float
        )

        if np.all(
            y_values == y_values[0]
        ):
            slopes[i] = 0.0
        else:
            slopes[i] = np.polyfit(
                x_values,
                y_values,
                1,
            )[0]

    return pd.Series(
        slopes,
        index=group.index,
    )


def create_trend_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create leakage-safe sensor trend features."""

    result = data.copy()

    result = (
        result
        .sort_values(
            ["unit_id", "cycle"]
        )
        .reset_index(drop=True)
    )

    feature_frames = []

    for sensor in SENSOR_COLUMNS:

        sensor_frame = pd.DataFrame(
            index=result.index
        )

        grouped = result.groupby(
            "unit_id",
            sort=False,
        )

        sensor_frame[
            f"{sensor}_delta_1"
        ] = (
            grouped[sensor]
            .shift(1)
        )

        sensor_frame[
            f"{sensor}_delta_1"
        ] = (
            result[sensor]
            - sensor_frame[
                f"{sensor}_delta_1"
            ]
        )

        sensor_frame[
            f"{sensor}_delta_5"
        ] = (
            result[sensor]
            - grouped[sensor].shift(5)
        )

        for window in [5, 20]:

            slopes = []

            for _, group in grouped:

                slope = calculate_group_slope(
                    group,
                    sensor,
                    window,
                )

                slopes.append(slope)

            slope_series = pd.concat(
                slopes
            ).sort_index()

            sensor_frame[
                f"{sensor}_slope_{window}"
            ] = slope_series

        feature_frames.append(
            sensor_frame
        )

    features = pd.concat(
        feature_frames,
        axis=1,
    )

    result = pd.concat(
        [
            result,
            features,
        ],
        axis=1,
    )

    return result


def validate_trend_features(
    original: pd.DataFrame,
    engineered: pd.DataFrame,
) -> None:
    """Validate trend feature generation."""

    assert len(original) == len(engineered)

    assert (
        engineered["unit_id"]
        .equals(original["unit_id"])
    )

    assert (
        engineered["cycle"]
        .equals(original["cycle"])
    )

    assert "RUL" in engineered.columns

    expected_feature_count = (
        len(SENSOR_COLUMNS) * 4
    )

    actual_feature_count = (
        len(engineered.columns)
        - len(original.columns)
    )

    assert (
        actual_feature_count
        == expected_feature_count
    )

    first_rows = (
        engineered
        .groupby("unit_id")
        .head(1)
    )

    for sensor in SENSOR_COLUMNS:

        assert (
            first_rows[
                f"{sensor}_delta_1"
            ]
            .isna()
            .all()
        )

        assert (
            first_rows[
                f"{sensor}_delta_5"
            ]
            .isna()
            .all()
        )


def main():
    """Run degradation trend feature engineering."""

    data = load_training_data()

    engineered = create_trend_features(
        data
    )

    validate_trend_features(
        data,
        engineered,
    )

    engineered.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    new_features = (
        len(engineered.columns)
        - len(data.columns)
    )

    print("\nDEGRADATION TREND FEATURES")
    print("=" * 70)

    print(
        f"Original columns : "
        f"{len(data.columns)}"
    )

    print(
        f"New features     : "
        f"{new_features}"
    )

    print(
        f"Final columns    : "
        f"{len(engineered.columns)}"
    )

    print(
        f"Rows             : "
        f"{len(engineered)}"
    )

    print(
        f"Engines          : "
        f"{engineered['unit_id'].nunique()}"
    )

    print("\nFeatures per sensor:")
    print("1-cycle delta")
    print("5-cycle delta")
    print("5-cycle slope")
    print("20-cycle slope")

    print("\nValidation: PASSED")

    print(f"\nSaved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()