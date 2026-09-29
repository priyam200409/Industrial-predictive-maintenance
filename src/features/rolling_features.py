from pathlib import Path

import pandas as pd

from src.data.load_data import load_train_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "train_with_rul.csv"
OUTPUT_PATH = PROCESSED_DIR / "train_rolling_features.csv"


SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
    if i not in [1, 5, 10, 16, 18, 19]
]

ROLLING_WINDOWS = [5, 10, 20]


def create_rolling_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create leakage-safe rolling statistics.

    Rolling statistics are calculated separately for each engine
    and use only previous observations.
    """

    result = data.copy()

    result = result.sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True)

    for sensor in SENSOR_COLUMNS:

        previous_values = (
            result
            .groupby("unit_id")[sensor]
            .shift(1)
        )

        for window in ROLLING_WINDOWS:

            rolling = (
                previous_values
                .groupby(result["unit_id"])
                .rolling(
                    window=window,
                    min_periods=1,
                )
            )

            result[
                f"{sensor}_rolling_mean_{window}"
            ] = (
                rolling
                .mean()
                .reset_index(
                    level=0,
                    drop=True,
                )
                .reset_index(drop=True)
            )

            result[
                f"{sensor}_rolling_std_{window}"
            ] = (
                rolling
                .std()
                .reset_index(
                    level=0,
                    drop=True,
                )
                .reset_index(drop=True)
            )

            result[
                f"{sensor}_rolling_min_{window}"
            ] = (
                rolling
                .min()
                .reset_index(
                    level=0,
                    drop=True,
                )
                .reset_index(drop=True)
            )

            result[
                f"{sensor}_rolling_max_{window}"
            ] = (
                rolling
                .max()
                .reset_index(
                    level=0,
                    drop=True,
                )
                .reset_index(drop=True)
            )

    return result


def validate_rolling_features(
    original: pd.DataFrame,
    engineered: pd.DataFrame,
) -> None:
    """Validate rolling feature generation."""

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
        len(SENSOR_COLUMNS)
        * len(ROLLING_WINDOWS)
        * 4
    )

    actual_feature_count = (
        len(engineered.columns)
        - len(original.columns)
    )

    assert (
        actual_feature_count
        == expected_feature_count
    )

    # First observation of every engine must not
    # contain information from another engine.
    first_rows = (
        engineered
        .groupby("unit_id")
        .head(1)
    )

    for sensor in SENSOR_COLUMNS:

        for window in ROLLING_WINDOWS:

            column = (
                f"{sensor}_rolling_mean_{window}"
            )

            assert column in engineered.columns

            # The first observation has no previous
            # sensor observation.
            assert (
                first_rows[column]
                .isna()
                .all()
            )


def main():
    """Run rolling feature engineering."""

    if INPUT_PATH.exists():

        data = pd.read_csv(INPUT_PATH)

    else:

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
            data["max_cycle"]
            - data["cycle"]
        )

        data.drop(
            columns=["max_cycle"],
            inplace=True,
        )

    engineered = create_rolling_features(data)

    validate_rolling_features(
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

    print("\nROLLING FEATURE ENGINEERING")
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

    print("\nWindows:")
    print(ROLLING_WINDOWS)

    print("\nValidation: PASSED")

    print(f"\nSaved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()