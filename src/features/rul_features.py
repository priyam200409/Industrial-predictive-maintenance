from pathlib import Path

import pandas as pd

from src.data.load_data import load_train_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "train_with_rul.csv"


def create_rul_target(train: pd.DataFrame) -> pd.DataFrame:
    """
    Create engine-specific Remaining Useful Life (RUL).

    RUL = maximum cycle of engine - current cycle.
    """

    data = train.copy()

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


def validate_rul(data: pd.DataFrame) -> None:
    """Validate the generated RUL target."""

    assert "RUL" in data.columns

    assert data["RUL"].notna().all()

    assert (data["RUL"] >= 0).all()

    assert (
        data.groupby("unit_id")["RUL"]
        .min()
        .eq(0)
        .all()
    )

    assert (
        data.groupby("unit_id")["RUL"]
        .max()
        .equals(
            data.groupby("unit_id")["cycle"]
            .max()
            .sub(
                data.groupby("unit_id")["cycle"]
                .min()
            )
        )
    )


def main():
    """Create and validate the training RUL dataset."""

    train = load_train_data()

    data = create_rul_target(train)

    validate_rul(data)

    data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nRUL FEATURE ENGINEERING")
    print("=" * 70)

    print(f"Rows       : {len(data)}")
    print(f"Engines    : {data['unit_id'].nunique()}")
    print(f"RUL min    : {data['RUL'].min()}")
    print(f"RUL max    : {data['RUL'].max()}")

    print("\nSample:")
    print(
        data[
            ["unit_id", "cycle", "RUL"]
        ].head(10).to_string(index=False)
    )

    print("\nValidation: PASSED")
    print(f"\nSaved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()