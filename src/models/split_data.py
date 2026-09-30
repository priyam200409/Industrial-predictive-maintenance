from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = PROCESSED_DIR / "train_features.csv"

SPLIT_METADATA_PATH = (
    PROJECT_ROOT
    / "reports"
    / "train_validation_split.csv"
)


# ============================================================
# SPLIT CONFIGURATION
# ============================================================

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

EXPECTED_ENGINES = 100
EXPECTED_ROWS = 20631


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_features() -> pd.DataFrame:
    """Load the final engineered feature dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    return data


# ============================================================
# ENGINE-LEVEL SPLIT
# ============================================================

def split_by_engine(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split observations by engine ID.

    Ensures that no engine appears in both
    training and validation datasets.
    """

    if "unit_id" not in data.columns:
        raise ValueError(
            "Required column 'unit_id' is missing."
        )

    engine_ids = (
        data["unit_id"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    train_engines, validation_engines = train_test_split(
        engine_ids,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_engines = set(train_engines)
    validation_engines = set(validation_engines)

    train = data[
        data["unit_id"].isin(train_engines)
    ].copy()

    validation = data[
        data["unit_id"].isin(validation_engines)
    ].copy()

    # Keep engine/cycle order deterministic.
    train = train.sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True)

    validation = validation.sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True)

    return train, validation


# ============================================================
# VALIDATE SPLIT
# ============================================================

def validate_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Validate that the engine-level split contains no leakage."""

    train_engines = set(
        train["unit_id"].unique()
    )

    validation_engines = set(
        validation["unit_id"].unique()
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    assert train_engines.isdisjoint(
        validation_engines
    ), (
        "DATA LEAKAGE DETECTED: "
        "an engine exists in both train and validation."
    )

    # --------------------------------------------------------
    # Engine count check
    # --------------------------------------------------------

    assert (
        len(train_engines)
        + len(validation_engines)
        == EXPECTED_ENGINES
    ), (
        f"Unexpected engine count: "
        f"{len(train_engines) + len(validation_engines)}"
    )

    # --------------------------------------------------------
    # Row count check
    # --------------------------------------------------------

    assert (
        len(train) + len(validation)
        == EXPECTED_ROWS
    ), (
        f"Unexpected total row count: "
        f"{len(train) + len(validation)}"
    )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    if "RUL" not in train.columns:
        raise ValueError(
            "Required target column 'RUL' is missing."
        )

    assert train["RUL"].notna().all(), (
        "Training data contains missing RUL values."
    )

    assert validation["RUL"].notna().all(), (
        "Validation data contains missing RUL values."
    )


# ============================================================
# SAVE SPLIT METADATA
# ============================================================

def save_split_metadata(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Save engine-level train/validation assignment."""

    train_engines = sorted(
        train["unit_id"].unique()
    )

    validation_engines = sorted(
        validation["unit_id"].unique()
    )

    rows = []

    for engine in train_engines:
        rows.append(
            {
                "unit_id": int(engine),
                "split": "train",
            }
        )

    for engine in validation_engines:
        rows.append(
            {
                "unit_id": int(engine),
                "split": "validation",
            }
        )

    metadata = pd.DataFrame(rows)

    metadata = metadata.sort_values(
        "unit_id"
    ).reset_index(drop=True)

    metadata.to_csv(
        SPLIT_METADATA_PATH,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """Create and validate deterministic engine-level split."""

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = load_features()

    # --------------------------------------------------------
    # Create engine-level split
    # --------------------------------------------------------

    train, validation = split_by_engine(
        data
    )

    # --------------------------------------------------------
    # Validate split
    # --------------------------------------------------------

    validate_split(
        train,
        validation,
    )

    # --------------------------------------------------------
    # Save only metadata
    #
    # We intentionally do NOT save train_split.csv and
    # validation_split.csv because the engineered dataset
    # is large and the deterministic engine assignments
    # are sufficient to recreate the split later.
    # --------------------------------------------------------

    save_split_metadata(
        train,
        validation,
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nENGINE-LEVEL TRAIN/VALIDATION SPLIT")
    print("=" * 70)

    print(
        f"Total engines      : "
        f"{data['unit_id'].nunique()}"
    )

    print(
        f"Training engines   : "
        f"{train['unit_id'].nunique()}"
    )

    print(
        f"Validation engines : "
        f"{validation['unit_id'].nunique()}"
    )

    print(
        f"\nTotal rows         : "
        f"{len(data)}"
    )

    print(
        f"Training rows      : "
        f"{len(train)}"
    )

    print(
        f"Validation rows    : "
        f"{len(validation)}"
    )

    print(
        f"\nTraining percentage: "
        f"{len(train) / len(data) * 100:.2f}%"
    )

    print(
        f"Validation percentage: "
        f"{len(validation) / len(data) * 100:.2f}%"
    )

    print("\nTraining engine IDs:")
    print(
        sorted(
            train["unit_id"].unique()
        )
    )

    print("\nValidation engine IDs:")
    print(
        sorted(
            validation["unit_id"].unique()
        )
    )

    print("\nLeakage check: PASSED")

    print(
        f"\nSaved split metadata:\n"
        f"{SPLIT_METADATA_PATH}"
    )

    print(
        "\nFull train/validation feature CSV files "
        "were not written."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()