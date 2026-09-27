from pathlib import Path

import pandas as pd


# Project root:
# industrial-predictive-maintenance/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset directory
CMAPSS_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"

# FD001 files
TRAIN_PATH = CMAPSS_DIR / "train_FD001.txt"
TEST_PATH = CMAPSS_DIR / "test_FD001.txt"
RUL_PATH = CMAPSS_DIR / "RUL_FD001.txt"


# C-MAPSS FD001 column names
COLUMNS = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]


def _check_file_exists(file_path: Path) -> None:
    """Check that a required dataset file exists and is not empty."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {file_path}")


def load_train_data() -> pd.DataFrame:
    """Load the FD001 training dataset."""
    _check_file_exists(TRAIN_PATH)

    return pd.read_csv(
        TRAIN_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python",
    )


def load_test_data() -> pd.DataFrame:
    """Load the FD001 test dataset."""
    _check_file_exists(TEST_PATH)

    return pd.read_csv(
        TEST_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python",
    )


def load_rul_data() -> pd.DataFrame:
    """Load the FD001 test-set RUL values."""
    _check_file_exists(RUL_PATH)

    return pd.read_csv(
        RUL_PATH,
        sep=r"\s+",
        header=None,
        names=["RUL"],
        engine="python",
    )


def load_fd001():
    """Load train, test and RUL data for FD001."""
    train = load_train_data()
    test = load_test_data()
    rul = load_rul_data()

    return train, test, rul


if __name__ == "__main__":
    train, test, rul = load_fd001()

    print("C-MAPSS FD001 Data Loaded Successfully")
    print("-" * 50)

    print(f"Training shape : {train.shape}")
    print(f"Testing shape  : {test.shape}")
    print(f"RUL shape      : {rul.shape}")

    print(f"Training units : {train['unit_id'].nunique()}")
    print(f"Testing units  : {test['unit_id'].nunique()}")

    print("\nTraining columns:")
    print(train.columns.tolist())

    print("\nFirst 5 training rows:")
    print(train.head())

    print("\nFirst 5 RUL values:")
    print(rul.head())