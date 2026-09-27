from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
METRICS_DIR = REPORT_DIR / "metrics"


# ============================================================
# C-MAPSS DATASET
# ============================================================

CMAPSS_DIR = RAW_DATA_DIR / "CMAPSSData"

TRAIN_FD001_PATH = CMAPSS_DIR / "train_FD001.txt"
TEST_FD001_PATH = CMAPSS_DIR / "test_FD001.txt"
RUL_FD001_PATH = CMAPSS_DIR / "RUL_FD001.txt"


# ============================================================
# DATASET SCHEMA
# ============================================================

CMAPSS_COLUMNS = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


EXPECTED_COLUMN_COUNT = len(CMAPSS_COLUMNS)


# ============================================================
# PROJECT SETTINGS
# ============================================================

RANDOM_STATE = 42