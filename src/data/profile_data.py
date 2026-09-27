from pathlib import Path

import pandas as pd

from src.data.load_data import load_fd001


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_PATH = REPORTS_DIR / "dataset_profile.csv"
ENGINE_LIFETIME_PATH = REPORTS_DIR / "engine_lifetime.csv"


def create_dataset_profile(
    train: pd.DataFrame,
    test: pd.DataFrame,
    rul: pd.DataFrame,
) -> pd.DataFrame:
    """Create a basic dataset-level profiling report."""

    rows = []

    datasets = {
        "train": train,
        "test": test,
        "rul": rul,
    }

    for name, df in datasets.items():
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": int(df.isnull().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
            }
        )

    profile = pd.DataFrame(rows)

    profile.to_csv(
        PROFILE_PATH,
        index=False,
    )

    return profile


def create_engine_lifetime_report(
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate lifetime statistics for every training engine."""

    lifetime = (
        train.groupby("unit_id")["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "lifetime_cycles"})
    )

    lifetime.to_csv(
        ENGINE_LIFETIME_PATH,
        index=False,
    )

    return lifetime


def print_summary(
    profile: pd.DataFrame,
    lifetime: pd.DataFrame,
) -> None:
    """Print the main profiling results."""

    print("\nDATASET PROFILE")
    print("=" * 60)
    print(profile.to_string(index=False))

    print("\nENGINE LIFETIME")
    print("=" * 60)

    print(
        lifetime["lifetime_cycles"]
        .describe()
        .to_string()
    )

    print("\nShortest engine lifetime:")
    print(lifetime["lifetime_cycles"].min())

    print("\nLongest engine lifetime:")
    print(lifetime["lifetime_cycles"].max())

    print("\nMedian engine lifetime:")
    print(lifetime["lifetime_cycles"].median())


def main():
    train, test, rul = load_fd001()

    profile = create_dataset_profile(
        train,
        test,
        rul,
    )

    lifetime = create_engine_lifetime_report(
        train,
    )

    print_summary(
        profile,
        lifetime,
    )

    print("\nReports created:")
    print(PROFILE_PATH)
    print(ENGINE_LIFETIME_PATH)


if __name__ == "__main__":
    main()