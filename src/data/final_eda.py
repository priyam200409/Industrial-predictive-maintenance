from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.load_data import load_fd001


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.csv"


def create_training_rul(train: pd.DataFrame) -> pd.DataFrame:
    """Create RUL values for training observations."""

    train = train.copy()

    max_cycles = (
        train.groupby("unit_id")["cycle"]
        .max()
        .rename("max_cycle")
    )

    train = train.merge(
        max_cycles,
        on="unit_id",
        how="left",
    )

    train["RUL"] = (
        train["max_cycle"] - train["cycle"]
    )

    train.drop(
        columns=["max_cycle"],
        inplace=True,
    )

    return train


def create_lifetime_plot(train: pd.DataFrame) -> None:
    """Create engine lifetime distribution."""

    lifetime = (
        train.groupby("unit_id")["cycle"]
        .max()
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        lifetime,
        bins=15,
    )

    plt.xlabel("Engine Lifetime (Cycles)")
    plt.ylabel("Number of Engines")
    plt.title("Training Engine Lifetime Distribution")

    plt.tight_layout()

    output = FIGURES_DIR / "engine_lifetime_distribution.png"

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()


def create_rul_plot(train: pd.DataFrame) -> None:
    """Create training RUL distribution."""

    plt.figure(figsize=(10, 6))

    plt.hist(
        train["RUL"],
        bins=30,
    )

    plt.xlabel("RUL (Cycles)")
    plt.ylabel("Number of Observations")
    plt.title("Training RUL Distribution")

    plt.tight_layout()

    output = FIGURES_DIR / "rul_distribution.png"

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()


def create_sensor_trend_plot(
    train: pd.DataFrame,
) -> None:
    """Plot selected degradation-sensitive sensors."""

    selected_sensors = [
        "sensor_11",
        "sensor_4",
        "sensor_12",
        "sensor_7",
    ]

    for sensor in selected_sensors:

        plt.figure(figsize=(10, 6))

        # Plot a few engines to avoid an unreadable figure.
        for unit_id in [1, 2, 3, 4, 5]:

            engine = train[
                train["unit_id"] == unit_id
            ]

            plt.plot(
                engine["cycle"],
                engine[sensor],
                label=f"Engine {unit_id}",
            )

        plt.xlabel("Cycle")
        plt.ylabel(sensor)
        plt.title(
            f"{sensor} Behavior Across Engine Life"
        )

        plt.legend()

        plt.tight_layout()

        output = (
            FIGURES_DIR
            / f"{sensor}_degradation_trend.png"
        )

        plt.savefig(
            output,
            dpi=150,
        )

        plt.close()


def create_eda_summary(train: pd.DataFrame) -> pd.DataFrame:
    """Create consolidated EDA summary."""

    sensors = [
        f"sensor_{i}"
        for i in range(1, 22)
    ]

    records = []

    for sensor in sensors:

        variance = train[sensor].var()

        cycle_corr = train[sensor].corr(
            train["cycle"],
            method="spearman",
        )

        rul_corr = train[sensor].corr(
            train["RUL"],
            method="spearman",
        )

        records.append(
            {
                "sensor": sensor,
                "variance": variance,
                "spearman_cycle_corr": cycle_corr,
                "spearman_rul_corr": rul_corr,
                "absolute_cycle_corr": (
                    abs(cycle_corr)
                    if pd.notna(cycle_corr)
                    else None
                ),
                "absolute_rul_corr": (
                    abs(rul_corr)
                    if pd.notna(rul_corr)
                    else None
                ),
            }
        )

    summary = pd.DataFrame(records)

    summary = summary.sort_values(
        "absolute_rul_corr",
        ascending=False,
        na_position="last",
    )

    summary.to_csv(
        EDA_SUMMARY_PATH,
        index=False,
    )

    return summary


def print_summary(
    train: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Print final EDA conclusions."""

    lifetime = (
        train.groupby("unit_id")["cycle"]
        .max()
    )

    print("\nFINAL EDA SUMMARY")
    print("=" * 70)

    print("\nDataset:")
    print(f"Training observations : {len(train)}")
    print(
        f"Training engines      : "
        f"{train['unit_id'].nunique()}"
    )

    print("\nEngine lifetime:")
    print(f"Minimum : {lifetime.min()}")
    print(f"Median  : {lifetime.median()}")
    print(f"Mean    : {lifetime.mean():.2f}")
    print(f"Maximum : {lifetime.max()}")

    print("\nTop sensors by absolute RUL relationship:")

    print(
        summary[
            [
                "sensor",
                "spearman_cycle_corr",
                "spearman_rul_corr",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    constant_sensors = [
        sensor
        for sensor in summary["sensor"]
        if train[sensor].nunique() <= 1
    ]

    print("\nConstant sensors:")
    print(constant_sensors)

    print("\nReports created:")
    print(EDA_SUMMARY_PATH)
    print(FIGURES_DIR)


def main():
    """Run final Day 2 EDA."""

    train, _, _ = load_fd001()

    train = create_training_rul(train)

    create_lifetime_plot(train)

    create_rul_plot(train)

    create_sensor_trend_plot(train)

    summary = create_eda_summary(train)

    print_summary(
        train,
        summary,
    )


if __name__ == "__main__":
    main()
    