from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RISK_PATH = (
    PROJECT_ROOT
    / "reports"
    / "engine_failure_risk.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "maintenance_priority.csv"
)


def load_risk_data():
    """Load observation-level failure-risk indicators."""

    if not RISK_PATH.exists():
        raise FileNotFoundError(
            f"Risk report not found: {RISK_PATH}"
        )

    df = pd.read_csv(RISK_PATH)

    required_columns = {
        "unit_id",
        "cycle",
        "predicted_RUL",
        "absolute_error",
        "risk_level",
        "risk_score",
    }

    missing = required_columns - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def calculate_priority(df):
    """Aggregate risk indicators at engine level."""

    rows = []

    for unit_id, group in df.groupby(
        "unit_id",
        sort=True,
    ):

        total_observations = len(group)

        critical_count = (
            group["risk_level"] == "Critical"
        ).sum()

        warning_count = (
            group["risk_level"] == "Warning"
        ).sum()

        critical_rate = (
            critical_count
            / total_observations
        )

        warning_rate = (
            warning_count
            / total_observations
        )

        priority_score = (
            critical_rate * 70
            + warning_rate * 30
        )

        latest_row = group.loc[
            group["cycle"].idxmax()
        ]

        rows.append(
            {
                "unit_id": int(unit_id),
                "observations": total_observations,
                "latest_cycle": int(
                    latest_row["cycle"]
                ),
                "minimum_predicted_RUL": group[
                    "predicted_RUL"
                ].min(),
                "latest_predicted_RUL": latest_row[
                    "predicted_RUL"
                ],
                "critical_observations": int(
                    critical_count
                ),
                "warning_observations": int(
                    warning_count
                ),
                "critical_rate": critical_rate,
                "warning_rate": warning_rate,
                "mean_absolute_error": group[
                    "absolute_error"
                ].mean(),
                "maximum_absolute_error": group[
                    "absolute_error"
                ].max(),
                "maintenance_priority_score": (
                    priority_score
                ),
            }
        )

    return pd.DataFrame(rows)


def assign_priority_band(score):
    """Assign a project-defined priority band."""

    if score >= 70:
        return "Critical"

    if score >= 30:
        return "Warning"

    return "Normal"


def validate_result(result):
    """Validate maintenance priority report."""

    if result.empty:
        raise ValueError(
            "Maintenance priority report is empty."
        )

    required_columns = {
        "unit_id",
        "observations",
        "latest_cycle",
        "minimum_predicted_RUL",
        "latest_predicted_RUL",
        "critical_observations",
        "warning_observations",
        "critical_rate",
        "warning_rate",
        "mean_absolute_error",
        "maximum_absolute_error",
        "maintenance_priority_score",
        "priority_band",
    }

    missing = required_columns - set(
        result.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if result.isna().any().any():
        raise ValueError(
            "Maintenance priority report contains NaN values."
        )

    if result["unit_id"].nunique() != len(result):
        raise ValueError(
            "Duplicate engine IDs detected."
        )

    if not (
        result["maintenance_priority_score"]
        .between(0, 100)
        .all()
    ):
        raise ValueError(
            "Priority scores must be between 0 and 100."
        )

    if not (
        result["critical_rate"]
        .between(0, 1)
        .all()
    ):
        raise ValueError(
            "Invalid critical rates."
        )

    if not (
        result["warning_rate"]
        .between(0, 1)
        .all()
    ):
        raise ValueError(
            "Invalid warning rates."
        )


def main():

    print("=" * 70)
    print("ENGINE MAINTENANCE PRIORITY ANALYSIS")
    print("=" * 70)

    print("\n[1/4] Loading risk indicators...")

    df = load_risk_data()

    print(
        f"Observations: {len(df)}"
    )

    print(
        f"Engines: {df['unit_id'].nunique()}"
    )

    print("\n[2/4] Calculating engine-level priority...")

    result = calculate_priority(df)

    result["priority_band"] = (
        result["maintenance_priority_score"]
        .apply(assign_priority_band)
    )

    print(
        f"Engine reports: {len(result)}"
    )

    print("\n[3/4] Validating results...")

    validate_result(result)

    print(
        "Validation passed."
    )

    print("\n[4/4] Saving report...")

    result = result.sort_values(
        "maintenance_priority_score",
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print("\nPriority-band distribution:")

    print(
        result["priority_band"]
        .value_counts()
        .to_string()
    )

    print("\nTop maintenance-priority engines:")

    print(
        result[
            [
                "unit_id",
                "maintenance_priority_score",
                "priority_band",
                "minimum_predicted_RUL",
                "latest_predicted_RUL",
                "critical_rate",
                "warning_rate",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nMaintenance priority analysis "
        "completed successfully."
    )


if __name__ == "__main__":
    main()