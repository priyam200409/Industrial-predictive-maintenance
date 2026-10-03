from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "failure_risk_thresholds.csv"
)


RISK_THRESHOLDS = [
    {
        "risk_level": "Critical",
        "min_rul": 0,
        "max_rul": 20,
        "description": (
            "Very low remaining useful life; "
            "requires highest maintenance attention."
        ),
    },
    {
        "risk_level": "Warning",
        "min_rul": 21,
        "max_rul": 50,
        "description": (
            "Reduced remaining useful life; "
            "maintenance planning should be prioritized."
        ),
    },
    {
        "risk_level": "Normal",
        "min_rul": 51,
        "max_rul": None,
        "description": (
            "Remaining useful life above the project warning threshold."
        ),
    },
]


def build_threshold_table():
    """Build the project-defined failure-risk threshold table."""

    return pd.DataFrame(RISK_THRESHOLDS)


def validate_thresholds(df):
    """Validate threshold definitions."""

    required_columns = {
        "risk_level",
        "min_rul",
        "max_rul",
        "description",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df["risk_level"].duplicated().any():
        raise ValueError(
            "Duplicate risk levels found."
        )

    if (df["min_rul"] < 0).any():
        raise ValueError(
            "RUL thresholds cannot be negative."
        )

    # Validate bounded ranges.
    bounded = df[df["max_rul"].notna()]

    if (
        bounded["max_rul"]
        < bounded["min_rul"]
    ).any():
        raise ValueError(
            "Maximum RUL cannot be lower than minimum RUL."
        )

    expected_levels = {
        "Critical",
        "Warning",
        "Normal",
    }

    if set(df["risk_level"]) != expected_levels:
        raise ValueError(
            "Unexpected risk levels."
        )


def classify_risk(rul):
    """Classify a single RUL value using project thresholds."""

    if rul < 0:
        raise ValueError(
            f"RUL cannot be negative: {rul}"
        )

    if rul <= 20:
        return "Critical"

    if rul <= 50:
        return "Warning"

    return "Normal"


def validate_classification():

    test_cases = {
        0: "Critical",
        10: "Critical",
        20: "Critical",
        21: "Warning",
        35: "Warning",
        50: "Warning",
        51: "Normal",
        100: "Normal",
        250: "Normal",
    }

    for rul, expected in test_cases.items():

        actual = classify_risk(rul)

        if actual != expected:
            raise ValueError(
                f"RUL {rul}: expected "
                f"{expected}, got {actual}"
            )


def main():

    print("=" * 70)
    print("FAILURE-RISK THRESHOLD DEFINITION")
    print("=" * 70)

    print("\n[1/3] Building threshold table...")

    df = build_threshold_table()

    print(
        df.to_string(index=False)
    )

    print("\n[2/3] Validating thresholds...")

    validate_thresholds(df)
    validate_classification()

    print(
        "Threshold and classification validation passed."
    )

    print("\n[3/3] Saving report...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print(
        "\nFailure-risk threshold definition "
        "completed successfully."
    )


if __name__ == "__main__":
    main()