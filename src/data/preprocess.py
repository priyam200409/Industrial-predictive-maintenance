import pandas as pd

from src.data.load_data import COLUMNS


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the C-MAPSS dataset schema."""
    if list(df.columns) != COLUMNS:
        raise ValueError(
            f"Invalid schema.\n"
            f"Expected: {COLUMNS}\n"
            f"Received: {list(df.columns)}"
        )

    if len(df.columns) != 26:
        raise ValueError(
            f"Expected 26 columns, found {len(df.columns)}"
        )


def validate_missing_values(df: pd.DataFrame) -> None:
    """Check for missing values."""
    missing = df.isnull().sum().sum()

    if missing > 0:
        raise ValueError(
            f"Dataset contains {missing} missing values."
        )


def validate_unit_ids(df: pd.DataFrame) -> None:
    """Validate engine/unit IDs."""
    if df["unit_id"].isnull().any():
        raise ValueError("unit_id contains missing values.")

    if (df["unit_id"] <= 0).any():
        raise ValueError("unit_id contains invalid values.")

    if not pd.api.types.is_integer_dtype(df["unit_id"]):
        raise ValueError("unit_id must contain integers.")


def validate_cycles(df: pd.DataFrame) -> None:
    """Validate engine cycle numbers."""
    if df["cycle"].isnull().any():
        raise ValueError("cycle contains missing values.")

    if (df["cycle"] <= 0).any():
        raise ValueError("cycle contains non-positive values.")

    # Check that cycle numbers are strictly increasing
    # for every engine.
    for unit_id, group in df.groupby("unit_id"):
        cycles = group["cycle"]

        if not cycles.is_monotonic_increasing:
            raise ValueError(
                f"Cycle numbers are not increasing for unit {unit_id}."
            )

        if cycles.duplicated().any():
            raise ValueError(
                f"Duplicate cycle detected for unit {unit_id}."
            )


def validate_duplicates(df: pd.DataFrame) -> None:
    """Check for completely duplicated rows."""
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        raise ValueError(
            f"Dataset contains {duplicates} duplicate rows."
        )


def validate_rul(rul: pd.DataFrame) -> None:
    """Validate RUL data."""
    if "RUL" not in rul.columns:
        raise ValueError("RUL column is missing.")

    if rul["RUL"].isnull().any():
        raise ValueError("RUL contains missing values.")

    if (rul["RUL"] < 0).any():
        raise ValueError("RUL contains negative values.")


def validate_test_rul_alignment(
    test: pd.DataFrame,
    rul: pd.DataFrame,
) -> None:
    """Check that RUL values match the number of test engines."""
    test_units = test["unit_id"].nunique()
    rul_rows = len(rul)

    if test_units != rul_rows:
        raise ValueError(
            f"Test/RUL mismatch: "
            f"{test_units} test engines vs {rul_rows} RUL values."
        )


def validate_fd001(
    train: pd.DataFrame,
    test: pd.DataFrame,
    rul: pd.DataFrame,
) -> None:
    """Run all FD001 validation checks."""

    print("Running FD001 validation...")
    print("-" * 50)

    print("1. Validating training schema...")
    validate_schema(train)
    print("   PASS")

    print("2. Validating testing schema...")
    validate_schema(test)
    print("   PASS")

    print("3. Checking training missing values...")
    validate_missing_values(train)
    print("   PASS")

    print("4. Checking testing missing values...")
    validate_missing_values(test)
    print("   PASS")

    print("5. Validating training unit IDs...")
    validate_unit_ids(train)
    print("   PASS")

    print("6. Validating testing unit IDs...")
    validate_unit_ids(test)
    print("   PASS")

    print("7. Validating training cycles...")
    validate_cycles(train)
    print("   PASS")

    print("8. Validating testing cycles...")
    validate_cycles(test)
    print("   PASS")

    print("9. Checking training duplicates...")
    validate_duplicates(train)
    print("   PASS")

    print("10. Checking testing duplicates...")
    validate_duplicates(test)
    print("   PASS")

    print("11. Validating RUL...")
    validate_rul(rul)
    print("   PASS")

    print("12. Checking test/RUL alignment...")
    validate_test_rul_alignment(test, rul)
    print("   PASS")

    print("-" * 50)
    print("ALL FD001 VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    from src.data.load_data import load_fd001

    train_df, test_df, rul_df = load_fd001()

    validate_fd001(
        train_df,
        test_df,
        rul_df,
    )