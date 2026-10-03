from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    robustness = pd.read_csv(
        REPORTS_DIR / "rul_robustness_by_lifecycle.csv"
    )

    thresholds = pd.read_csv(
        REPORTS_DIR / "failure_risk_thresholds.csv"
    )

    engine_risk = pd.read_csv(
        REPORTS_DIR / "engine_failure_risk.csv"
    )

    priority = pd.read_csv(
        REPORTS_DIR / "maintenance_priority.csv"
    )

    summary = {
        "lifecycle_ranges": len(robustness),
        "validation_engines": priority["unit_id"].nunique(),
        "validation_observations": engine_risk.shape[0],
        "overall_mae": engine_risk["absolute_error"].mean(),
        "overall_max_absolute_error": engine_risk["absolute_error"].max(),
        "critical_observations": (
            engine_risk["risk_level"] == "Critical"
        ).sum(),
        "warning_observations": (
            engine_risk["risk_level"] == "Warning"
        ).sum(),
        "normal_observations": (
            engine_risk["risk_level"] == "Normal"
        ).sum(),
        "critical_engines": (
            priority["priority_band"] == "Critical"
        ).sum(),
        "warning_engines": (
            priority["priority_band"] == "Warning"
        ).sum(),
        "normal_engines": (
            priority["priority_band"] == "Normal"
        ).sum(),
        "max_maintenance_priority_score": (
            priority["maintenance_priority_score"].max()
        ),
        "min_maintenance_priority_score": (
            priority["maintenance_priority_score"].min()
        ),
    }

    output = pd.DataFrame(
        list(summary.items()),
        columns=["metric", "value"]
    )

    output_path = REPORTS_DIR / "day7_risk_summary.csv"
    output.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Shape: {output.shape}")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()