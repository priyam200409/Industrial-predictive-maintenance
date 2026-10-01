"""
Create a consolidated XGBoost explainability report.

Combines:
1. XGBoost built-in feature importance
2. Permutation importance

The report allows comparison of both importance methods
for the same 138 model features.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

XGB_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "xgboost_feature_importance.csv"
)

PERMUTATION_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "xgboost_permutation_importance.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "xgboost_explainability_report.csv"
)


# ============================================================
# LOAD REPORTS
# ============================================================

def load_reports():
    """Load both feature importance reports."""

    if not XGB_IMPORTANCE_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost importance report not found: "
            f"{XGB_IMPORTANCE_PATH}"
        )

    if not PERMUTATION_IMPORTANCE_PATH.exists():
        raise FileNotFoundError(
            f"Permutation importance report not found: "
            f"{PERMUTATION_IMPORTANCE_PATH}"
        )

    xgb_df = pd.read_csv(
        XGB_IMPORTANCE_PATH
    )

    permutation_df = pd.read_csv(
        PERMUTATION_IMPORTANCE_PATH
    )

    return xgb_df, permutation_df


# ============================================================
# BUILD CONSOLIDATED REPORT
# ============================================================

def build_report(
    xgb_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge both importance analyses."""

    xgb = xgb_df[
        [
            "feature",
            "rank",
            "importance",
        ]
    ].rename(
        columns={
            "rank": "xgb_rank",
            "importance": "xgb_importance",
        }
    )

    permutation = permutation_df[
        [
            "feature",
            "rank",
            "importance_mean",
            "importance_std",
        ]
    ].rename(
        columns={
            "rank": "permutation_rank",
        }
    )

    report = xgb.merge(
        permutation,
        on="feature",
        how="inner",
    )

    if len(report) != len(xgb_df):
        raise ValueError(
            "Feature mismatch detected between "
            "XGBoost and permutation importance reports."
        )

    # --------------------------------------------------------
    # Rank agreement
    # --------------------------------------------------------

    report["rank_difference"] = (
        report["xgb_rank"]
        - report["permutation_rank"]
    ).abs()

    # --------------------------------------------------------
    # Combined normalized score
    # --------------------------------------------------------

    xgb_max = report["xgb_importance"].max()

    permutation_max = report["importance_mean"].max()

    if xgb_max == 0 or permutation_max == 0:
        raise ValueError(
            "Cannot normalize importance because "
            "one importance scale has a maximum of zero."
        )

    report["xgb_importance_normalized"] = (
        report["xgb_importance"]
        / xgb_max
    )

    report["permutation_importance_normalized"] = (
        report["importance_mean"]
        / permutation_max
    )

    report["combined_importance"] = (
        report["xgb_importance_normalized"]
        + report["permutation_importance_normalized"]
    ) / 2

    # --------------------------------------------------------
    # Final ranking
    # --------------------------------------------------------

    report = report.sort_values(
        by="combined_importance",
        ascending=False,
    ).reset_index(drop=True)

    report.insert(
        0,
        "final_rank",
        range(1, len(report) + 1),
    )

    return report


# ============================================================
# MAIN
# ============================================================

def main():
    """Generate consolidated explainability report."""

    print("Loading XGBoost feature importance...")
    xgb_df, permutation_df = load_reports()

    print(
        f"XGBoost features      : {len(xgb_df)}"
    )

    print(
        f"Permutation features  : "
        f"{len(permutation_df)}"
    )

    print("\nCombining importance analyses...")

    report = build_report(
        xgb_df,
        permutation_df,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nTop 20 consolidated features:")
    print(
        report.head(20).to_string(
            index=False
        )
    )

    print(
        f"\nSaved report to:\n{OUTPUT_PATH}"
    )

    print(
        f"Total features analyzed: "
        f"{len(report)}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()