import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from src.features.inference_features import build_inference_features


REPORTS = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "xgboost_rul_model.joblib"


st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    padding: 18px;
    border-radius: 12px;
}

.risk-critical {
    padding: 18px;
    border-radius: 12px;
    background: rgba(220, 38, 38, 0.15);
    border-left: 5px solid #dc2626;
}

.risk-warning {
    padding: 18px;
    border-radius: 12px;
    background: rgba(234, 179, 8, 0.15);
    border-left: 5px solid #eab308;
}

.risk-normal {
    padding: 18px;
    border-radius: 12px;
    background: rgba(34, 197, 94, 0.15);
    border-left: 5px solid #22c55e;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    predictions = pd.read_csv(
        REPORTS / "xgboost_validation_predictions.csv"
    )

    risk = pd.read_csv(
        REPORTS / "engine_failure_risk.csv"
    )

    priority = pd.read_csv(
        REPORTS / "maintenance_priority.csv"
    )

    metrics = pd.read_csv(
        REPORTS / "xgboost_rul_metrics.csv"
    )

    shap = pd.read_csv(
        REPORTS / "shap" / "global_shap_importance.csv"
    )

    return predictions, risk, priority, metrics, shap


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


predictions, risk, priority, metrics, shap = load_data()
model = load_model()

mae = metrics.loc[0, "MAE"]
rmse = metrics.loc[0, "RMSE"]

engines = risk["unit_id"].nunique()
critical = (risk["risk_level"] == "Critical").sum()
warning = (risk["risk_level"] == "Warning").sum()
normal = (risk["risk_level"] == "Normal").sum()


pages = {
    "🏠 Overview": "overview",
    "🔮 Predict Machine": "predict",
    "🔧 Engine Monitor": "monitor",
    "🚨 Maintenance Center": "maintenance",
    "🧠 Model Intelligence": "model",
}


st.sidebar.title("⚙️ Predictive Maintenance")
st.sidebar.caption("Machine Health & RUL Intelligence")

page = st.sidebar.radio(
    "Navigation",
    list(pages.keys())
)

st.sidebar.divider()
st.sidebar.caption("NASA C-MAPSS • FD001")
st.sidebar.caption("XGBoost RUL Model")


# ============================================================
# OVERVIEW
# ============================================================

if pages[page] == "overview":

    st.title("Industrial Predictive Maintenance")
    st.caption(
        "Predict machine degradation before unexpected downtime."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Engines Monitored", engines)
    c2.metric("Critical Observations", critical)
    c3.metric("Warning Observations", warning)
    c4.metric("Normal Observations", normal)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Maintenance Risk")

        risk_counts = pd.Series({
            "Normal": normal,
            "Warning": warning,
            "Critical": critical
        })

        st.bar_chart(risk_counts)

    with right:
        st.subheader("Model Performance")

        st.metric(
            "MAE",
            f"{mae:.2f} cycles"
        )

        st.metric(
            "RMSE",
            f"{rmse:.2f} cycles"
        )

        st.caption(
            "Validation results from the selected XGBoost model."
        )

    st.divider()

    st.subheader("Priority Maintenance Queue")

    queue = priority[
        [
            "unit_id",
            "latest_predicted_RUL",
            "critical_rate",
            "warning_rate",
            "maintenance_priority_score",
            "priority_band",
        ]
    ].sort_values(
        "maintenance_priority_score",
        ascending=False
    )

    st.dataframe(
        queue,
        width="stretch",
        hide_index=True
    )

    st.info(
        "The system helps maintenance teams identify degrading "
        "machines, estimate remaining useful life, and prioritize "
        "inspection."
    )


# ============================================================
# PREDICT MACHINE
# ============================================================

elif pages[page] == "predict":

    st.title("🔮 Predict Machine Health")

    st.write(
        "Analyze a machine's time-series sensor history and "
        "estimate its remaining useful life."
    )

    st.divider()

    st.subheader("Start an Analysis")

    option = st.radio(
        "Choose data source",
        [
            "Try Demo Machine",
            "Upload Sensor CSV"
        ],
        horizontal=True,
    )

    # --------------------------------------------------------
    # DEMO
    # --------------------------------------------------------

    if option == "Try Demo Machine":

        st.info(
            "Demo mode uses an existing validated C-MAPSS engine "
            "from the project dataset."
        )

        engine_ids = sorted(
            priority["unit_id"].unique()
        )

        engine = st.selectbox(
            "Select demo engine",
            engine_ids
        )

        if st.button(
            "Analyze Machine",
            type="primary",
            width="stretch"
        ):

            row = priority[
                priority["unit_id"] == engine
            ].iloc[0]

            rul = row["latest_predicted_RUL"]
            score = row["maintenance_priority_score"]

            if row["critical_rate"] > 0.10:
                status = "Critical"
            elif row["warning_rate"] > 0.10:
                status = "Warning"
            else:
                status = "Normal"

            st.success("Analysis complete.")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Predicted RUL",
                f"{rul:.1f} cycles"
            )

            c2.metric(
                "Risk",
                status
            )

            c3.metric(
                "Maintenance Score",
                f"{score:.1f}"
            )

            if status == "Critical":

                st.markdown(
                    '<div class="risk-critical">'
                    '<b>🔴 CRITICAL</b><br>'
                    'Prioritize this engine for maintenance inspection.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            elif status == "Warning":

                st.markdown(
                    '<div class="risk-warning">'
                    '<b>🟡 WARNING</b><br>'
                    'Increase monitoring and plan maintenance.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    '<div class="risk-normal">'
                    '<b>🟢 NORMAL</b><br>'
                    'No immediate maintenance priority detected.'
                    '</div>',
                    unsafe_allow_html=True,
                )

    # --------------------------------------------------------
    # LIVE CSV PREDICTION
    # --------------------------------------------------------

    else:

        st.subheader("Upload Machine Sensor Data")

        st.caption(
            "Compatible with the NASA C-MAPSS FD001 sensor format."
        )

        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"]
        )

        if uploaded:

            data = pd.read_csv(uploaded)

            required = [
                "unit_id",
                "cycle",
                "op_setting_1",
                "op_setting_2",
                "op_setting_3",
            ] + [
                f"sensor_{i}"
                for i in range(1, 22)
            ]

            missing = [
                column
                for column in required
                if column not in data.columns
            ]

            if missing:

                st.error(
                    "Invalid CSV. Missing columns: "
                    + ", ".join(missing)
                )

            else:

                st.success(
                    f"Valid sensor data: {len(data):,} rows"
                )

                st.dataframe(
                    data.head(),
                    width="stretch",
                    hide_index=True
                )

                if st.button(
                    "Run Prediction",
                    type="primary",
                    width="stretch"
                ):

                    with st.spinner(
                        "Generating features and running XGBoost..."
                    ):

                        features = build_inference_features(
                            data
                        )

                        latest = (
                            features
                            .groupby("unit_id")
                            .tail(1)
                            .copy()
                        )

                        X = latest.drop(
                            columns=[
                                "unit_id",
                                "cycle"
                            ]
                        )

                        predicted_rul = model.predict(X)

                        results = latest[
                            [
                                "unit_id",
                                "cycle"
                            ]
                        ].copy()

                        results["predicted_RUL"] = predicted_rul

                    st.success(
                        "Prediction completed successfully."
                    )

                    st.divider()

                    st.subheader("Prediction Result")

                    for _, row in results.iterrows():

                        rul = row["predicted_RUL"]

                        if rul <= 20:
                            status = "Critical"
                        elif rul <= 50:
                            status = "Warning"
                        else:
                            status = "Normal"

                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "Machine",
                            int(row["unit_id"])
                        )

                        c2.metric(
                            "Current Cycle",
                            int(row["cycle"])
                        )

                        c3.metric(
                            "Predicted RUL",
                            f"{rul:.2f} cycles"
                        )

                        if status == "Critical":

                            st.markdown(
                                '<div class="risk-critical">'
                                '<b>🔴 CRITICAL</b><br>'
                                'Prioritize this machine for maintenance inspection.'
                                '</div>',
                                unsafe_allow_html=True,
                            )

                        elif status == "Warning":

                            st.markdown(
                                '<div class="risk-warning">'
                                '<b>🟡 WARNING</b><br>'
                                'Increase monitoring and plan maintenance.'
                                '</div>',
                                unsafe_allow_html=True,
                            )

                        else:

                            st.markdown(
                                '<div class="risk-normal">'
                                '<b>🟢 NORMAL</b><br>'
                                'No immediate maintenance priority detected.'
                                '</div>',
                                unsafe_allow_html=True,
                            )

                    st.divider()

                    st.subheader("Live Predictions")

                    st.dataframe(
                        results,
                        width="stretch",
                        hide_index=True
                    )


# ============================================================
# ENGINE MONITOR
# ============================================================

elif pages[page] == "monitor":

    st.title("🔧 Engine Monitor")

    engine_ids = sorted(
        priority["unit_id"].unique()
    )

    engine = st.selectbox(
        "Select Engine",
        engine_ids
    )

    engine_data = priority[
        priority["unit_id"] == engine
    ].iloc[0]

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current Cycle",
        int(engine_data["latest_cycle"])
    )

    c2.metric(
        "Predicted RUL",
        f"{engine_data['latest_predicted_RUL']:.2f}"
    )

    c3.metric(
        "Critical Rate",
        f"{engine_data['critical_rate']:.1%}"
    )

    c4.metric(
        "Maintenance Score",
        f"{engine_data['maintenance_priority_score']:.2f}"
    )

    st.divider()

    engine_predictions = predictions[
        predictions["unit_id"] == engine
    ]

    if not engine_predictions.empty:

        st.subheader("RUL Prediction Trend")

        trend_cols = [
            col
            for col in [
                "cycle",
                "actual_RUL",
                "predicted_RUL"
            ]
            if col in engine_predictions.columns
        ]

        if len(trend_cols) > 1:

            st.line_chart(
                engine_predictions[
                    trend_cols
                ].set_index("cycle")
            )

    st.subheader("Engine Diagnostics")

    st.dataframe(
        engine_data.to_frame().T,
        width="stretch",
        hide_index=True
    )


# ============================================================
# MAINTENANCE CENTER
# ============================================================

elif pages[page] == "maintenance":

    st.title("🚨 Maintenance Center")

    st.write(
        "Prioritized maintenance queue based on predicted "
        "RUL, observed risk, and maintenance priority."
    )

    risk_filter = st.selectbox(
        "Priority Band",
        [
            "All",
            "Critical",
            "Warning",
            "Normal"
        ]
    )

    queue = priority.copy()

    if risk_filter != "All":

        queue = queue[
            queue["priority_band"] == risk_filter
        ]

    queue = queue.sort_values(
        "maintenance_priority_score",
        ascending=False
    )

    st.dataframe(
        queue,
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.subheader("How to use this queue")

    st.write(
        "Inspect higher-priority machines first. The score is "
        "a project-defined decision-support measure and is not "
        "an autonomous maintenance command."
    )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

else:

    st.title("🧠 Model Intelligence")

    st.caption(
        "Technical view of the predictive-maintenance model."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Model",
        "XGBoost"
    )

    c2.metric(
        "MAE",
        f"{mae:.2f}"
    )

    c3.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    st.divider()

    st.subheader("Global SHAP Importance")

    top_shap = shap.head(15)

    if "feature" in top_shap.columns:

        chart = (
            top_shap
            .set_index("feature")["mean_abs_shap"]
            .sort_values()
        )

        st.bar_chart(chart)

    st.divider()

    st.subheader("What the model is doing")

    st.write(
        """
        The model estimates Remaining Useful Life (RUL) from
        time-series machine sensor behavior.

        The selected XGBoost model achieved a validation MAE of
        22.89 cycles and RMSE of 32.24 cycles.

        SHAP is used to explain which engineered sensor features
        contribute most strongly to predictions.
        """
    )

    st.info(
        "Important: this system is a maintenance decision-support "
        "tool. It does not diagnose the exact physical cause of a "
        "machine fault."
    )