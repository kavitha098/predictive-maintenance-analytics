import streamlit as st
import pandas as pd

# -----------------------------
# SAFE IMPORTS
# -----------------------------
try:
    from utils.data_loader import (
        load_data,
        get_equipment_ids,
        filter_equipment
    )

    from maintenance_analytics.anomaly_detection import (
        detect_anomalies
    )

    from maintenance_analytics.feature_engineering import (
        add_rolling_features
    )

    from maintenance_analytics.root_cause_analysis import (
        equipment_failure_analysis,
        failure_type_analysis
    )

    from maintenance_models.train_models import (
        train_models
    )

except Exception as e:
    st.error(f"Import Error: {e}")
    st.stop()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance Analytics",
    layout="wide"
)

st.title("🔧 Predictive Maintenance Dashboard")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload Sensor CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = load_data(uploaded_file)

    except Exception as e:
        st.error(f"Dataset Loading Error: {e}")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Useful for debugging
    with st.expander("Dataset Columns"):
        st.write(df.columns.tolist())

    # -----------------------------
    # SIDEBAR
    # -----------------------------
    threshold = st.sidebar.slider(
        "Z-Score Threshold",
        min_value=2.0,
        max_value=4.0,
        value=3.0,
        step=0.1
    )

    equipment_ids = get_equipment_ids(df)

    if len(equipment_ids) > 0:

        selected_equipment = st.sidebar.selectbox(
            "Equipment ID",
            equipment_ids
        )

        df = filter_equipment(
            df,
            selected_equipment
        )

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    try:
        df = add_rolling_features(df)

    except Exception as e:
        st.error(
            f"Feature Engineering Error: {e}"
        )
        st.stop()

    # -----------------------------
    # ANOMALY DETECTION
    # -----------------------------
    try:
        df = detect_anomalies(
            df,
            threshold
        )

    except Exception as e:
        st.error(
            f"Anomaly Detection Error: {e}"
        )
        st.stop()

    # -----------------------------
    # METRICS
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        len(df)
    )

    if "Temperature (°C)" in df.columns:

        current_temp = round(
            float(
                df["Temperature (°C)"].iloc[-1]
            ),
            2
        )

    else:
        current_temp = "N/A"

    col2.metric(
        "Current Temperature",
        current_temp
    )

    alert_count = (
        int(df["Anomaly_Alert"].sum())
        if "Anomaly_Alert" in df.columns
        else 0
    )

    col3.metric(
        "Critical Alerts",
        alert_count
    )

    # -----------------------------
    # TEMPERATURE TREND
    # -----------------------------
    if "Temperature (°C)" in df.columns:

        st.subheader(
            "Temperature Trend"
        )

        st.line_chart(
            df["Temperature (°C)"]
        )

        failure_limit = (
            df["Temperature (°C)"].mean()
            +
            3 *
            df["Temperature (°C)"].std()
        )

        st.metric(
            "Failure Ceiling",
            round(failure_limit, 2)
        )

    # -----------------------------
    # ALERT TABLE
    # -----------------------------
    st.subheader(
        "Active Failure Alerts"
    )

    if "Anomaly_Alert" in df.columns:

        alerts_df = df[
            df["Anomaly_Alert"] == 1
        ]

        st.dataframe(alerts_df)

    # -----------------------------
    # ROOT CAUSE ANALYSIS
    # -----------------------------
    st.subheader(
        "Equipment Failure Analysis"
    )

    try:
        failure_analysis = (
            equipment_failure_analysis(df)
        )

        st.dataframe(
            failure_analysis
        )

    except Exception as e:
        st.warning(
            f"Failure Analysis Error: {e}"
        )

    # -----------------------------
    # FAILURE TYPE ANALYSIS
    # -----------------------------
    st.subheader(
        "Failure Type Analysis"
    )

    try:

        failure_type_df = (
            failure_type_analysis(df)
        )

        st.dataframe(
            failure_type_df
        )

    except Exception as e:
        st.warning(
            f"Failure Type Error: {e}"
        )

    # -----------------------------
    # MACHINE LEARNING
    # -----------------------------
    st.subheader(
        "Machine Learning Evaluation"
    )

    if st.button(
        "Train & Evaluate Models"
    ):

        try:

            results = train_models(df)

            results_df = (
                pd.DataFrame(results)
                .T
            )

            st.dataframe(
                results_df
            )

            if "RMSE" in results_df.columns:

                st.bar_chart(
                    results_df[
                        ["RMSE"]
                    ]
                )

        except Exception as e:

            st.error(
                f"Model Training Error: {e}"
            )

else:

    st.info(
        "Upload a CSV file to begin analysis."
    )
