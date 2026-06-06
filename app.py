import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance Analytics",
    layout="wide"
)

st.title("🔧 Predictive Maintenance Analytics Dashboard")

# -----------------------------
# IMPORTS
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
# FILE UPLOADER
# -----------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload Sensor CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin analysis.")
    st.stop()

# -----------------------------
# LOAD DATA
# -----------------------------
try:
    df = load_data(uploaded_file)

except Exception as e:
    st.error(f"Data Loading Error: {e}")
    st.stop()

# -----------------------------
# DATA PREVIEW
# -----------------------------
with st.expander("Dataset Preview"):
    st.dataframe(df.head())

with st.expander("Dataset Columns"):
    st.write(df.columns.tolist())

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
threshold = st.sidebar.slider(
    "Z-Score Threshold",
    min_value=2.0,
    max_value=4.0,
    value=3.0,
    step=0.1
)

equipment_ids = get_equipment_ids(df)

equipment_options = ["All Equipment"]

if len(equipment_ids) > 0:
    equipment_options.extend(list(equipment_ids))

selected_equipment = st.sidebar.selectbox(
    "Equipment ID",
    equipment_options
)

if selected_equipment != "All Equipment":
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

if "Anomaly_Alert" in df.columns:
    active_alerts = int(
        df["Anomaly_Alert"].sum()
    )
else:
    active_alerts = 0

col3.metric(
    "Active Alerts",
    active_alerts
)

# -----------------------------
# TEMPERATURE TREND
# -----------------------------
if (
    "Temperature (°C)" in df.columns
    and len(df) > 0
):

    st.subheader("🌡 Temperature Trend")

    chart_df = df.reset_index()

    failure_limit = (
        chart_df["Temperature (°C)"].mean()
        +
        3 *
        chart_df["Temperature (°C)"].std()
    )

    fig = px.line(
        chart_df,
        x="Timestamp",
        y="Temperature (°C)",
        title="Temperature Over Time"
    )

    fig.add_hline(
        y=failure_limit,
        line_dash="dash",
        annotation_text="Failure Ceiling"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.metric(
        "Failure Ceiling",
        round(failure_limit, 2)
    )

# -----------------------------
# VIBRATION TREND
# -----------------------------
if (
    "Vibration (m/s²)" in df.columns
    and len(df) > 0
):

    st.subheader("📈 Vibration Trend")

    chart_df = df.reset_index()

    fig = px.line(
        chart_df,
        x="Timestamp",
        y="Vibration (m/s²)",
        title="Vibration Over Time"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -----------------------------
# ANOMALY ALERTS
# -----------------------------
st.subheader(
    "🚨 Active Failure Alerts"
)

if "Anomaly_Alert" in df.columns:

    alerts_df = df[
        df["Anomaly_Alert"] == 1
    ]

    st.dataframe(
        alerts_df
    )

# -----------------------------
# ROOT CAUSE ANALYSIS
# -----------------------------
st.subheader(
    "🔍 Equipment Failure Analysis"
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
        f"Equipment Analysis Error: {e}"
    )

# -----------------------------
# FAILURE TYPE ANALYSIS
# -----------------------------
st.subheader(
    "⚙ Failure Type Analysis"
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
    "🤖 Machine Learning Evaluation"
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

            st.subheader(
                "RMSE Comparison"
            )

            st.bar_chart(
                results_df[
                    ["RMSE"]
                ]
            )

    except Exception as e:

        st.error(
            f"Model Training Error: {e}"
        )
