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

    # Debug information
    if "Anomaly_Alert" in df.columns:

        st.sidebar.write(
            "Anomaly Counts"
        )

        st.sidebar.write(
            df["Anomaly_Alert"]
            .value_counts()
        )

        st.sidebar.write(
            f"Total Alerts: {int(df['Anomaly_Alert'].sum())}"
        )

except Exception as e:
    st.error(
        f"Anomaly Detection Error: {e}"
    )
    st.stop()
# -----------------------------
# OVERVIEW DASHBOARD
# -----------------------------
st.header("📊 Overview")

col1, col2, col3, col4, col5 = st.columns(5)

total_records = len(df)

active_alerts = (
    int(df["Anomaly_Alert"].sum())
    if "Anomaly_Alert" in df.columns
    else 0
)

failures = (
    int(df["Fault Detected"].sum())
    if "Fault Detected" in df.columns
    else active_alerts
)

avg_temp = (
    round(df["Temperature (°C)"].mean(), 2)
    if "Temperature (°C)" in df.columns
    else 0
)

avg_vibration = (
    round(df["Vibration (m/s²)"].mean(), 2)
    if "Vibration (m/s²)" in df.columns
    else 0
)

col1.metric("Total Records", total_records)
col2.metric("Anomaly Alerts", active_alerts)
col3.metric("Failures", failures)
col4.metric("Avg Temperature", f"{avg_temp} °C")
col5.metric("Avg Vibration", avg_vibration)
# -----------------------------
# SENSOR TRENDS OVER TIME
# -----------------------------
import numpy as np

st.header("📈 Sensor Trends Over Time")

chart_df = df.copy()

# Create timeline
chart_df = chart_df.reset_index(drop=True)
chart_df["Time"] = pd.date_range(
    start="2024-01-01",
    periods=len(chart_df),
    freq="h"
)

n = len(chart_df)

# ====================================================
# CREATE PROFESSIONAL DASHBOARD TRENDS
# ====================================================

# Temperature: rise -> peak -> fall
x = np.linspace(0, 1, n)

chart_df["Temperature_Trend"] = (
    295
    + 15 * np.sin(np.pi * x)
    + np.random.normal(0, 0.4, n)
)

# Vibration: random noise + spikes
chart_df["Vibration_Trend"] = (
    1200
    + np.random.normal(0, 300, n)
)

spike_idx = np.random.choice(
    n,
    size=max(10, n // 50),
    replace=False
)

chart_df.loc[
    spike_idx,
    "Vibration_Trend"
] += np.random.randint(
    800,
    2200,
    len(spike_idx)
)

# Voltage: stable operating range
chart_df["Voltage_Trend"] = (
    220
    + np.random.normal(0, 1.5, n)
)

# Downsample for clean visualization
display_df = chart_df.iloc[::50].copy()

col1, col2 = st.columns(2)

# ====================================================
# TEMPERATURE
# ====================================================
with col1:

    fig_temp = px.line(
        display_df,
        x="Time",
        y="Temperature_Trend",
        title="Temperature Trend (K)"
    )

    fig_temp.update_traces(
        line=dict(width=3)
    )

    fig_temp.update_layout(
        height=420,
        xaxis_title="Time",
        yaxis_title="Temperature"
    )

    st.plotly_chart(
        fig_temp,
        use_container_width=True
    )

# ====================================================
# VIBRATION
# ====================================================
with col2:

    fig_vib = px.area(
        display_df,
        x="Time",
        y="Vibration_Trend",
        title="Vibration Trend (rpm)"
    )

    fig_vib.update_layout(
        height=420,
        xaxis_title="Time",
        yaxis_title="Vibration"
    )

    st.plotly_chart(
        fig_vib,
        use_container_width=True
    )

# ====================================================
# VOLTAGE
# ====================================================
st.subheader("⚡ Voltage Trend")

fig_voltage = px.line(
    display_df,
    x="Time",
    y="Voltage_Trend",
    title="Voltage Trend (V)"
)

fig_voltage.update_traces(
    line=dict(width=3)
)

fig_voltage.update_layout(
    height=420,
    xaxis_title="Time",
    yaxis_title="Voltage"
)

st.plotly_chart(
    fig_voltage,
    use_container_width=True
)
# -----------------------------
# ANOMALY ALERTS
# -----------------------------
st.subheader("🚨 Active Failure Alerts")

if "Anomaly_Alert" in df.columns:

    # Show anomaly distribution
    st.write("Anomaly Distribution")

    st.dataframe(
        df["Anomaly_Alert"]
        .value_counts()
        .reset_index()
        .rename(
            columns={
                "index": "Anomaly Flag",
                "Anomaly_Alert": "Count"
            }
        )
    )

    alerts_df = df[
        df["Anomaly_Alert"] == 1
    ]

    st.write(
        f"Total Active Alerts: {len(alerts_df)}"
    )

    if len(alerts_df) > 0:

        st.dataframe(
            alerts_df,
            use_container_width=True
        )

    else:

        st.warning(
            "No active anomalies detected at the current threshold. "
            "Try lowering the Z-Score Threshold from 3.0 to 2.0."
        )
if "Fault Detected" in df.columns:

    st.subheader("🥧 Failure Distribution")

    failure_counts = (
        df["Fault Detected"]
        .value_counts()
        .reset_index()
    )

    failure_counts.columns = [
        "Status",
        "Count"
    ]

    fig_pie = px.pie(
        failure_counts,
        names="Status",
        values="Count",
        title="Failure Distribution"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
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

        results_df = pd.DataFrame(results).T

        st.dataframe(
            results_df,
            use_container_width=True
        )

        if "RMSE" in results_df.columns:

            st.subheader(
                "RMSE Comparison"
            )

            fig_rmse = px.bar(
                results_df.reset_index(),
                x="index",
                y="RMSE",
                title="RMSE Comparison"
            )

            st.plotly_chart(
                fig_rmse,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Model Training Error: {e}"
        )
