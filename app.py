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
st.header("📈 Sensor Trends Over Time")

chart_df = df.copy()

# Convert index back to column
if "Timestamp" not in chart_df.columns:
    chart_df = chart_df.reset_index()

chart_df["Timestamp"] = pd.to_datetime(
    chart_df["Timestamp"],
    errors="coerce"
)

chart_df = chart_df.sort_values("Timestamp")

# Daily aggregation
chart_df = (
    chart_df
    .set_index("Timestamp")
    .resample("1D")
    .mean(numeric_only=True)
    .reset_index()
)

col1, col2 = st.columns(2)

# -----------------------------
# TEMPERATURE TREND
# -----------------------------
with col1:

    if "Temperature (°C)" in chart_df.columns:

        temp_df = chart_df.copy()

        # Large rolling window to create smooth rise-then-fall trend
        temp_df["Temp_Trend"] = (
            temp_df["Temperature (°C)"]
            .rolling(window=7, center=True, min_periods=1)
            .mean()
        )

        fig_temp = px.line(
            temp_df,
            x="Timestamp",
            y="Temp_Trend",
            title="Temperature (°C)"
        )

        fig_temp.update_traces(
            line=dict(width=3)
        )

        fig_temp.update_layout(
            height=400,
            xaxis_title="Timestamp",
            yaxis_title="Temperature (°C)"
        )

        st.plotly_chart(
            fig_temp,
            use_container_width=True
        )

# -----------------------------
# VIBRATION TREND
# -----------------------------
with col2:

    if "Vibration (m/s²)" in df.columns:

        vib_df = df.copy()

        if "Timestamp" not in vib_df.columns:
            vib_df = vib_df.reset_index()

        vib_df["Timestamp"] = pd.to_datetime(
            vib_df["Timestamp"],
            errors="coerce"
        )

        vib_df = vib_df.sort_values("Timestamp")

        # Keep high-frequency vibration data
        vib_df = vib_df.iloc[::5]

        fig_vib = px.area(
            vib_df,
            x="Timestamp",
            y="Vibration (m/s²)",
            title="Vibration (m/s²)"
        )

        fig_vib.update_layout(
            height=400,
            xaxis_title="Timestamp",
            yaxis_title="Vibration"
        )

        st.plotly_chart(
            fig_vib,
            use_container_width=True
        )

# -----------------------------
# VOLTAGE TREND
# -----------------------------
if "Voltage (V)" in chart_df.columns:

    st.subheader("Voltage (V)")

    voltage_df = chart_df.copy()

    voltage_df["Voltage_Trend"] = (
        voltage_df["Voltage (V)"]
        .rolling(window=5, center=True, min_periods=1)
        .mean()
    )

    fig_voltage = px.line(
        voltage_df,
        x="Timestamp",
        y="Voltage_Trend",
        title="Voltage (V)"
    )

    fig_voltage.update_traces(
        line=dict(width=2)
    )

    fig_voltage.update_layout(
        height=400,
        xaxis_title="Timestamp",
        yaxis_title="Voltage (V)"
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
