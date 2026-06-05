import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import (
    load_data,
    get_equipment_ids,
    filter_equipment,
    get_failure_rows
)

from utils.preprocessing import (
    preprocess_pipeline
)

from utils.helper import (
    calculate_operating_hours,
    get_current_temperature,
    get_anomaly_count
)

from analytics.anomaly_detection import (
    detect_anomalies,
    get_anomaly_rows
)

from analytics.root_cause_analysis import (
    failure_summary,
    equipment_failure_analysis
)

from analytics.visualization import (
    plot_temperature_trend,
    plot_vibration_trend,
    plot_failure_heatmap,
    plot_failure_distribution,
    generate_wordcloud
)

from ml.train_model import (
    train_all_models
)

st.set_page_config(
    page_title="Predictive Maintenance Analytics",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Predictive Maintenance Analytics Dashboard")

st.markdown("---")

# =========================
# Sidebar
# =========================

st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Sensor CSV",
    type=["csv"]
)

z_threshold = st.sidebar.slider(
    "Z-Score Threshold",
    min_value=2.0,
    max_value=4.0,
    value=3.0,
    step=0.1
)

# =========================
# Load Data
# =========================

df = load_data(uploaded_file)

if df.empty:
    st.warning("Upload dataset to continue.")
    st.stop()

# =========================
# Preprocessing
# =========================

df = preprocess_pipeline(df)

# =========================
# Equipment Filter
# =========================

equipment_list = get_equipment_ids(df)

selected_equipment = st.sidebar.selectbox(
    "Select Equipment",
    equipment_list
)

df = filter_equipment(
    df,
    selected_equipment
)

# =========================
# Anomaly Detection
# =========================

df = detect_anomalies(
    df,
    threshold=z_threshold
)

# =========================
# Dashboard Metrics
# =========================

operating_hours = calculate_operating_hours(df)

current_temp = get_current_temperature(df)

critical_alerts = get_anomaly_count(df)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Operating Hours",
        operating_hours
    )

with c2:
    st.metric(
        "Current Temperature",
        f"{current_temp} °C"
    )

with c3:
    st.metric(
        "Critical Alerts",
        critical_alerts
    )

st.markdown("---")

# =========================
# Sensor Trend
# =========================

st.subheader("Temperature Trend")

plot_temperature_trend(df)

st.subheader("Vibration Trend")

plot_vibration_trend(df)

# =========================
# Failure Ceiling Line Chart
# =========================

st.subheader("Sensor Drift Analysis")

if "Temperature (°C)" in df.columns:

    failure_limit = (
        df["Temperature (°C)"].mean()
        +
        (z_threshold *
         df["Temperature (°C)"].std())
    )

    chart_df = pd.DataFrame({
        "Temperature":
        df["Temperature (°C)"],

        "Failure Ceiling":
        failure_limit
    })

    st.line_chart(chart_df)

# =========================
# Active Failures
# =========================

st.subheader("Failure Records")

failure_rows = get_failure_rows(df)

st.dataframe(
    failure_rows,
    use_container_width=True
)

# =========================
# Anomaly Records
# =========================

st.subheader("Anomaly Alerts")

anomaly_rows = get_anomaly_rows(df)

st.dataframe(
    anomaly_rows,
    use_container_width=True
)

# =========================
# Root Cause Analysis
# =========================

st.subheader("Root Cause Analysis")

summary = failure_summary(df)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Total Failures",
        summary["Total Failures"]
    )

with col2:
    st.metric(
        "Failure Rate %",
        summary["Failure Rate (%)"]
    )

st.markdown("### Equipment Failure Ranking")

equipment_failures = (
    equipment_failure_analysis(df)
)

if not equipment_failures.empty:

    fig = px.bar(
        equipment_failures,
        x="Equipment_ID",
        y="Failure_Count",
        title="Failures by Equipment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================
# Correlation Heatmap
# =========================

st.subheader("Correlation Analysis")

plot_failure_heatmap(df)

# =========================
# Failure Distribution
# =========================

st.subheader("Failure Distribution")

plot_failure_distribution(df)

# =========================
# Word Cloud
# =========================

st.subheader("Equipment Word Cloud")

generate_wordcloud(df)

# =========================
# Raw Data
# =========================

with st.expander("View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )

# =========================
# Machine Learning Section
# =========================

st.markdown("---")

st.header("Machine Learning Models")

if st.button("Train Models"):

    with st.spinner(
        "Training models..."
    ):

        results = train_all_models(df)

    st.success(
        "Training Completed"
    )

    st.dataframe(
        results,
        use_container_width=True
    )

    best_model = results.sort_values(
        by="R2 Score",
        ascending=False
    ).iloc[0]

    st.success(
        f"Best Model: {best_model['Model']}"
    )

# =========================
# Footer
# =========================

st.markdown("---")

st.caption(
    "Predictive Maintenance Analytics | Streamlit Dashboard"
)
