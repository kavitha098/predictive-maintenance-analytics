import streamlit as st
import pandas as pd
import os

from utils.data_loader import (
    load_data,
    convert_timestamp,
    get_equipment_list,
    filter_equipment,
    get_dataset_info
)

from utils.preprocessing import (
    full_preprocessing,
    prepare_features
)

from utils.helper import (
    calculate_operating_hours,
    current_temperature,
    count_alerts
)

from analytics.feature_engineering import (
    create_all_features
)

from analytics.anomaly_detection import (
    detect_anomalies,
    get_anomaly_records
)

from analytics.root_cause_analysis import (
    failure_summary,
    root_cause_records
)

from analytics.visualization import (
    plot_temperature,
    plot_vibration,
    plot_failure_distribution,
    plot_correlation_heatmap,
    create_wordcloud,
    plot_anomaly_trend
)

from ml.train_models import (
    train_and_save_models
)

from analytics.generate_pdf import (
    create_root_cause_report
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Predictive Maintenance System")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Sensor CSV File",
    type=["csv"]
)

threshold = st.sidebar.slider(
    "Z-Score Threshold",
    min_value=2.0,
    max_value=4.0,
    value=3.0,
    step=0.1
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

if uploaded_file is not None:

    df = load_data(uploaded_file)

    df = convert_timestamp(df)

    df = full_preprocessing(df)

    df = create_all_features(df)

    # Equipment Filter
    equipment_list = get_equipment_list(df)

    selected_equipment = st.sidebar.selectbox(
        "Select Equipment",
        ["All"] + list(equipment_list)
    )

    df = filter_equipment(
        df,
        selected_equipment
    )

    # Detect Anomalies
    df = detect_anomalies(
        df,
        threshold
    )

    # --------------------------------------------------
    # DATASET INFO
    # --------------------------------------------------

    st.subheader("Dataset Information")

    info = get_dataset_info(df)

    col_a, col_b, col_c = st.columns(3)

    col_a.metric("Rows", info["Rows"])
    col_b.metric("Columns", info["Columns"])
    col_c.metric("Missing Values", info["Missing Values"])

    # --------------------------------------------------
    # KPI SECTION
    # --------------------------------------------------

    st.subheader("Real-Time Equipment Metrics")

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "Operating Hours",
        calculate_operating_hours(df)
    )

    k2.metric(
        "Current Temperature (°C)",
        current_temperature(df)
    )

    k3.metric(
        "Critical Alerts",
        count_alerts(df)
    )

    # --------------------------------------------------
    # RAW DATA
    # --------------------------------------------------

    st.subheader("Sensor Data")

    st.dataframe(
        df,
        use_container_width=True
    )

    # --------------------------------------------------
    # TEMPERATURE TREND
    # --------------------------------------------------

    st.subheader("Temperature Trend")

    temp_fig = plot_temperature(df)

    st.pyplot(temp_fig)

    # --------------------------------------------------
    # VIBRATION TREND
    # --------------------------------------------------

    st.subheader("Vibration Trend")

    vib_fig = plot_vibration(df)

    st.pyplot(vib_fig)

    # --------------------------------------------------
    # ANOMALY TREND
    # --------------------------------------------------

    st.subheader("Anomaly Detection")

    anomaly_fig = plot_anomaly_trend(df)

    st.pyplot(anomaly_fig)

    # --------------------------------------------------
    # ANOMALY RECORDS
    # --------------------------------------------------

    st.subheader("Anomaly Records")

    anomaly_df = get_anomaly_records(df)

    st.dataframe(
        anomaly_df,
        use_container_width=True
    )

    # Save Report CSV

    os.makedirs(
        "reports",
        exist_ok=True
    )

    anomaly_df.to_csv(
        "reports/anomaly_report.csv",
        index=False
    )

    # --------------------------------------------------
    # FAILURE ANALYSIS
    # --------------------------------------------------

    st.subheader("Failure Summary")

    summary = failure_summary(df)

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Total Failures",
        summary["Total Failures"]
    )

    s2.metric(
        "Avg Temp",
        summary["Average Temperature"]
    )

    s3.metric(
        "Avg Vibration",
        summary["Average Vibration"]
    )

    s4.metric(
        "Avg Voltage",
        summary["Average Voltage"]
    )

    # --------------------------------------------------
    # ROOT CAUSE ANALYSIS
    # --------------------------------------------------

    st.subheader("Root Cause Analysis")

    root_df = root_cause_records(df)

    st.dataframe(
        root_df,
        use_container_width=True
    )

    # --------------------------------------------------
    # FAILURE DISTRIBUTION
    # --------------------------------------------------

    st.subheader("Failure Distribution")

    fail_fig = plot_failure_distribution(df)

    st.pyplot(fail_fig)

    # --------------------------------------------------
    # HEATMAP
    # --------------------------------------------------

    st.subheader("Correlation Heatmap")

    heatmap_fig = plot_correlation_heatmap(df)

    st.pyplot(heatmap_fig)

    # --------------------------------------------------
    # WORD CLOUD
    # --------------------------------------------------

    st.subheader("Failure Type Word Cloud")

    wc_fig = create_wordcloud(df)

    if wc_fig is not None:
        st.pyplot(wc_fig)

    # --------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------

    st.subheader("Machine Learning Model Evaluation")

    try:

        results_df = train_and_save_models(df)

        st.dataframe(
            results_df,
            use_container_width=True
        )

        results_df.to_csv(
            "reports/model_evaluation.csv",
            index=False
        )

        st.success(
            "Models trained successfully."
        )

    except Exception as e:

        st.error(
            f"Model Training Error: {e}"
        )

    # --------------------------------------------------
    # PDF REPORT
    # --------------------------------------------------

    try:

        failure_df = df[
            df["Fault Detected"] == 1
        ]

        create_root_cause_report(
            failure_df
        )

        st.success(
            "Root Cause PDF Report Generated."
        )

    except Exception as e:

        st.warning(
            f"PDF Generation Error: {e}"
        )

    # --------------------------------------------------
    # DOWNLOADS
    # --------------------------------------------------

    st.subheader("Download Reports")

    if os.path.exists(
        "reports/model_evaluation.csv"
    ):
        with open(
            "reports/model_evaluation.csv",
            "rb"
        ) as f:

            st.download_button(
                "Download Model Evaluation",
                f,
                file_name="model_evaluation.csv"
            )

    if os.path.exists(
        "reports/anomaly_report.csv"
    ):
        with open(
            "reports/anomaly_report.csv",
            "rb"
        ) as f:

            st.download_button(
                "Download Anomaly Report",
                f,
                file_name="anomaly_report.csv"
            )

    if os.path.exists(
        "reports/root_cause_report.pdf"
    ):
        with open(
            "reports/root_cause_report.pdf",
            "rb"
        ) as f:

            st.download_button(
                "Download Root Cause PDF",
                f,
                file_name="root_cause_report.pdf"
            )

else:

    st.info(
        "Upload sensor_maintenance_data.csv to start analysis."
    )
