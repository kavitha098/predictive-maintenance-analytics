import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import (
    load_data,
    get_equipment_ids,
    filter_equipment
)

from analytics.anomaly_detection import detect_anomalies
from analytics.feature_engineering import add_rolling_features
from analytics.root_cause_analysis import (
    equipment_failure_analysis,
    failure_type_analysis
)

from models.train_models import train_models

st.set_page_config(
    page_title="Predictive Maintenance Analytics",
    layout="wide"
)

st.title("Predictive Maintenance Dashboard")

uploaded_file = st.sidebar.file_uploader(
    "Upload Sensor CSV",
    type=["csv"]
)

if uploaded_file:

    df = load_data(uploaded_file)

    threshold = st.sidebar.slider(
        "Z Score Threshold",
        2.0,
        4.0,
        3.0,
        0.1
    )

    equipment_ids = get_equipment_ids(df)

    selected_equipment = st.sidebar.selectbox(
        "Equipment",
        equipment_ids
    )

    df = filter_equipment(
        df,
        selected_equipment
    )

    df = add_rolling_features(df)

    df = detect_anomalies(
        df,
        threshold
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Operating Hours",
        len(df)
    )

    col2.metric(
        "Current Temperature",
        round(
            df["Temperature (°C)"].iloc[-1],
            2
        )
    )

    col3.metric(
        "Critical Alerts",
        int(df["Anomaly_Alert"].sum())
    )

    st.subheader("Temperature Trend")

    st.line_chart(
        df["Temperature (°C)"]
    )

    failure_limit = (
        df["Temperature (°C)"].mean()
        + 3 *
        df["Temperature (°C)"].std()
    )

    st.write(
        f"Failure Ceiling: {failure_limit:.2f}"
    )

    st.subheader(
        "Active Failure Alerts"
    )

    st.dataframe(
        df[df["Anomaly_Alert"] == 1]
    )

    st.subheader(
        "Equipment Failure Analysis"
    )

    st.dataframe(
        equipment_failure_analysis(df)
    )

    st.subheader(
        "Failure Type Analysis"
    )

    st.dataframe(
        failure_type_analysis(df)
    )

    st.subheader(
        "Machine Learning Evaluation"
    )

    results = train_models(df)

    st.dataframe(
        pd.DataFrame(results).T
    )

    st.bar_chart(
        pd.DataFrame(results).T[
            ["RMSE"]
        ]
    )
