# analytics/anomaly_detection.py

import numpy as np
import pandas as pd


def calculate_z_scores(df):
    """
    Calculate Z-Scores for Temperature and Vibration
    """

    df = df.copy()

    if "Temperature (°C)" in df.columns:
        df["Temp_Z"] = (
            df["Temperature (°C)"]
            - df["Temperature (°C)"].mean()
        ) / df["Temperature (°C)"].std()

    if "Vibration (m/s²)" in df.columns:
        df["Vibration_Z"] = (
            df["Vibration (m/s²)"]
            - df["Vibration (m/s²)"].mean()
        ) / df["Vibration (m/s²)"].std()

    return df


def detect_anomalies(df, threshold=3.0):
    """
    Generate anomaly alerts
    """

    df = calculate_z_scores(df)

    df["Anomaly_Alert"] = np.where(
        (abs(df["Temp_Z"]) > threshold)
        | (abs(df["Vibration_Z"]) > threshold),
        1,
        0,
    )

    return df


def get_anomaly_count(df):
    if "Anomaly_Alert" not in df.columns:
        return 0

    return int(df["Anomaly_Alert"].sum())


def get_anomaly_rows(df):
    if "Anomaly_Alert" not in df.columns:
        return pd.DataFrame()

    return df[df["Anomaly_Alert"] == 1]
