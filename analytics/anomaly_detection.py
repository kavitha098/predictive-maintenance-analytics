import numpy as np
import pandas as pd


def calculate_zscore(series):
    """
    Calculate Z-score
    """

    mean = series.mean()
    std = series.std()

    if std == 0:
        return pd.Series([0] * len(series))

    return (series - mean) / std


def detect_anomalies(df, threshold=3.0):
    """
    Detect anomalies using Z-score
    """

    df = df.copy()

    df["Temp_Z"] = calculate_zscore(
        df["Temperature (°C)"]
    )

    df["Vibration_Z"] = calculate_zscore(
        df["Vibration (m/s²)"]
    )

    df["Voltage_Z"] = calculate_zscore(
        df["Voltage (V)"]
    )

    df["Anomaly_Alert"] = np.where(
        (abs(df["Temp_Z"]) > threshold)
        |
        (abs(df["Vibration_Z"]) > threshold)
        |
        (abs(df["Voltage_Z"]) > threshold),
        1,
        0
    )

    return df


def get_anomaly_records(df):

    return df[
        df["Anomaly_Alert"] == 1
    ]
