# utils/helper.py

import numpy as np
import pandas as pd


def calculate_operating_hours(df):

    if not isinstance(
        df.index,
        pd.DatetimeIndex
    ):
        return 0

    hours = (
        df.index.max()
        -
        df.index.min()
    ).total_seconds() / 3600

    return round(hours, 2)


def get_current_temperature(df):

    if "Temperature (°C)" not in df.columns:
        return 0

    return round(
        df["Temperature (°C)"].iloc[-1],
        2
    )


def get_current_vibration(df):

    if "Vibration (m/s²)" not in df.columns:
        return 0

    return round(
        df["Vibration (m/s²)"].iloc[-1],
        2
    )


def get_failure_count(df):

    possible_targets = [
        "Failure_Status",
        "Fault Detected",
        "Predictive Maintenance Trigger"
    ]

    for col in possible_targets:

        if col in df.columns:

            return int(
                df[col].sum()
            )

    return 0


def get_anomaly_count(df):

    if "Anomaly_Alert" not in df.columns:
        return 0

    return int(
        df["Anomaly_Alert"].sum()
    )


def get_numeric_columns(df):

    return df.select_dtypes(
        include=["number"]
    ).columns.tolist()


def get_feature_columns(df):

    ignore_columns = [
        "Failure_Status",
        "Fault Detected",
        "Predictive Maintenance Trigger",
        "Target",
        "Anomaly_Alert"
    ]

    numeric_cols = get_numeric_columns(df)

    features = [
        col
        for col in numeric_cols
        if col not in ignore_columns
    ]

    return features


def model_comparison_table(results):

    return pd.DataFrame(
        results,
        columns=[
            "Model",
            "MSE",
            "RMSE",
            "R2 Score"
        ]
    )


def detect_failure_threshold(
    df,
    temperature_limit=80,
    vibration_limit=5
):

    if (
        "Temperature (°C)" not in df.columns
        or
        "Vibration (m/s²)" not in df.columns
    ):
        return pd.DataFrame()

    failures = df[
        (
            df["Temperature (°C)"]
            >
            temperature_limit
        )
        |
        (
            df["Vibration (m/s²)"]
            >
            vibration_limit
        )
    ]

    return failures
