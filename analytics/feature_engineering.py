# analytics/feature_engineering.py

import pandas as pd


def create_time_features(df):

    df = df.copy()

    if isinstance(df.index, pd.DatetimeIndex):

        df["Year"] = df.index.year
        df["Month"] = df.index.month
        df["Day"] = df.index.day
        df["Hour"] = df.index.hour

    return df


def create_rolling_features(df, window=10):

    df = df.copy()

    if "Temperature (°C)" in df.columns:

        df["Temp_Rolling_Mean"] = (
            df["Temperature (°C)"]
            .rolling(window)
            .mean()
        )

        df["Temp_Rolling_STD"] = (
            df["Temperature (°C)"]
            .rolling(window)
            .std()
        )

    if "Vibration (m/s²)" in df.columns:

        df["Vibration_Rolling_Mean"] = (
            df["Vibration (m/s²)"]
            .rolling(window)
            .mean()
        )

        df["Vibration_Rolling_STD"] = (
            df["Vibration (m/s²)"]
            .rolling(window)
            .std()
        )

    return df


def create_failure_feature(df):

    possible_targets = [
        "Failure_Status",
        "Fault Detected",
        "Predictive Maintenance Trigger"
    ]

    for col in possible_targets:

        if col in df.columns:

            df["Failure_Target"] = df[col]

            break

    return df
