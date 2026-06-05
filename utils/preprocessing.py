# utils/preprocessing.py

import pandas as pd
import numpy as np


def clean_data(df):
    """
    Basic cleaning
    """

    df = df.copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Fill numeric missing values
    numeric_cols = df.select_dtypes(
        include=["number"]
    ).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(
            df[col].median()
        )

    # Fill categorical missing values
    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    for col in categorical_cols:
        df[col] = df[col].fillna(
            "Unknown"
        )

    return df


def convert_timestamp(df):

    if "Timestamp" in df.columns:

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

        df = df.sort_values(
            "Timestamp"
        )

        df.set_index(
            "Timestamp",
            inplace=True
        )

    return df


def create_target_column(df):

    possible_targets = [
        "Failure_Status",
        "Fault Detected",
        "Predictive Maintenance Trigger"
    ]

    for col in possible_targets:

        if col in df.columns:

            df["Target"] = df[col]

            break

    return df


def create_rolling_features(
    df,
    window=10
):

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


def create_time_features(df):

    if isinstance(
        df.index,
        pd.DatetimeIndex
    ):

        df["Year"] = df.index.year
        df["Month"] = df.index.month
        df["Day"] = df.index.day
        df["Hour"] = df.index.hour

    return df


def preprocess_pipeline(df):

    df = clean_data(df)

    df = convert_timestamp(df)

    df = create_target_column(df)

    df = create_rolling_features(df)

    df = create_time_features(df)

    return df
