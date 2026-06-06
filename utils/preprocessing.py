import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def clean_data(df):
    """
    Clean dataset.
    """

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill numeric null values
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical null values
    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    return df


def encode_categorical_columns(df):
    """
    Encode categorical columns.
    """

    df = df.copy()

    encoders = {}

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in categorical_cols:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = encoder

    return df, encoders


def create_target_column(df):
    """
    Create Failure_Status if missing.
    """

    df = df.copy()

    if "Failure_Status" not in df.columns:

        if "Fault Detected" in df.columns:
            df["Failure_Status"] = df["Fault Detected"]

        elif "Predictive Maintenance Trigger" in df.columns:
            df["Failure_Status"] = (
                df["Predictive Maintenance Trigger"]
            )

        else:
            df["Failure_Status"] = 0

    return df


def add_rolling_features(df, window=10):
    """
    Create rolling statistics.
    """

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

        df["Vib_Rolling_Mean"] = (
            df["Vibration (m/s²)"]
            .rolling(window)
            .mean()
        )

        df["Vib_Rolling_STD"] = (
            df["Vibration (m/s²)"]
            .rolling(window)
            .std()
        )

    df = df.bfill()

    return df


def select_model_features(df):
    """
    Select ML features.
    """

    features = []

    possible_features = [
        "Temperature (°C)",
        "Vibration (m/s²)",
        "Voltage (V)",
        "Temp_Rolling_Mean",
        "Vib_Rolling_Mean",
        "Temp_Rolling_STD",
        "Vib_Rolling_STD"
    ]

    for feature in possible_features:
        if feature in df.columns:
            features.append(feature)

    return features
