import pandas as pd
import numpy as np


def handle_missing_values(df):
    """
    Fill missing values
    """

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(
            df[col].median()
        )

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    return df


def create_rolling_features(df):
    """
    Rolling mean and std
    """

    if "Temperature (°C)" in df.columns:

        df["Temp_Rolling_Mean"] = (
            df["Temperature (°C)"]
            .rolling(window=10)
            .mean()
        )

        df["Temp_Rolling_STD"] = (
            df["Temperature (°C)"]
            .rolling(window=10)
            .std()
        )

    if "Vibration (m/s²)" in df.columns:

        df["Vib_Rolling_Mean"] = (
            df["Vibration (m/s²)"]
            .rolling(window=10)
            .mean()
        )

        df["Vib_Rolling_STD"] = (
            df["Vibration (m/s²)"]
            .rolling(window=10)
            .std()
        )

    return df


def remove_duplicates(df):
    """
    Remove duplicate rows
    """
    return df.drop_duplicates()


def prepare_features(df):
    """
    Select ML features
    """

    features = [
        "Temperature (°C)",
        "Vibration (m/s²)",
        "Voltage (V)",
        "Current (A)",
        "Humidity (%)",
        "Power (W)"
    ]

    available_features = [
        col for col in features
        if col in df.columns
    ]

    X = df[available_features]

    y = df["Fault Detected"]

    return X, y


def full_preprocessing(df):

    df = handle_missing_values(df)

    df = remove_duplicates(df)

    df = create_rolling_features(df)

    return df
