import numpy as np


def add_rolling_features(df):

    df = df.copy()

    if "Temperature (°C)" in df.columns:

        df["Temp_Rolling_Mean"] = (
            df["Temperature (°C)"]
            .rolling(window=10, min_periods=1)
            .mean()
        )

        df["Temp_Rolling_STD"] = (
            df["Temperature (°C)"]
            .rolling(window=10, min_periods=1)
            .std()
        )

    if "Vibration (m/s²)" in df.columns:

        df["Vib_Rolling_Mean"] = (
            df["Vibration (m/s²)"]
            .rolling(window=10, min_periods=1)
            .mean()
        )

        df["Vib_Rolling_STD"] = (
            df["Vibration (m/s²)"]
            .rolling(window=10, min_periods=1)
            .std()
        )

    numeric_cols = df.select_dtypes(include=np.number).columns

    df[numeric_cols] = df[numeric_cols].bfill()
    df[numeric_cols] = df[numeric_cols].ffill()

    return df
