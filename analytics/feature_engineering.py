import pandas as pd


def create_time_features(df):

    if "Timestamp" in df.columns:

        df["Hour"] = df["Timestamp"].dt.hour

        df["Day"] = df["Timestamp"].dt.day

        df["Month"] = df["Timestamp"].dt.month

        df["Weekday"] = (
            df["Timestamp"]
            .dt.day_name()
        )

    return df


def rolling_features(df):

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


def operating_conditions(df):

    df["Power_Factor"] = (
        df["Voltage (V)"] *
        df["Current (A)"]
    )

    return df


def create_all_features(df):

    df = create_time_features(df)

    df = rolling_features(df)

    df = operating_conditions(df)

    return df
