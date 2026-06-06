def add_rolling_features(df):

    df = df.copy()

    df["Temp_Rolling_Mean"] = (
        df["Temperature (°C)"]
        .rolling(window=10)
        .mean()
    )

    df["Vib_Rolling_Mean"] = (
        df["Vibration (m/s²)"]
        .rolling(window=10)
        .mean()
    )

    df.fillna(method="bfill", inplace=True)

    return df
