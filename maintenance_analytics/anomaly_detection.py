import numpy as np


def detect_anomalies(df, threshold=3.0):

    df = df.copy()

    temp_col = "Temperature (°C)"
    vib_col = "Vibration (m/s²)"

    if temp_col not in df.columns:
        raise ValueError(f"Missing column: {temp_col}")

    if vib_col not in df.columns:
        raise ValueError(f"Missing column: {vib_col}")

    temp_mean = df[temp_col].mean()
    temp_std = df[temp_col].std()

    vib_mean = df[vib_col].mean()
    vib_std = df[vib_col].std()

    if temp_std == 0:
        temp_std = 1

    if vib_std == 0:
        vib_std = 1

    df["Temp_Z"] = (
        df[temp_col] - temp_mean
    ) / temp_std

    df["Vibration_Z"] = (
        df[vib_col] - vib_mean
    ) / vib_std

    df["Anomaly_Alert"] = np.where(
        (
            abs(df["Temp_Z"]) > threshold
        )
        |
        (
            abs(df["Vibration_Z"]) > threshold
        ),
        1,
        0
    )

    return df
