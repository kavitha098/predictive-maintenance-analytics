import numpy as np


def detect_anomalies(df, threshold=3):

    df = df.copy()

    cols = [
        "Temperature (°C)",
        "Vibration (m/s²)"
    ]

    for col in cols:

        mean = df[col].mean()
        std = df[col].std()

        zscore = (df[col] - mean) / std

        df[f"{col}_Z"] = zscore

    df["Anomaly_Alert"] = np.where(
        (
            abs(df["Temperature (°C)_Z"]) > threshold
        ) |
        (
            abs(df["Vibration (m/s²)_Z"]) > threshold
        ),
        1,
        0
    )

    return df
