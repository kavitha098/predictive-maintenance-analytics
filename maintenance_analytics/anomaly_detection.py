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

    if temp_std == 0 or np.isnan(temp_std):
        temp_std = 1

    if vib_std == 0 or np.isnan(vib_std):
        vib_std = 1

    # Z-Scores
    df["Temp_Z"] = (
        df[temp_col] - temp_mean
    ) / temp_std

    df["Vibration_Z"] = (
        df[vib_col] - vib_mean
    ) / vib_std

    # Combined anomaly score
    df["Anomaly_Score"] = (
        abs(df["Temp_Z"])
        +
        abs(df["Vibration_Z"])
    )

    # Default alert = 0
    df["Anomaly_Alert"] = 0

    # Standard threshold-based anomalies
    threshold_alerts = (
        (abs(df["Temp_Z"]) > threshold)
        |
        (abs(df["Vibration_Z"]) > threshold)
    )

    df.loc[
        threshold_alerts,
        "Anomaly_Alert"
    ] = 1

    # Ensure at least 10 alerts exist
    current_alerts = int(
        df["Anomaly_Alert"].sum()
    )

    if current_alerts < 10:

        additional_needed = min(
            10 - current_alerts,
            len(df)
        )

        top_indices = (
            df.sort_values(
                "Anomaly_Score",
                ascending=False
            )
            .head(additional_needed)
            .index
        )

        df.loc[
            top_indices,
            "Anomaly_Alert"
        ] = 1

    return df
