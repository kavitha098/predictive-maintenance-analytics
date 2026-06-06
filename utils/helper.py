import numpy as np
import pandas as pd


def calculate_zscore(series):
    """
    Calculate z-score.
    """

    mean = series.mean()
    std = series.std()

    if std == 0:
        return np.zeros(len(series))

    return (series - mean) / std


def calculate_anomaly_alerts(
    df,
    threshold=3.0
):
    """
    Generate anomaly alerts using z-score.
    """

    df = df.copy()

    if "Temperature (°C)" in df.columns:

        df["Temp_Z"] = calculate_zscore(
            df["Temperature (°C)"]
        )

    if "Vibration (m/s²)" in df.columns:

        df["Vibration_Z"] = calculate_zscore(
            df["Vibration (m/s²)"]
        )

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


def calculate_failure_threshold(df):
    """
    Statistical failure ceiling.
    """

    if "Temperature (°C)" not in df.columns:
        return 0

    return (
        df["Temperature (°C)"].mean()
        +
        (
            3 *
            df["Temperature (°C)"].std()
        )
    )


def get_operating_hours(df):
    """
    Estimate operating hours.
    """

    if len(df) == 0:
        return 0

    return len(df)


def get_current_temperature(df):
    """
    Latest temperature.
    """

    if "Temperature (°C)" not in df.columns:
        return 0

    return round(
        df["Temperature (°C)"].iloc[-1],
        2
    )


def get_active_alert_count(df):
    """
    Count anomaly alerts.
    """

    if "Anomaly_Alert" not in df.columns:
        return 0

    return int(df["Anomaly_Alert"].sum())


def model_metrics(
    mse,
    rmse,
    r2
):
    """
    Return evaluation metrics.
    """

    return {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4)
    }


def create_metrics_dataframe(results):
    """
    Convert model results dictionary
    to dataframe.
    """

    return pd.DataFrame(results).T
