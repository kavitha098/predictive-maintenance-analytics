import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)


def calculate_rmse(y_true, y_pred):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    return np.sqrt(mse)


def evaluate_model(y_true, y_pred):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4)
    }


def calculate_operating_hours(df):

    return len(df)


def current_temperature(df):

    if "Temperature (°C)" in df.columns:
        return round(
            df["Temperature (°C)"].iloc[-1],
            2
        )

    return 0


def count_alerts(df):

    if "Anomaly_Alert" in df.columns:
        return int(
            df["Anomaly_Alert"].sum()
        )

    return 0


def safe_division(a, b):

    if b == 0:
        return 0

    return a / b


def format_percentage(value):

    return f"{value:.2f}%"
