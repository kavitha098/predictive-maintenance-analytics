# ml/evaluate_model.py

import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)


def calculate_metrics(
    y_true,
    y_pred
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mse
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
