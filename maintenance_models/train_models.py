import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

import numpy as np


def train_models(df):

    features = [
        "Temperature (°C)",
        "Vibration (m/s²)",
        "Voltage (V)"
    ]

    X = df[features]

    y = df["Fault Detected"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "KNN": KNeighborsRegressor(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor()
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mse = mean_squared_error(y_test, pred)

        rmse = np.sqrt(mse)

        r2 = r2_score(y_test, pred)

        results[name] = {
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        joblib.dump(model, f"models/{filename}")

    return results
