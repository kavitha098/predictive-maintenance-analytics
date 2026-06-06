import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor


def train_models(df):

    required_columns = [
        "Temperature (°C)",
        "Vibration (m/s²)",
        "Voltage (V)",
        "Fault Detected"
    ]

    missing_cols = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}"
        )

    features = [
        "Temperature (°C)",
        "Vibration (m/s²)",
        "Voltage (V)"
    ]

    X = df[features].copy()
    y = df["Fault Detected"].copy()

    X = X.fillna(X.mean())
    y = y.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),

        "KNN": KNeighborsRegressor(
            n_neighbors=5
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=100,
            random_state=42,
            objective="reg:squarederror"
        )
    }

    # Create folder automatically
    os.makedirs(
        "maintenance_models",
        exist_ok=True
    )

    results = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(
            X_test
        )

        mse = mean_squared_error(
            y_test,
            pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            pred
        )

        results[name] = {
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        }

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        joblib.dump(
            model,
            f"maintenance_models/{filename}"
        )

    return results
