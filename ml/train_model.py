# ml/train_model.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)


MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def get_target_column(df):

    possible_targets = [
        "Failure_Status",
        "Fault Detected",
        "Predictive Maintenance Trigger"
    ]

    for col in possible_targets:
        if col in df.columns:
            return col

    raise Exception(
        "No target column found."
    )


def prepare_data(df):

    target = get_target_column(df)

    numeric_cols = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    if target in numeric_cols:
        numeric_cols.remove(target)

    X = df[numeric_cols]

    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def evaluate(y_test, y_pred):

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    return mse, rmse, r2


def train_linear_regression(
    X_train,
    X_test,
    y_train,
    y_test
):

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    joblib.dump(
        model,
        f"{MODEL_DIR}/linear_regression.pkl"
    )

    return evaluate(
        y_test,
        y_pred
    )


def train_knn(
    X_train,
    X_test,
    y_train,
    y_test
):

    model = KNeighborsRegressor(
        n_neighbors=5
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    joblib.dump(
        model,
        f"{MODEL_DIR}/knn_regressor.pkl"
    )

    return evaluate(
        y_test,
        y_pred
    )


def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    joblib.dump(
        model,
        f"{MODEL_DIR}/random_forest.pkl"
    )

    return evaluate(
        y_test,
        y_pred
    )


def train_xgboost(
    X_train,
    X_test,
    y_train,
    y_test
):

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    joblib.dump(
        model,
        f"{MODEL_DIR}/xgboost_model.pkl"
    )

    return evaluate(
        y_test,
        y_pred
    )


def train_all_models(df):

    X_train, X_test, y_train, y_test = prepare_data(df)

    results = []

    mse, rmse, r2 = train_linear_regression(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append([
        "Linear Regression",
        mse,
        rmse,
        r2
    ])

    mse, rmse, r2 = train_knn(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append([
        "KNN",
        mse,
        rmse,
        r2
    ])

    mse, rmse, r2 = train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append([
        "Random Forest",
        mse,
        rmse,
        r2
    ])

    mse, rmse, r2 = train_xgboost(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append([
        "XGBoost",
        mse,
        rmse,
        r2
    ])

    return pd.DataFrame(
        results,
        columns=[
            "Model",
            "MSE",
            "RMSE",
            "R2 Score"
        ]
    )
