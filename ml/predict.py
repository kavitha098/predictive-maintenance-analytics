# ml/predict.py

import joblib
import pandas as pd


def load_model(model_name):

    model_paths = {
        "Linear Regression":
        "models/linear_regression.pkl",

        "KNN":
        "models/knn_regressor.pkl",

        "Random Forest":
        "models/random_forest.pkl",

        "XGBoost":
        "models/xgboost_model.pkl"
    }

    return joblib.load(
        model_paths[model_name]
    )


def make_prediction(
    model_name,
    input_df
):

    model = load_model(model_name)

    prediction = model.predict(
        input_df
    )

    return prediction
