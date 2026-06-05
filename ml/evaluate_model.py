import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

# Load Dataset
df = pd.read_csv("data/sensor_maintenance_data.csv")

df = df.dropna()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

target_col = numeric_cols[-1]

X = df[numeric_cols[:-1]]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Linear Regression":
        joblib.load("models/linear_regression.pkl"),

    "KNN Regression":
        joblib.load("models/knn_regressor.pkl"),

    "Random Forest":
        joblib.load("models/random_forest.pkl")
}

print("\nModel Evaluation Results")
print("-" * 60)

for name, model in models.items():

    y_pred = model.predict(X_test)

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    print(f"\n{name}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
