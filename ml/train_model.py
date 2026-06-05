import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/sensor_maintenance_data.csv")

# Remove missing values
df = df.dropna()

# Select numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    raise ValueError("Dataset must contain at least 2 numeric columns.")

# Last numeric column as target
target_col = numeric_cols[-1]

# Features
X = df[numeric_cols[:-1]]

# Target
y = df[target_col]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# KNN Regression
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

# Save Models
joblib.dump(lr, "models/linear_regression.pkl")
joblib.dump(knn, "models/knn_regressor.pkl")
joblib.dump(rf, "models/random_forest.pkl")

print("Models trained and saved successfully.")
print(f"Target Column: {target_col}")
