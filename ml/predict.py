import pandas as pd
import numpy as np
import joblib

# Load dataset
df = pd.read_csv("data/sensor_maintenance_data.csv")

df = df.dropna()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

target_col = numeric_cols[-1]

feature_cols = numeric_cols[:-1]

# Load Model
model = joblib.load(
    "models/random_forest.pkl"
)

print("\nRequired Features:")
print(feature_cols)

# Use first row as sample
sample = df[feature_cols].iloc[[0]]

prediction = model.predict(sample)

print("\nPrediction Result")
print("-----------------")
print(f"Predicted Value: {prediction[0]}")
