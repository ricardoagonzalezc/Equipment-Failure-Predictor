import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_model():
    # Load the sensor data
    df = pd.read_csv("data/HVAC_sensor_data.csv")

    # Select only the numeric sensor columns for training
    features = [
        "supply_air_temp",
        "return_air_temp",
        "chilled_water_temp",
        "compressor_pressure",
        "vibration"
    ]

    X = df[features].values

    # Train Isolation Forest
    # contamination = expected proportion of anomalies in the data
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    model.fit(X)

    # Save the trained model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/isolation_forest.pkl")
    print("Model trained and saved to model/isolation_forest.pkl")

if __name__ == "__main__":
    train_model()