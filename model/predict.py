import pandas as pd
import joblib

FEATURES = [
    "supply_air_temp",
    "return_air_temp",
    "chilled_water_temp",
    "compressor_pressure",
    "vibration"
]

def get_anomaly_scores(df):
    """
    Run the trained Isolation Forest model on the dataframe.
    Returns the dataframe with two new columns:
    - anomaly_score: 0.0 (normal) to 1.0 (anomalous)
    - is_anomaly: True/False flag
    """
    model = joblib.load("model/isolation_forest.pkl")

    X = df[FEATURES].values

    # predict() returns 1 (normal) or -1 (anomaly)
    predictions = model.predict(X)

    # decision_function() returns raw scores — lower = more anomalous
    raw_scores = model.decision_function(X)

    # Normalize scores to 0.0 - 1.0 range (1.0 = most anomalous)
    normalized = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

    df = df.copy()
    df["anomaly_score"] = normalized.round(3)
    df["is_anomaly"]    = predictions == -1

    return df