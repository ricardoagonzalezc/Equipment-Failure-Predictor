import pandas as pd

# Define the alert thresholds for each sensor
THRESHOLDS = {
    "supply_air_temp":     {"min": 10.0,  "max": 18.0,  "unit": "°C"},
    "return_air_temp":     {"min": 20.0,  "max": 28.0,  "unit": "°C"},
    "chilled_water_temp":  {"min": 5.0,   "max": 12.0,  "unit": "°C"},
    "compressor_pressure": {"min": 180.0, "max": 270.0, "unit": "PSI"},
    "vibration":           {"min": 0.0,   "max": 4.0,   "unit": "mm/s"},
}

def check_alerts(df):
    """
    Check each row of sensor data against thresholds.
    Returns a dataframe of triggered alerts.
    """
    alerts = []

    for _, row in df.iterrows():
        for sensor, limits in THRESHOLDS.items():
            value = row[sensor]
            if value > limits["max"]:
                alerts.append({
                    "timestamp": row["timestamp"],
                    "sensor":    sensor,
                    "value":     value,
                    "limit":     limits["max"],
                    "type":      "HIGH",
                    "unit":      limits["unit"]
                })
            elif value < limits["min"]:
                alerts.append({
                    "timestamp": row["timestamp"],
                    "sensor":    sensor,
                    "value":     value,
                    "limit":     limits["min"],
                    "type":      "LOW",
                    "unit":      limits["unit"]
                })

    if alerts:
        return pd.DataFrame(alerts)
    else:
        return pd.DataFrame(columns=["timestamp", "sensor", "value", "limit", "type", "unit"])