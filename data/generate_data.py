import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000  # 1000 hourly readings (~41 days)

# Timestamps
timestamps = pd.date_range(start="2024-01-01", periods=n, freq="h")

# Normal sensor readings
supply_air_temp      = np.random.normal(14.0, 0.8, n)
return_air_temp      = np.random.normal(24.0, 0.8, n)
chilled_water_temp   = np.random.normal(8.0,  0.5, n)
compressor_pressure  = np.random.normal(225,  10,  n)
vibration            = np.random.normal(1.5,  0.3, n)
runtime_hours        = np.cumsum(np.ones(n))

# Inject fault events
compressor_pressure[300:320] += np.random.uniform(50, 80, 20)
vibration[600:615]           += np.random.uniform(3, 5, 15)
chilled_water_temp[800:820]  += np.random.uniform(5, 8, 20)

# Build dataframe
df = pd.DataFrame({
    "timestamp":            timestamps,
    "supply_air_temp":      supply_air_temp.round(2),
    "return_air_temp":      return_air_temp.round(2),
    "chilled_water_temp":   chilled_water_temp.round(2),
    "compressor_pressure":  compressor_pressure.round(2),
    "vibration":            vibration.round(3),
    "runtime_hours":        runtime_hours.astype(int)
})

df.to_csv("data/HVAC_sensor_data.csv", index=False)
print(f"Dataset created: {len(df)} rows with 3 injected fault events.")