import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from rules.thresholds import check_alerts
from model.predict import get_anomaly_scores

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="HVAC Failure Predictor", layout="wide")
st.title("🌡️ HVAC Equipment Failure Predictor")
st.markdown("Real-time anomaly detection using rule-based alerts and machine learning.")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("1. Data Source")
data_source = st.sidebar.radio("Choose data source:", ["Use Sample Data", "Upload CSV"])

df = None

if data_source == "Use Sample Data":
    df = pd.read_csv("data/HVAC_sensor_data.csv", parse_dates=["timestamp"])
    st.sidebar.success("Sample data loaded (1000 readings)")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
        st.sidebar.success(f"{len(df)} rows loaded")

# ── Main Panel ────────────────────────────────────────────────
if df is not None:

    # Run ML scoring
    df = get_anomaly_scores(df)

    # Run rule-based alerts
    alerts_df = check_alerts(df)

    # ── Sensor Cards ─────────────────────────────────────────
    st.subheader("📡 Latest Sensor Readings")
    latest = df.iloc[-1]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Supply Air Temp",     f"{latest['supply_air_temp']} °C")
    c2.metric("Return Air Temp",     f"{latest['return_air_temp']} °C")
    c3.metric("Chilled Water Temp",  f"{latest['chilled_water_temp']} °C")
    c4.metric("Compressor Pressure", f"{latest['compressor_pressure']} PSI")
    c5.metric("Vibration",           f"{latest['vibration']} mm/s")

    # ── ML Anomaly Score ─────────────────────────────────────
    st.subheader("ML Anomaly Risk Score")
    avg_score = df["anomaly_score"].iloc[-50:].mean()
    total_anomalies = df["is_anomaly"].sum()

    col1, col2 = st.columns(2)

    with col1:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(avg_score * 100, 1),
            title={"text": "Anomaly Risk (last 50 readings)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "darkred"},
                "steps": [
                    {"range": [0,  40],  "color": "green"},
                    {"range": [40, 70],  "color": "orange"},
                    {"range": [70, 100], "color": "red"},
                ],
                "threshold": {
                    "line":  {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 70
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.metric("Total Anomalies Detected", f"{total_anomalies} / {len(df)} readings")
        if avg_score > 0.70:
            st.error("🔴 HIGH RISK — Immediate inspection recommended")
        elif avg_score > 0.40:
            st.warning("⚠️ MODERATE RISK — Monitor closely")
        else:
            st.success("✅ System operating normally")

    # ── Time Series Chart ─────────────────────────────────────
    st.subheader("📈 Sensor Trends Over Time")
    sensor_options = [
        "supply_air_temp", "return_air_temp",
        "chilled_water_temp", "compressor_pressure", "vibration"
    ]
    selected = st.multiselect(
        "Select sensors to display:",
        sensor_options,
        default=["compressor_pressure", "vibration"]
    )

    if selected:
        fig = go.Figure()
        for sensor in selected:
            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df[sensor],
                mode="lines",
                name=sensor
            ))

        # Highlight anomaly regions
        anomaly_times = df[df["is_anomaly"]]["timestamp"]
        for t in anomaly_times:
            fig.add_vline(x=t, line_color="red", opacity=0.1)

        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Sensor Value",
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Alert Log ─────────────────────────────────────────────
    st.subheader("🚨 Rule-Based Alert Log")
    if len(alerts_df) > 0:
        st.error(f"{len(alerts_df)} alerts triggered")
        st.dataframe(alerts_df, use_container_width=True)
    else:
        st.success("No threshold violations detected")

else:
    st.info("👈 Select a data source in the sidebar to get started.")