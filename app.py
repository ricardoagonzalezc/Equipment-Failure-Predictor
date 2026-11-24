import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import theme
from rules.thresholds import check_alerts, THRESHOLDS
from model.predict import get_anomaly_scores

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="HVAC Failure Predictor", layout="wide")
theme.apply_theme()
theme.header(
    "HVAC Equipment Failure Predictor",
    "Real-time anomaly detection using rule-based alerts and machine learning.",
    eyebrow="Project · Predictive maintenance",
)

SENSOR_LABELS = {
    "supply_air_temp":     "Supply Air Temp",
    "return_air_temp":     "Return Air Temp",
    "chilled_water_temp":  "Chilled Water Temp",
    "compressor_pressure": "Compressor Pressure",
    "vibration":           "Vibration",
}
TILE_LABELS = {   # short labels so the tiles stay one line
    "supply_air_temp": "Supply air", "return_air_temp": "Return air",
    "chilled_water_temp": "Chilled water", "compressor_pressure": "Compressor", "vibration": "Vibration",
}
# One fixed colour per sensor, so a sensor keeps its colour whatever else is selected.
SENSOR_COLORS = dict(zip(SENSOR_LABELS, theme.SERIES))

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("1. Data Source")
data_source = st.sidebar.radio("Choose data source:", ["Use Sample Data", "Upload CSV"],
                               label_visibility="collapsed")

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
    st.subheader("Latest Sensor Readings")
    latest = df.iloc[-1]

    cols = st.columns(len(SENSOR_LABELS))
    for col, (sensor, label) in zip(cols, TILE_LABELS.items()):
        lim, value = THRESHOLDS[sensor], latest[sensor]
        if value > lim["max"]:
            status, note = "bad", f"HIGH · max {lim['max']:g} {lim['unit']}"
        elif value < lim["min"]:
            status, note = "bad", f"LOW · min {lim['min']:g} {lim['unit']}"
        else:
            status, note = "ok", f"OK · {lim['min']:g}–{lim['max']:g} {lim['unit']}"
        col.markdown(theme.tile(label, f"{value:g}", note, status), unsafe_allow_html=True)
    st.write("")

    # ── ML Anomaly Score ─────────────────────────────────────
    st.subheader("ML Anomaly Risk Score")
    avg_score = df["anomaly_score"].iloc[-50:].mean()
    total_anomalies = df["is_anomaly"].sum()

    col1, col2 = st.columns(2)

    with col1:
        # Gauge chart (zones tinted, needle bar in light ink)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(avg_score * 100, 1),
            title={"text": "Anomaly Risk (last 50 readings)",
                   "font": {"family": theme.FONT_MONO, "size": 13, "color": theme.INK_DIM}},
            number={"font": {"family": theme.FONT_MONO, "color": theme.INK}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": theme.LINE_STRONG,
                         "tickfont": {"family": theme.FONT_MONO, "color": theme.INK_DIM}},
                "bar":  {"color": theme.INK, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  40],  "color": "rgba(102, 217, 160, 0.35)"},
                    {"range": [40, 70],  "color": "rgba(255, 176, 32, 0.35)"},
                    {"range": [70, 100], "color": "rgba(255, 122, 122, 0.40)"},
                ],
                "threshold": {
                    "line":  {"color": theme.CORAL, "width": 3},
                    "thickness": 0.8,
                    "value": 70
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=10))
        st.plotly_chart(fig_gauge, width="stretch")

    with col2:
        st.markdown(theme.tile("Total anomalies detected", f"{total_anomalies} / {len(df)}",
                               "readings flagged by the model"), unsafe_allow_html=True)
        st.write("")
        if avg_score > 0.70:
            st.error("HIGH RISK — Immediate inspection recommended")
        elif avg_score > 0.40:
            st.warning("MODERATE RISK — Monitor closely")
        else:
            st.success("System operating normally")

    # ── Time Series Chart ─────────────────────────────────────
    st.subheader("Sensor Trends Over Time")
    selected = st.multiselect(
        "Select sensors to display:",
        list(SENSOR_LABELS),
        default=["compressor_pressure", "vibration"],
        format_func=lambda s: SENSOR_LABELS[s],
    )

    if selected:
        # One panel per sensor: the sensors have very different scales (e.g. pressure ~220 vs
        # vibration ~1), so sharing one axis would flatten the small ones.
        fig = make_subplots(rows=len(selected), cols=1, shared_xaxes=True, vertical_spacing=0.07)
        for i, sensor in enumerate(selected, start=1):
            lim = THRESHOLDS[sensor]
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[sensor], mode="lines", name=SENSOR_LABELS[sensor],
                line=dict(color=SENSOR_COLORS[sensor], width=1.5),
            ), row=i, col=1)
            for y in (lim["min"], lim["max"]):        # alert limits
                fig.add_hline(y=y, row=i, col=1, line=dict(color=theme.INK_DIM, width=1, dash="dash"))
            fig.update_yaxes(title_text=lim["unit"], row=i, col=1)

        # Merge consecutive anomalous readings into single bands
        flag = df["is_anomaly"].to_numpy()
        ts = df["timestamp"].to_numpy()
        start = None
        for k in range(len(flag) + 1):
            on = k < len(flag) and flag[k]
            if on and start is None:
                start = k
            elif not on and start is not None:
                fig.add_vrect(x0=ts[start], x1=ts[k - 1], fillcolor=theme.CORAL, opacity=0.18,
                              line_width=0, row="all", col=1)
                start = None

        fig.update_xaxes(title_text="Timestamp", row=len(selected), col=1)
        fig.update_layout(height=140 + 230 * len(selected), showlegend=True,
                          margin=dict(l=64, r=24, t=48, b=48))
        st.plotly_chart(fig, width="stretch")
        st.caption("Dashed lines are the alert limits. Coral bands mark readings the ML model flagged as anomalous.")

    # ── Alert Log ─────────────────────────────────────────────
    st.subheader("Rule-Based Alert Log")
    if len(alerts_df) > 0:
        st.error(f"{len(alerts_df)} alerts triggered")
        st.dataframe(alerts_df, width="stretch")
    else:
        st.success("No threshold violations detected")

else:
    st.info("Select a data source to get started.")
