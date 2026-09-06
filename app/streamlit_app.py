"""London AQI Dashboard — redesigned single-city interface."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="London Air Quality Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🌫️",
)

# ── Custom styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Overall background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Card panels */
    .aqi-card {
        background: #1c1f2b;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #2e3148;
        margin-bottom: 12px;
    }

    /* Big AQI badge */
    .aqi-badge {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        width: 130px;
        height: 130px;
        margin: 0 auto 10px auto;
        font-size: 2.4rem;
        font-weight: 800;
        color: #fff;
        box-shadow: 0 0 28px rgba(0,0,0,0.5);
    }
    .aqi-badge small {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
        opacity: 0.85;
    }

    /* Section headers */
    .section-title {
        font-size: 0.78rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #7b8199;
        margin-bottom: 6px;
    }

    /* Metric overrides */
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; color: #dde3f5 !important; }
    [data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: #7b8199 !important; }
    [data-testid="stMetricDelta"] { font-size: 0.72rem !important; }

    /* Hide streamlit default hamburger / footer */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

from src import config
from src.feature_pipeline.feature_store import fetch_features
from src.inference_pipeline.predict import daily_summary, predict_next_72h

CITY = "london"

# ── AQI helpers ───────────────────────────────────────────────────────────────
AQI_BANDS = [
    (50,  "Good",                          "#27ae60"),
    (100, "Moderate",                       "#f39c12"),
    (150, "Unhealthy for Sensitive Groups", "#e67e22"),
    (200, "Unhealthy",                      "#e74c3c"),
    (300, "Very Unhealthy",                 "#8e44ad"),
    (999, "Hazardous",                      "#7d0d1b"),
]

def aqi_band(aqi: float) -> tuple[str, str]:
    for ceiling, label, color in AQI_BANDS:
        if aqi <= ceiling:
            return label, color
    return "Hazardous", "#7d0d1b"


# ── Cloud login ───────────────────────────────────────────────────────────────
@st.cache_resource
def _warm_login():
    import hopsworks
    return hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT or None,
    )

@st.cache_data(ttl=600, show_spinner=False)
def cached_features() -> pd.DataFrame:
    df = fetch_features(CITY, limit=200)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data(ttl=600, show_spinner=False)
def cached_forecast() -> pd.DataFrame:
    fc = predict_next_72h(CITY)
    fc["timestamp"] = pd.to_datetime(fc["timestamp"])
    return fc

try:
    _warm_login()
except Exception:
    pass

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([0.08, 0.92])
with col_logo:
    st.markdown("<div style='font-size:2.8rem;padding-top:6px'>🌫️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown(
        "<h1 style='margin:0;padding-top:8px;font-size:1.9rem;color:#dde3f5;'>London Air Quality Monitor</h1>"
        "<p style='margin:0;color:#7b8199;font-size:0.85rem;'>Real-time AQI · 72-hour ML Forecast · Cloud-served</p>",
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border-color:#2e3148;margin:12px 0 18px 0'>", unsafe_allow_html=True)

# ── Data load ────────────────────────────────────────────────────────────────
try:
    with st.spinner("Loading London data..."):
        features = cached_features()
except Exception as e:
    st.error(f"Feature load failed: {e}")
    st.stop()

if features.empty:
    st.warning("No cloud data for London. Run the feature pipeline first.")
    st.stop()

hist = features.dropna(subset=["aqi"]).copy()
last_aqi  = float(hist.iloc[-1]["aqi"])
last_ts   = pd.to_datetime(hist.iloc[-1]["timestamp"])
label, color = aqi_band(last_aqi)

# ── Top row: badge + 4 KPI tiles ─────────────────────────────────────────────
badge_col, kpi1, kpi2, kpi3, kpi4 = st.columns([1.3, 1, 1, 1, 1])

with badge_col:
    st.markdown(
        f"<div class='aqi-badge' style='background:{color};'>"
        f"{last_aqi:.0f}<small>{label}</small></div>",
        unsafe_allow_html=True,
    )

with kpi1:
    st.markdown("<div class='aqi-card'>", unsafe_allow_html=True)
    st.metric("Current AQI", f"{last_aqi:.0f}")
    st.markdown("</div>", unsafe_allow_html=True)

with kpi2:
    avg24 = hist.tail(24)["aqi"].mean() if len(hist) >= 24 else hist["aqi"].mean()
    st.markdown("<div class='aqi-card'>", unsafe_allow_html=True)
    st.metric("24h Average", f"{avg24:.0f}")
    st.markdown("</div>", unsafe_allow_html=True)

with kpi3:
    st.markdown("<div class='aqi-card'>", unsafe_allow_html=True)
    st.metric("Data Points", f"{len(hist)}")
    st.markdown("</div>", unsafe_allow_html=True)

with kpi4:
    st.markdown("<div class='aqi-card'>", unsafe_allow_html=True)
    st.metric("Last Updated", last_ts.strftime("%d %b · %H:%M"))
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── AQI scale strip ──────────────────────────────────────────────────────────
scale_html = "<div style='display:flex;gap:6px;margin-bottom:18px;'>"
for _, lbl, clr in AQI_BANDS:
    is_active = (lbl == label)
    border = "3px solid #fff" if is_active else "3px solid transparent"
    scale_html += (
        f"<div style='flex:1;background:{clr};border-radius:6px;padding:6px 4px;"
        f"text-align:center;font-size:0.68rem;font-weight:600;color:#fff;"
        f"border:{border};opacity:{'1' if is_active else '0.55'}'>{lbl}</div>"
    )
scale_html += "</div>"
st.markdown(scale_html, unsafe_allow_html=True)

# ── Forecast load ────────────────────────────────────────────────────────────
try:
    with st.spinner("Generating 72h forecast..."):
        fc = cached_forecast()
except Exception as e:
    st.error(f"Forecast failed: {e}")
    st.stop()

# ── Charts side-by-side ───────────────────────────────────────────────────────
chart_dark = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=340,
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
    font=dict(color="#c0c6d9"),
    xaxis=dict(gridcolor="#252836", linecolor="#252836"),
    yaxis=dict(gridcolor="#252836", linecolor="#252836", title="AQI"),
)

ch1, ch2 = st.columns(2)

with ch1:
    st.markdown("<p class='section-title'>Observed History</p>", unsafe_allow_html=True)
    hfig = go.Figure()
    hfig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["aqi"],
        name="Observed", mode="lines",
        line=dict(color="#4fc3f7", width=2),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.07)",
    ))
    hfig.update_layout(**chart_dark, title=dict(text="", x=0))
    st.plotly_chart(hfig, use_container_width=True)

with ch2:
    st.markdown("<p class='section-title'>72-Hour Forecast</p>", unsafe_allow_html=True)
    ffig = go.Figure()
    ffig.add_trace(go.Scatter(
        x=fc["timestamp"], y=fc["upper_bound"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    ffig.add_trace(go.Scatter(
        x=fc["timestamp"], y=fc["lower_bound"],
        name="Confidence", fill="tonexty",
        fillcolor="rgba(179,136,255,0.15)", line=dict(width=0), hoverinfo="skip",
    ))
    ffig.add_trace(go.Scatter(
        x=fc["timestamp"], y=fc["predicted_aqi"],
        name="Forecast", mode="lines+markers",
        line=dict(color="#b388ff", width=2.5),
        marker=dict(size=3),
    ))
    ffig.update_layout(
        **chart_dark,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(ffig, use_container_width=True)

# ── 3-Day summary cards ───────────────────────────────────────────────────────
st.markdown("<p class='section-title'>3-Day Outlook</p>", unsafe_allow_html=True)
daily = daily_summary(fc)
day_cols = st.columns(len(daily))
for i, (_, row) in enumerate(daily.iterrows()):
    lbl, clr = aqi_band(row["mean"])
    with day_cols[i]:
        st.markdown(
            f"<div class='aqi-card' style='text-align:center;border-left:4px solid {clr};'>"
            f"<div style='font-size:0.78rem;color:#7b8199;margin-bottom:4px'>{row['date']}</div>"
            f"<div style='font-size:2rem;font-weight:700;color:{clr}'>{row['mean']:.0f}</div>"
            f"<div style='font-size:0.72rem;color:#aab0c6'>{lbl}</div>"
            f"<div style='font-size:0.68rem;color:#5c6380;margin-top:4px'>"
            f"↓{row['min']:.0f} · ↑{row['max']:.0f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── Full forecast table ───────────────────────────────────────────────────────
st.markdown("<p class='section-title'>Full 72-Hour Forecast Table</p>", unsafe_allow_html=True)
table = fc[["timestamp", "predicted_aqi", "lower_bound", "upper_bound", "model_used"]].rename(
    columns={
        "timestamp": "Time", "predicted_aqi": "AQI",
        "lower_bound": "CI Low", "upper_bound": "CI High", "model_used": "Model",
    }
)
st.dataframe(table, use_container_width=True, height=380)

# ── Recent history expander ───────────────────────────────────────────────────
with st.expander("🕒 Recent Observed Readings (last 48h)"):
    h = hist[["timestamp", "aqi"]].tail(48).rename(
        columns={"timestamp": "Time", "aqi": "AQI"}
    )
    st.dataframe(h, use_container_width=True, height=280)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<hr style='border-color:#2e3148;margin:18px 0 8px 0'>"
    f"<p style='font-size:0.72rem;color:#3d4160;text-align:center'>"
    f"Model: {fc['model_used'].iloc[0]} · Cloud: Hopsworks · Cache TTL: 10 min · "
    f"Horizon: 72 h direct multi-output</p>",
    unsafe_allow_html=True,
)
