"""Feature computation: time/cyclical, lags, rolling stats, change-rate/trend,
pollutant & weather interactions, and multi-horizon forecast targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# Columns that are never model inputs.
_NON_FEATURE = {"timestamp", "city", "aqi"}

RAW_NUMERIC = ["pm25", "pm10", "o3", "no2", "so2", "co",
               "temp", "humidity", "pressure", "wind_speed"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time, cyclical, lag, rolling, change-rate and interaction features.

    Input: raw rows (one city) with at least timestamp + aqi + RAW_NUMERIC.
    Output: same rows + feature columns (leading rows hold NaN lags).
    """
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- time-based ---
    ts = df["timestamp"].dt
    df["hour"] = ts.hour
    df["day"] = ts.day
    df["month"] = ts.month
    df["dayofweek"] = ts.dayofweek
    df["is_weekend"] = (ts.dayofweek >= 5).astype(int)

    # --- cyclical encodings ---
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- lags ---
    for lag in config.LAG_HOURS:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    # --- rolling stats ---
    for w in config.ROLLING_WINDOWS:
        df[f"aqi_roll_mean_{w}"] = df["aqi"].shift(1).rolling(w, min_periods=1).mean()
        df[f"aqi_roll_std_{w}"] = df["aqi"].shift(1).rolling(w, min_periods=1).std()
    df[f"aqi_roll_max_{config.ROLLING_WINDOWS[-1]}"] = (
        df["aqi"].shift(1).rolling(config.ROLLING_WINDOWS[-1], min_periods=1).max()
    )

    # --- derived: change rate + short trend (no future leakage) ---
    df["aqi_change_rate"] = df["aqi"].shift(1).diff()
    df["aqi_trend_3h"] = df["aqi"].shift(1) - df["aqi"].shift(4)

    # --- interactions ---
    df["pm25_pm10_ratio"] = df["pm25"] / (df["pm10"] + 1e-6)
    df["temp_humidity"] = df["temp"] * df["humidity"] / 100.0
    df["wind_dispersion"] = df["pm25"] / (df["wind_speed"] + 1.0)

    df["aqi_roll_std_6"] = df["aqi_roll_std_6"].fillna(0.0)
    return df


def create_forecast_targets(df: pd.DataFrame, horizons=None) -> pd.DataFrame:
    """Add aqi_t+{h}h columns for direct multi-horizon supervised learning.

    Target at horizon h = AQI shifted -h (future value). Trailing rows where the
    future is unknown become NaN and are dropped at training time.
    """
    if horizons is None:
        horizons = list(range(1, config.FORECAST_HOURS + 1))
    df = df.copy()
    for h in horizons:
        df[f"aqi_t+{h}h"] = df["aqi"].shift(-h)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Model-input columns: numeric, not raw identifiers, not any target."""
    cols = []
    for c in df.columns:
        if c in _NON_FEATURE or c.startswith("aqi_t+"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols
