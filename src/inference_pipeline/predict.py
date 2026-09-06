"""Inference: one-shot 72-hour forecast from the latest feature row.

Multi-horizon model -> 72 distinct predicted values in a single forward pass.
No autoregressive frozen-feature rollout, so the curve genuinely varies.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src import config
from src.feature_pipeline.feature_engineer import engineer_features
from src.feature_pipeline.feature_store import fetch_features
from src.training_pipeline.model_registry import best_model_name, load_model

logger = logging.getLogger(__name__)


def _recent_engineered(city: str) -> pd.DataFrame:
    raw = fetch_features(city, limit=config.LOOKBACK_HOURS * 3)
    if raw is None or raw.empty:
        raise RuntimeError(f"No features for {city}. Run the feature pipeline/backfill first.")
    # stored rows already engineered; re-engineer defensively if raw-only
    if "hour_sin" not in raw.columns:
        raw = engineer_features(raw)
    return raw


def predict_current(city: str) -> dict:
    """Latest observed AQI for the city."""
    df = _recent_engineered(city)
    last = df.dropna(subset=["aqi"]).iloc[-1]
    return {
        "city": city,
        "timestamp": pd.to_datetime(last["timestamp"]).isoformat(),
        "aqi": round(float(last["aqi"]), 1),
    }


def predict_next_72h(city: str, model_name: str | None = None) -> pd.DataFrame:
    """Return a 72-row dataframe: timestamp, predicted_aqi, lower/upper band."""
    name = model_name or best_model_name(city)
    model, meta = load_model(city, name)
    feat_cols = meta["feature_columns"]

    df = _recent_engineered(city).dropna(subset=feat_cols)
    if df.empty:
        raise RuntimeError(f"No complete feature row for {city}.")

    base = pd.to_datetime(df["timestamp"].max())

    if meta.get("is_keras"):
        window = df[feat_cols].to_numpy(dtype="float32")[-config.LOOKBACK_HOURS:]
        if len(window) < config.LOOKBACK_HOURS:
            pad = np.repeat(window[:1], config.LOOKBACK_HOURS - len(window), axis=0)
            window = np.vstack([pad, window])
        preds = model.predict(window[np.newaxis, ...], verbose=0)[0]
    else:
        x_last = df[feat_cols].iloc[-1:].to_numpy(dtype="float32")
        preds = np.asarray(model.predict(x_last)[0], dtype=float)

    preds = np.clip(preds, 0, 500)
    out = pd.DataFrame({
        "timestamp": [base + pd.Timedelta(hours=h) for h in range(1, len(preds) + 1)],
        "predicted_aqi": np.round(preds, 1),
    })
    out["lower_bound"] = np.round((out["predicted_aqi"] * 0.90).clip(lower=0), 1)
    out["upper_bound"] = np.round(out["predicted_aqi"] * 1.10, 1)
    out["model_used"] = meta["model_name"]
    return out


def daily_summary(forecast: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 72h forecast into 3 daily min/mean/max rows."""
    f = forecast.copy()
    f["date"] = pd.to_datetime(f["timestamp"]).dt.date
    g = f.groupby("date")["predicted_aqi"].agg(["min", "mean", "max"]).round(1)
    return g.reset_index()


def _cli():
    p = argparse.ArgumentParser(description="3-day AQI forecast")
    p.add_argument("--city", default="london", choices=config.CITY_NAMES)
    p.add_argument("--model", default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fc = predict_next_72h(args.city, args.model)
    print(f"Current: {predict_current(args.city)}")
    print(fc.to_string(index=False))
    print("\nDaily summary:")
    print(daily_summary(fc).to_string(index=False))
    print(f"\nForecast std (non-flat check): {fc['predicted_aqi'].std():.2f}")


if __name__ == "__main__":
    _cli()
