"""Training pipeline: fetch features -> multi-horizon targets -> CV-train
Ridge / RandomForest / LSTM -> evaluate (RMSE/MAE/R²/MAPE) -> registry.

The forecast is REAL: models predict aqi_t+1h … aqi_t+72h directly (no
autoregressive frozen-feature rollout), so the 72-h curve genuinely varies.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src import config
from src.feature_pipeline.feature_engineer import (
    create_forecast_targets,
    engineer_features,
    get_feature_columns,
)
from src.feature_pipeline.feature_store import fetch_features
from src.training_pipeline import evaluate as ev
from src.training_pipeline.model_registry import save_model
from src.training_pipeline.models.lstm_model import (
    build_multistep_model,
    create_multistep_sequences,
    tensorflow_available,
)
from src.training_pipeline.models.random_forest_model import build_random_forest_model
from src.training_pipeline.models.ridge_model import build_ridge_model

logger = logging.getLogger(__name__)

HORIZONS = list(range(1, config.FORECAST_HOURS + 1))   # t+1h … t+72h
TARGET_COLS = [f"aqi_t+{h}h" for h in HORIZONS]


def prepare_data(df: pd.DataFrame):
    """Engineer features + multi-horizon targets; return X, Y, feature_cols."""
    df = engineer_features(df)
    df = create_forecast_targets(df, horizons=HORIZONS)
    feature_cols = [c for c in get_feature_columns(df) if c not in TARGET_COLS]
    df = df.dropna(subset=feature_cols + TARGET_COLS).reset_index(drop=True)
    X = df[feature_cols].to_numpy(dtype="float32")
    Y = df[TARGET_COLS].to_numpy(dtype="float32")
    return X, Y, feature_cols, df


def _cv_score(model_builder, X, Y, n_splits: int = 5) -> dict:
    """TimeSeriesSplit CV averaging metrics over folds."""
    n_splits = max(2, min(n_splits, len(X) // 50)) if len(X) > 100 else 2
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for tr, te in tscv.split(X):
        m = model_builder()
        m.fit(X[tr], Y[tr])
        folds.append(ev.evaluate_model(Y[te], m.predict(X[te])))
    return {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}


def train_sklearn(name: str, builder, X, Y, feature_cols, city: str) -> dict:
    metrics = _cv_score(builder, X, Y)
    model = builder()
    model.fit(X, Y)                      # final fit on all data
    save_model(model, city, name, metrics,
               {"feature_columns": feature_cols, "target_columns": TARGET_COLS})
    logger.info("[%s] %s rmse=%.2f mae=%.2f r2=%.3f", city, name,
                metrics["rmse"], metrics["mae"], metrics["r2"])
    return {"metrics": metrics}


def train_lstm(X, Y, feature_cols, full_df, city: str) -> dict | None:
    if not tensorflow_available():
        logger.warning("TensorFlow unavailable — skipping LSTM.")
        return None
    fm = full_df[feature_cols].to_numpy(dtype="float32")
    aqi = full_df["aqi"].to_numpy(dtype="float32")
    Xs, Ys = create_multistep_sequences(fm, aqi)
    if len(Xs) < 30:
        logger.warning("Not enough sequences for LSTM (%d).", len(Xs))
        return None
    split = int(len(Xs) * 0.8)
    model = build_multistep_model(n_features=fm.shape[1])
    model.fit(Xs[:split], Ys[:split], validation_split=0.1,
              epochs=15, batch_size=32, verbose=0)
    metrics = ev.evaluate_model(Ys[split:], model.predict(Xs[split:], verbose=0))
    save_model(model, city, "lstm", metrics,
               {"feature_columns": feature_cols, "target_columns": TARGET_COLS,
                "is_keras": True})
    logger.info("[%s] lstm rmse=%.2f mae=%.2f r2=%.3f", city,
                metrics["rmse"], metrics["mae"], metrics["r2"])
    return {"metrics": metrics}


def train_city(city: str, df: pd.DataFrame | None = None) -> dict:
    """Train all models for one city. Returns per-model metrics."""
    if df is None:
        df = fetch_features(city)
    if df is None or len(df) < config.MIN_TRAIN_ROWS:
        logger.warning("Not enough feature data for %s (have %s, need %d) — skipping.",
                       city, 0 if df is None else len(df), config.MIN_TRAIN_ROWS)
        return {}

    X, Y, feature_cols, full_df = prepare_data(df)
    if len(X) < config.MIN_TRAIN_ROWS // 2:
        logger.warning("Too few rows after target alignment for %s (%d).", city, len(X))
        return {}

    results: dict[str, dict] = {}
    results["ridge"] = train_sklearn(
        "ridge", lambda: build_ridge_model(alpha=1.0), X, Y, feature_cols, city)
    results["random_forest"] = train_sklearn(
        "random_forest", build_random_forest_model, X, Y, feature_cols, city)
    lstm = train_lstm(X, Y, feature_cols, full_df, city)
    if lstm:
        results["lstm"] = lstm

    best = ev.compare_models(results)
    logger.info("[%s] best model: %s", city, best)
    return results


def main(cities=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cities = cities or config.CITY_NAMES
    summary = {}
    for c in cities:
        summary[c] = train_city(c)
    return summary


if __name__ == "__main__":
    main()
