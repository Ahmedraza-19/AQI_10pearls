"""LSTM sequence-to-vector — deep learning model. TensorFlow optional.

Input window LOOKBACK_HOURS -> output vector FORECAST_HOURS (Dense(72)).
Import is lazy so the rest of the system runs without TensorFlow installed.
"""
from __future__ import annotations

import numpy as np

from src import config


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def create_multistep_sequences(
    feature_matrix: np.ndarray,
    aqi_series: np.ndarray,
    lookback: int = config.LOOKBACK_HOURS,
    horizon: int = config.FORECAST_HOURS,
):
    """Sliding windows: X=(n, lookback, n_feat), Y=(n, horizon)."""
    X, Y = [], []
    n = len(feature_matrix)
    for i in range(lookback, n - horizon + 1):
        X.append(feature_matrix[i - lookback:i])
        Y.append(aqi_series[i:i + horizon])
    if not X:
        return np.empty((0, lookback, feature_matrix.shape[1])), np.empty((0, horizon))
    return np.asarray(X, dtype="float32"), np.asarray(Y, dtype="float32")


def build_multistep_model(n_features: int,
                          lookback: int = config.LOOKBACK_HOURS,
                          horizon: int = config.FORECAST_HOURS):
    """LSTM -> Dense(horizon). Requires TensorFlow."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    model = Sequential([
        Input(shape=(lookback, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(horizon),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
