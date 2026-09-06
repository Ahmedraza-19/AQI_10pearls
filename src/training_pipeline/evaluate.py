"""Regression metrics + model comparison."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_model(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


def compare_models(results: dict[str, dict]) -> str:
    """Pick best model name by lowest RMSE."""
    return min(results, key=lambda m: results[m]["metrics"]["rmse"])
