"""Evaluation metric tests."""
import numpy as np

from src.training_pipeline.evaluate import (
    compare_models,
    evaluate_model,
    mae,
    mape,
    r2,
    rmse,
)


def test_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert r2(y, y) == 1.0
    assert mape(y, y) == 0.0


def test_known_values():
    y = np.array([0.0, 0.0, 0.0])
    p = np.array([1.0, 1.0, 1.0])
    assert np.isclose(rmse(y, p), 1.0)
    assert np.isclose(mae(y, p), 1.0)


def test_evaluate_model_keys():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.1, 1.9, 3.2, 3.8])
    m = evaluate_model(y, p)
    assert set(m) == {"rmse", "mae", "r2", "mape"}
    assert m["rmse"] >= 0
    assert m["r2"] <= 1.0


def test_compare_models_picks_lowest_rmse():
    results = {
        "ridge": {"metrics": {"rmse": 12.0}},
        "random_forest": {"metrics": {"rmse": 8.5}},
        "lstm": {"metrics": {"rmse": 9.9}},
    }
    assert compare_models(results) == "random_forest"


def test_multioutput_metrics():
    """Metrics work on (n, 72) target matrices."""
    rng = np.random.default_rng(0)
    y = rng.normal(80, 20, (50, 72))
    p = y + rng.normal(0, 5, (50, 72))
    m = evaluate_model(y, p)
    assert m["rmse"] > 0 and m["r2"] < 1.0
