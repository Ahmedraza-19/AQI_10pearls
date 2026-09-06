"""SHAP feature-importance explanations. Graceful if shap/model unsupported."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def explain_model(model, X: np.ndarray, feature_names: list[str],
                  max_samples: int = 100) -> pd.DataFrame | None:
    """Mean |SHAP| per feature, ranked. Returns None if SHAP can't run.

    For multi-output models, importance is averaged across output horizons.
    """
    try:
        import shap
    except Exception as e:  # noqa: BLE001
        logger.warning("shap unavailable: %s", e)
        return None

    X = np.asarray(X)[:max_samples]
    try:
        try:
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X)
        except Exception:
            explainer = shap.Explainer(model.predict, X)
            vals = explainer(X).values
    except Exception as e:  # noqa: BLE001
        logger.warning("SHAP explanation failed: %s", e)
        return None

    arr = np.asarray(vals)
    # collapse multi-output / multi-sample dims down to per-feature
    while arr.ndim > 2:
        arr = np.mean(np.abs(arr), axis=0)
    imp = np.mean(np.abs(arr), axis=0)
    imp = np.asarray(imp).ravel()[:len(feature_names)]

    return (pd.DataFrame({"feature": feature_names[:len(imp)], "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))
