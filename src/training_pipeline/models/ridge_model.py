"""Ridge regression — statistical baseline. Multi-output via wrapper."""
from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge_model(alpha: float = 1.0, multi_output: bool = True):
    """Scaled Ridge. multi_output=True -> predicts the 72-h target vector."""
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=42)),
    ])
    return MultiOutputRegressor(base) if multi_output else base
