"""Feature engineering + multi-horizon target tests. No API keys needed."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src import config
from src.feature_pipeline.data_fetcher import RAW_COLUMNS, DataFetcher
from src.feature_pipeline.feature_engineer import (
    create_forecast_targets,
    engineer_features,
    get_feature_columns,
)


@pytest.fixture
def raw_df():
    return DataFetcher().fetch_history("london", days=10)   # 240 rows, synthetic


def test_fetch_history_schema(raw_df):
    assert list(raw_df.columns) == RAW_COLUMNS
    assert len(raw_df) == 240
    assert raw_df["aqi"].between(0, 500).all()


def test_time_features(raw_df):
    df = engineer_features(raw_df)
    for col in ["hour", "day", "month", "dayofweek", "is_weekend",
                "hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert col in df.columns
    assert df["hour"].between(0, 23).all()
    assert set(df["is_weekend"].unique()).issubset({0, 1})


def test_lag_and_rolling_and_changerate(raw_df):
    df = engineer_features(raw_df)
    for lag in config.LAG_HOURS:
        assert f"aqi_lag_{lag}" in df.columns
    assert "aqi_roll_mean_6" in df.columns
    assert "aqi_change_rate" in df.columns
    # lag_1 equals previous aqi
    assert np.isclose(df["aqi_lag_1"].iloc[5], df["aqi"].iloc[4])


def test_no_future_leakage_in_features(raw_df):
    """Features must use only past info: lag_1 at row i = aqi at row i-1."""
    df = engineer_features(raw_df)
    assert np.isclose(df["aqi_lag_1"].iloc[10], df["aqi"].iloc[9])


def test_create_forecast_targets(raw_df):
    df = create_forecast_targets(raw_df, horizons=[1, 24, 72])
    assert {"aqi_t+1h", "aqi_t+24h", "aqi_t+72h"}.issubset(df.columns)
    # target t+1h at row i = aqi at row i+1
    assert np.isclose(df["aqi_t+1h"].iloc[0], df["aqi"].iloc[1])
    # last rows have NaN future targets
    assert pd.isna(df["aqi_t+72h"].iloc[-1])


def test_get_feature_columns_excludes_targets(raw_df):
    df = engineer_features(raw_df)
    df = create_forecast_targets(df, horizons=[1, 2])
    cols = get_feature_columns(df)
    assert "aqi" not in cols
    assert "timestamp" not in cols
    assert not any(c.startswith("aqi_t+") for c in cols)
    assert "hour_sin" in cols
