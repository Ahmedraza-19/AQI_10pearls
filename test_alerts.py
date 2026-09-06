"""AQI classification + hazardous alert tests."""
import pandas as pd

from src import config
from src.utils.alerts import (
    check_alerts,
    classify_aqi,
    get_color,
    get_emoji,
    is_hazardous,
)


def test_classify_boundaries():
    assert classify_aqi(0)["level"] == "Good"
    assert classify_aqi(50)["level"] == "Good"
    assert classify_aqi(51)["level"] == "Moderate"
    assert classify_aqi(150)["level"] == "Unhealthy for Sensitive Groups"
    assert classify_aqi(151)["level"] == "Unhealthy"
    assert classify_aqi(301)["level"] == "Hazardous"
    assert classify_aqi(600)["level"] == "Hazardous"  # clamp above 500


def test_color_and_emoji():
    assert get_color(25).startswith("#")
    assert isinstance(get_emoji(25), str) and get_emoji(25)


def test_is_hazardous():
    assert not is_hazardous(config.ALERT_THRESHOLD - 1)
    assert is_hazardous(config.ALERT_THRESHOLD)
    assert is_hazardous(400)


def test_check_alerts_filters_hazardous():
    fc = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
        "predicted_aqi": [40, 160, 90, 320],
    })
    alerts = check_alerts(fc)
    assert len(alerts) == 2
    assert all(a["aqi"] >= config.ALERT_THRESHOLD for a in alerts)
    assert "message" in alerts[0]
