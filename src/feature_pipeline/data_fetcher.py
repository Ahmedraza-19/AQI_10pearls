"""API clients for raw AQI + weather data, with graceful synthetic fallback.

If a key is missing the corresponding fetcher returns physically-plausible
synthetic data so the whole pipeline (and CI) runs offline. Real keys swap in
live data with no code change.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)

RAW_COLUMNS = [
    "timestamp", "city", "aqi",
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temp", "humidity", "pressure", "wind_speed",
]


# ----------------------------------------------------------------------------
# Synthetic generator (offline / backfill / CI)
# ----------------------------------------------------------------------------
def _synthetic_series(city: str, start: datetime, hours: int) -> pd.DataFrame:
    """Deterministic-per-city hourly series with daily + weekly seasonality."""
    seed = abs(hash(city)) % (2**32)
    rng = np.random.default_rng(seed + start.toordinal())

    base = {  # rough mean AQI per city
        "london": 45, "new york": 55, "beijing": 120, "delhi": 165,
        "paris": 60, "karachi": 140, "lahore": 175,
    }.get(city, 80)

    ts = [start + timedelta(hours=h) for h in range(hours)]
    t = np.arange(hours)
    daily = 18 * np.sin(2 * np.pi * (t % 24) / 24 - np.pi / 2)   # peak afternoon
    weekly = 8 * np.sin(2 * np.pi * (t % 168) / 168)
    drift = np.cumsum(rng.normal(0, 1.2, hours))
    aqi = np.clip(base + daily + weekly + drift + rng.normal(0, 6, hours), 5, 480)

    pm25 = np.clip(aqi * 0.45 + rng.normal(0, 4, hours), 1, 400)
    pm10 = np.clip(pm25 * 1.7 + rng.normal(0, 6, hours), 1, 600)
    return pd.DataFrame({
        "timestamp": ts,
        "city": city,
        "aqi": np.round(aqi, 1),
        "pm25": np.round(pm25, 1),
        "pm10": np.round(pm10, 1),
        "o3":  np.round(np.clip(rng.normal(40, 12, hours), 0, 200), 1),
        "no2": np.round(np.clip(rng.normal(30, 10, hours), 0, 200), 1),
        "so2": np.round(np.clip(rng.normal(12, 5, hours), 0, 150), 1),
        "co":  np.round(np.clip(rng.normal(0.8, 0.3, hours), 0, 10), 2),
        "temp": np.round(15 + 10 * np.sin(2 * np.pi * (t % 24) / 24) + rng.normal(0, 2, hours), 1),
        "humidity": np.round(np.clip(rng.normal(60, 15, hours), 5, 100), 0),
        "pressure": np.round(np.clip(rng.normal(1013, 8, hours), 970, 1050), 0),
        "wind_speed": np.round(np.clip(rng.normal(3.5, 1.8, hours), 0, 25), 1),
    })


# ----------------------------------------------------------------------------
# AQICN — real-time AQI + pollutants
# ----------------------------------------------------------------------------
class AQICNFetcher:
    BASE = "https://api.waqi.info/feed"

    def __init__(self, token: str | None = None):
        self.token = token if token is not None else config.AQICN_API_TOKEN

    @property
    def available(self) -> bool:
        return bool(self.token)

    def fetch_current(self, city: str) -> dict | None:
        if not self.available:
            return None
        slug = config.CITIES[city]["aqicn"]
        try:
            r = requests.get(f"{self.BASE}/{slug}/", params={"token": self.token}, timeout=15)
            r.raise_for_status()
            data = r.json()
            if data.get("status") != "ok":
                logger.warning("AQICN status not ok for %s: %s", city, data.get("data"))
                return None
            d = data["data"]
            iaqi = d.get("iaqi", {})
            g = lambda k: iaqi.get(k, {}).get("v")  # noqa: E731
            return {
                "aqi": d.get("aqi"),
                "pm25": g("pm25"), "pm10": g("pm10"), "o3": g("o3"),
                "no2": g("no2"), "so2": g("so2"), "co": g("co"),
            }
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.warning("AQICN fetch failed for %s: %s", city, e)
            return None


# ----------------------------------------------------------------------------
# OpenWeather — weather + air pollution
# ----------------------------------------------------------------------------
class OpenWeatherFetcher:
    WEATHER = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, key: str | None = None):
        self.key = key if key is not None else config.OPENWEATHER_API_KEY

    @property
    def available(self) -> bool:
        return bool(self.key)

    def fetch_current(self, city: str) -> dict | None:
        if not self.available:
            return None
        c = config.CITIES[city]
        try:
            r = requests.get(self.WEATHER, params={
                "lat": c["lat"], "lon": c["lon"], "appid": self.key, "units": "metric",
            }, timeout=15)
            r.raise_for_status()
            d = r.json()
            return {
                "temp": d["main"].get("temp"),
                "humidity": d["main"].get("humidity"),
                "pressure": d["main"].get("pressure"),
                "wind_speed": d.get("wind", {}).get("speed"),
            }
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.warning("OpenWeather fetch failed for %s: %s", city, e)
            return None


# ----------------------------------------------------------------------------
# Unified fetcher
# ----------------------------------------------------------------------------
class DataFetcher:
    def __init__(self):
        self.aqicn = AQICNFetcher()
        self.weather = OpenWeatherFetcher()

    def fetch_current(self, city: str) -> pd.DataFrame:
        """One row of the latest observation for a city (live or synthetic)."""
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        row = _synthetic_series(city, now, 1).iloc[0].to_dict()  # baseline

        aqi_part = self.aqicn.fetch_current(city)
        wx_part = self.weather.fetch_current(city)
        if aqi_part:
            row.update({k: v for k, v in aqi_part.items() if v is not None})
        if wx_part:
            row.update({k: v for k, v in wx_part.items() if v is not None})
        if not aqi_part and not wx_part:
            logger.info("No API keys/live data for %s — using synthetic.", city)

        row["timestamp"] = now
        row["city"] = city
        return pd.DataFrame([row])[RAW_COLUMNS]

    def fetch_history(self, city: str, days: int) -> pd.DataFrame:
        """Hourly history for backfill. Live history APIs are paid, so we
        synthesise a plausible series ending now (used to train models)."""
        hours = days * 24
        start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours - 1)
        df = _synthetic_series(city, start, hours)
        return df[RAW_COLUMNS]
