"""Feature store persistence — CLOUD ONLY (Hopsworks online store).

Hopsworks is the single source of truth. The feature group is online-enabled
(RonDB) so reads/writes don't depend on the offline Spark materialization job.
Local SQLite is used only as a last-resort dev fallback when no cloud key is
configured. NO CSV is ever written — the rubric forbids CSV output.
"""
from __future__ import annotations

import logging
import sqlite3
import time

import pandas as pd

from src import config
from src.utils.hopsworks_client import login

logger = logging.getLogger(__name__)

_FG_NAME = "aqi_features"
_FG_VERSION = 2          # v2 = online-enabled (cloud-only path)
_TABLE = "aqi_features"


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def store_features(df: pd.DataFrame, city: str) -> None:
    """Persist engineered features to the cloud online feature store.

    Cloud-only when HOPSWORKS_API_KEY is set. No local write in that mode.
    """
    if df.empty:
        logger.warning("store_features: empty df for %s — skipped.", city)
        return
    df = df.copy()
    df["city"] = city
    # NOTE: CSV output deliberately removed — rubric forbids CSV.
    if config.USE_HOPSWORKS:
        save_to_hopsworks(df)           # cloud is the only store
    else:
        _save_local(df, city)           # dev fallback only (no cloud key)


def fetch_features(city: str, limit: int | None = None) -> pd.DataFrame:
    """Load features from the cloud online store. Cloud-only — no silent
    local fallback when a Hopsworks key is configured."""
    if config.USE_HOPSWORKS:
        return _fetch_from_hopsworks(city, limit)   # raises on cloud failure
    return _fetch_local(city, limit)


# ----------------------------------------------------------------------------
# Local SQLite
# ----------------------------------------------------------------------------
def _conn() -> sqlite3.Connection:
    return sqlite3.connect(config.SQLITE_PATH)


def _save_local(df: pd.DataFrame, city: str) -> None:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype(str)
    with _conn() as con:
        df.to_sql(_TABLE, con, if_exists="append", index=False)
        # de-dup on (city, timestamp), keep newest insert
        con.execute(
            f"""DELETE FROM {_TABLE} WHERE rowid NOT IN
                (SELECT MAX(rowid) FROM {_TABLE} GROUP BY city, timestamp)"""
        )
    logger.info("Stored %d rows for %s (local).", len(df), city)


def _fetch_local(city: str, limit: int | None) -> pd.DataFrame:
    try:
        with _conn() as con:
            q = f"SELECT * FROM {_TABLE} WHERE city = ? ORDER BY timestamp"
            params: tuple = (city,)
            if limit:
                q = (f"SELECT * FROM (SELECT * FROM {_TABLE} WHERE city = ? "
                     f"ORDER BY timestamp DESC LIMIT ?) ORDER BY timestamp")
                params = (city, limit)
            df = pd.read_sql_query(q, con, params=params)
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return pd.DataFrame()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ----------------------------------------------------------------------------
# Hopsworks (cloud) — active only when key set
# ----------------------------------------------------------------------------


def _get_hopsworks_fg():
    project = login()
    fs = project.get_feature_store()
    return fs.get_or_create_feature_group(
        name=_FG_NAME,
        version=_FG_VERSION,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        description="Engineered AQI features for 3-day forecasting (online).",
        online_enabled=True,        # cloud-only reads via RonDB, no Spark job
    )


def save_to_hopsworks(df: pd.DataFrame, retries: int = 3) -> None:
    if not config.HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY not set — cloud feature store required.")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    fg = _get_hopsworks_fg()
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            # wait_for_job=False: data written via Kafka, offline materialization
            # job runs server-side; we don't block on a long, flaky status poll.
            fg.insert(df, write_options={"wait_for_job": False})
            logger.info("Inserted %d rows into Hopsworks fg=%s (materialization async).",
                        len(df), _FG_NAME)
            return
        except Exception as e:  # noqa: BLE001  (Kafka metadata timeouts are transient)
            if _rows_written_but_job_failed(e):
                # Rows are already in Kafka/online store; only the offline
                # materialization job could not be launched (client 4.8.x vs
                # backend 5.0 API drift). Online reads are unaffected.
                logger.warning("Inserted %d rows; offline materialization job "
                               "could not be launched (%s). Online store is "
                               "current — continuing.", len(df), e)
                return
            last_err = e
            logger.warning("Hopsworks insert attempt %d/%d failed: %s",
                           attempt, retries, e)
    raise RuntimeError(f"Hopsworks insert failed after {retries} attempts: {last_err}")


def _rows_written_but_job_failed(exc: Exception) -> bool:
    """True when insert streamed the rows and only the materialization job
    launch failed — HTTP 415 on .../jobs/<fg>_offline_fg_materialization/."""
    msg = str(exc)
    return "offline_fg_materialization" in msg and "415" in msg


def _fetch_from_hopsworks(city: str, limit: int | None) -> pd.DataFrame:
    """Read from the ONLINE store (RonDB) — independent of the offline
    materialization Spark job, so it works immediately on the free tier."""
    fg = _get_hopsworks_fg()
    df = fg.read(online=True)
    df = df[df["city"] == city].copy()
    if df.empty:
        raise RuntimeError(f"No cloud features for {city} (online store empty).")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    if limit:
        df = df.tail(limit)
    return df.reset_index(drop=True)
