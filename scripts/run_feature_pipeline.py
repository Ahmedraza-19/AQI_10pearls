"""Hourly feature pipeline: fetch latest -> engineer -> store. No CSV.

Exits 0 even on transient data gaps so scheduled CI runs stay green.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.feature_pipeline.data_fetcher import RAW_COLUMNS, DataFetcher  # noqa: E402
from src.feature_pipeline.feature_engineer import engineer_features  # noqa: E402
from src.feature_pipeline.feature_store import fetch_features, store_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("feature_pipeline")


def run(city: str) -> bool:
    try:
        latest = DataFetcher().fetch_current(city)

        # Prepend recent raw history so lag/rolling features compute correctly.
        # First-ever run has no cloud history yet — that's fine, not an error.
        try:
            hist = fetch_features(city, limit=config.LOOKBACK_HOURS * 2)
        except Exception as e:  # noqa: BLE001
            log.info("No cloud history for %s yet (%s) — seeding from latest.", city, e)
            hist = None

        frames = [latest]
        if hist is not None and not hist.empty:
            raw_hist = hist[[c for c in RAW_COLUMNS if c in hist.columns]]
            frames = [raw_hist, latest]
        combined = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["timestamp"], keep="last")

        feats = engineer_features(combined).tail(1)
        store_features(feats, city)
        log.info("Stored features for %s.", city)
        return True
    except Exception as e:  # noqa: BLE001 — never crash a scheduled run
        log.warning("Pipeline failed for %s: %s — skipping (exit 0).", city, e)
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default=None, help="single city; default = all")
    args = p.parse_args()
    cities = [args.city] if args.city else config.CITY_NAMES
    ok = 0
    for c in cities:
        if c not in config.CITIES:
            log.error("Unknown city: %s", c)
            continue
        ok += run(c)
    log.info("Feature pipeline done: %d/%d cities updated.", ok, len(cities))
    sys.exit(0)   # always green; data gaps are warnings, not failures


if __name__ == "__main__":
    main()
