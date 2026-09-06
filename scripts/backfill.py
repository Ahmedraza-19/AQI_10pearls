"""Backfill historical (features, targets) to seed the feature store.

Runs the feature computation over a range of past dates -> training data.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.feature_pipeline.data_fetcher import DataFetcher  # noqa: E402
from src.feature_pipeline.feature_engineer import engineer_features  # noqa: E402
from src.feature_pipeline.feature_store import store_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("backfill")


def backfill(city: str, days: int) -> int:
    raw = DataFetcher().fetch_history(city, days)
    feats = engineer_features(raw)
    store_features(feats, city)
    log.info("Backfilled %d rows for %s (%d days).", len(feats), city, days)
    return len(feats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default=None, help="single city; default = all")
    p.add_argument("--days", type=int, default=90)
    args = p.parse_args()
    cities = [args.city] if args.city else config.CITY_NAMES
    total = 0
    for c in cities:
        if c not in config.CITIES:
            log.error("Unknown city: %s", c)
            continue
        total += backfill(c, args.days)
    log.info("Backfill complete: %d rows across %d cities.", total, len(cities))


if __name__ == "__main__":
    main()
