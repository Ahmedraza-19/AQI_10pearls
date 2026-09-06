"""One-shot full pipeline for ALL cities: backfill -> push features to cloud
-> train -> push models to cloud registry. Trains on in-memory features to
avoid Hopsworks read-after-write materialization lag.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.feature_pipeline.data_fetcher import DataFetcher  # noqa: E402
from src.feature_pipeline.feature_engineer import engineer_features  # noqa: E402
from src.feature_pipeline.feature_store import store_features  # noqa: E402
from src.training_pipeline.train import train_city  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("full_run")

DAYS = 90


def main():
    fetcher = DataFetcher()
    log.info("Hopsworks cloud: %s (project=%s)", config.USE_HOPSWORKS, config.HOPSWORKS_PROJECT)
    summary = {}
    for city in config.CITY_NAMES:
        log.info("==================== %s ====================", city)
        try:
            raw = fetcher.fetch_history(city, DAYS)
            feats = engineer_features(raw)
            store_features(feats, city)              # -> cloud feature group + local
            res = train_city(city, feats)            # -> cloud model registry + local
            summary[city] = {m: r["metrics"]["rmse"] for m, r in res.items()} if res else {}
        except Exception as e:  # noqa: BLE001
            log.error("City %s failed: %s", city, e)
            summary[city] = {"ERROR": str(e)[:80]}
    log.info("==================== SUMMARY (rmse) ====================")
    for c, m in summary.items():
        log.info("%-10s %s", c, m)


if __name__ == "__main__":
    main()
