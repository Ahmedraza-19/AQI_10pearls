"""Daily training pipeline: fetch features -> train+eval -> registry.

If a city lacks enough data it is skipped with a warning (exit 0) so the
scheduled CI run stays green.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.feature_pipeline.feature_store import fetch_features  # noqa: E402
from src.training_pipeline.train import train_city  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("training_pipeline")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default=None, help="single city; default = all")
    p.add_argument("--days", type=int, default=90,
                   help="auto-backfill this many days if store is empty")
    args = p.parse_args()
    cities = [args.city] if args.city else config.CITY_NAMES
    trained: list[str] = []
    failed: list[str] = []

    for c in cities:
        if c not in config.CITIES:
            log.error("Unknown city: %s", c)
            continue
        try:
            try:
                df = fetch_features(c)
            except Exception as e:  # noqa: BLE001 — empty store is not fatal
                log.info("No cloud features for %s yet (%s).", c, e)
                df = None
            if df is None or len(df) < config.MIN_TRAIN_ROWS:
                log.warning("Store thin for %s (%s rows); seeding %d days.",
                            c, 0 if df is None else len(df), args.days)
                from scripts.backfill import backfill
                backfill(c, args.days)
                df = fetch_features(c)
            res = train_city(c, df)
            if not res:
                log.warning("No model trained for %s (insufficient data).", c)
            else:
                trained.append(c)
        except Exception as e:  # noqa: BLE001 — a data gap must not crash CI
            log.warning("Training failed for %s: %s", c, e)
            failed.append(c)

    if failed and not trained:
        # Every city failed for a non-data reason (auth, project, cloud API).
        # Exiting 0 here would paint a broken scheduled run green forever.
        log.error("No models trained; failures: %s", ", ".join(failed))
        sys.exit(1)
    if failed:
        log.warning("Partial run — trained: %s; failed: %s",
                    ", ".join(trained), ", ".join(failed))
    sys.exit(0)


if __name__ == "__main__":
    main()
