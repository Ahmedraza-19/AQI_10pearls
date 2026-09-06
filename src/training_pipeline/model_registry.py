"""Versioned model registry.

Cloud-first: when HOPSWORKS_API_KEY is set, models push to the Hopsworks Model
Registry and load from it. Local versioned folders + latest.json are the dev/CI
fallback so inference never hard-fails.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib

from src import config
from src.utils.hopsworks_client import login

logger = logging.getLogger(__name__)


def _city_dir(city: str) -> Path:
    d = config.MODELS_DIR / city.replace(" ", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_model(model, city: str, model_name: str, metrics: dict, meta: dict) -> str:
    """Save a model version locally (and to Hopsworks if configured)."""
    version = time.strftime("%Y%m%d_%H%M%S")
    cdir = _city_dir(city)
    payload = {
        "model": model,
        "metrics": metrics,
        "feature_columns": meta.get("feature_columns", []),
        "target_columns": meta.get("target_columns", []),
        "model_name": model_name,
        "version": version,
        "is_keras": meta.get("is_keras", False),
    }
    path = cdir / f"{model_name}_{version}.joblib"

    if meta.get("is_keras"):
        kpath = cdir / f"{model_name}_{version}.keras"
        model.save(kpath)
        payload["model"] = None
        payload["keras_path"] = str(kpath)
    joblib.dump(payload, path)

    _update_latest(city, model_name, str(path), metrics)
    logger.info("Saved %s v%s for %s (rmse=%.3f).", model_name, version, city,
                metrics.get("rmse", float("nan")))

    if config.USE_HOPSWORKS:
        try:
            _push_to_hopsworks(path, city, model_name, metrics)
        except Exception as e:  # noqa: BLE001
            logger.warning("Hopsworks model push failed: %s", e)
    return version


def _update_latest(city: str, model_name: str, path: str, metrics: dict) -> None:
    lp = _city_dir(city) / "latest.json"
    latest = json.loads(lp.read_text()) if lp.exists() else {}
    latest[model_name] = {"path": path, "metrics": metrics}
    # track best (lowest rmse) across model families
    best = latest.get("_best")
    if best is None or metrics["rmse"] < best["metrics"]["rmse"]:
        latest["_best"] = {"model_name": model_name, "path": path, "metrics": metrics}
    lp.write_text(json.dumps(latest, indent=2))


def best_model_name(city: str) -> str | None:
    lp = _city_dir(city) / "latest.json"
    if not lp.exists():
        return None
    return json.loads(lp.read_text()).get("_best", {}).get("model_name")


def load_model(city: str, model_name: str | None = None) -> tuple[object, dict]:
    """Load a model + metadata. CLOUD ONLY when a Hopsworks key is set —
    no silent local fallback; raises if the cloud registry load fails."""
    if config.USE_HOPSWORKS:
        return _load_from_hopsworks(city, model_name)   # raises on failure

    lp = _city_dir(city) / "latest.json"
    if not lp.exists():
        raise FileNotFoundError(f"No registry entry for city={city}. Train first.")
    latest = json.loads(lp.read_text())

    if model_name is None:
        entry = latest.get("_best")
    else:
        entry = latest.get(model_name)
    if entry is None:
        raise FileNotFoundError(f"Model {model_name} not found for {city}.")

    payload = joblib.load(entry["path"])
    model = payload["model"]
    if payload.get("is_keras"):
        from tensorflow.keras.models import load_model as keras_load
        model = keras_load(payload["keras_path"])
    meta = {
        "feature_columns": payload["feature_columns"],
        "target_columns": payload["target_columns"],
        "model_name": payload["model_name"],
        "version": payload["version"],
        "is_keras": payload.get("is_keras", False),
        "metrics": payload["metrics"],
    }
    return model, meta


# ----------------------------------------------------------------------------
# Hopsworks Model Registry (active only when key set)
# ----------------------------------------------------------------------------


def _hopsworks_mr():
    return login().get_model_registry()


def _push_to_hopsworks(local_path: Path, city: str, model_name: str, metrics: dict) -> None:
    mr = _hopsworks_mr()
    name = f"aqi_{city.replace(' ', '_')}_{model_name}"
    model = mr.python.create_model(name=name, metrics=metrics,
                                   description=f"AQI 72h forecaster for {city}.")
    model.save(str(local_path))
    logger.info("Pushed %s to Hopsworks Model Registry.", name)


def _download_with_retry(model_obj, retries: int = 3):
    """Download model artifact; retry on dropped connection (144MB transfer)."""
    from requests.exceptions import ConnectionError as ReqConnErr
    last = None
    for attempt in range(1, retries + 1):
        try:
            return model_obj.download()
        except (ReqConnErr, ConnectionError, OSError) as e:
            last = e
            logger.warning("Model download attempt %d/%d failed: %s",
                           attempt, retries, e)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Model download failed after {retries} attempts: {last}")


def _load_from_hopsworks(city: str, model_name: str | None):
    mr = _hopsworks_mr()
    name = f"aqi_{city.replace(' ', '_')}_{model_name or 'random_forest'}"
    m = None
    if model_name is None:
        try:
            m = mr.get_best_model(name, "rmse", "min")
        except Exception:  # noqa: BLE001
            m = None
    if m is None:
        # latest version: get_model(name) may return None without a version,
        # so enumerate all versions and pick the highest.
        models = mr.get_models(name)
        if not models:
            raise FileNotFoundError(f"No cloud model named {name}.")
        m = max(models, key=lambda x: x.version)

    # Disk cache: skip the 144MB download if this exact version already pulled.
    cache_dir = config.MODELS_DIR / "_hopsworks_cache" / f"{name}_v{m.version}"
    cached = list(cache_dir.glob("*.joblib")) if cache_dir.exists() else []
    if cached:
        files = cached
    else:
        d = _download_with_retry(m)
        files = list(Path(d).glob("*.joblib"))
        if files:
            cache_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            for f in files:
                shutil.copy2(f, cache_dir / f.name)
            files = list(cache_dir.glob("*.joblib"))
    if not files:
        raise FileNotFoundError("No joblib payload in downloaded Hopsworks model.")
    payload = joblib.load(files[0])
    meta = {
        "feature_columns": payload["feature_columns"],
        "target_columns": payload["target_columns"],
        "model_name": payload["model_name"],
        "version": payload["version"],
        "is_keras": payload.get("is_keras", False),
        "metrics": payload["metrics"],
    }
    return payload["model"], meta
