"""Shared Hopsworks login.

One cached login per process, used by both the feature store and the model
registry. Handles two free-tier realities:

* the serving API drops connections during login's default-config probe, so
  logins are retried;
* Hopsworks project names are case-sensitive, and a project created as
  "Anusha" is not reachable as "anusha". When the configured name is not
  found we fall back to the account's default project and accept it if the
  name matches case-insensitively.
"""
from __future__ import annotations

import logging
import os
import time

from src import config

logger = logging.getLogger(__name__)

_PROJECT = None  # module-level cache: login once per process


def _not_found(exc: Exception) -> bool:
    return "could not find project" in str(exc).lower()


def login(retries: int = 4):
    global _PROJECT
    if _PROJECT is not None:
        return _PROJECT

    import hopsworks
    from requests.exceptions import ConnectionError as ReqConnErr

    wanted = config.HOPSWORKS_PROJECT or None
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _PROJECT = hopsworks.login(
                api_key_value=config.HOPSWORKS_API_KEY, project=wanted)
            return _PROJECT
        except (ReqConnErr, ConnectionError, OSError) as e:
            last = e
            logger.warning("Hopsworks login attempt %d/%d failed: %s",
                           attempt, retries, e)
            time.sleep(2 * attempt)
        except Exception as e:  # noqa: BLE001
            if not (wanted and _not_found(e)):
                raise
            # Wrong case in HOPSWORKS_PROJECT — resolve via the default project.
            # hopsworks.login() re-reads the HOPSWORKS_PROJECT env var when
            # project is None (hopsworks/__init__.py), and load_dotenv has
            # already put the bad name there, so drop it for this call.
            last = e
            # The failed login leaves a half-initialised client cached in
            # hsml/hsfs — reusing it raises "'Client' object has no attribute
            # '_project_id'" later, at model-download time. Reset it first.
            try:
                hopsworks.logout()
            except Exception:  # noqa: BLE001 — nothing to tear down
                pass
            saved = os.environ.pop("HOPSWORKS_PROJECT", None)
            try:
                proj = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY)
            finally:
                if saved is not None:
                    os.environ["HOPSWORKS_PROJECT"] = saved
            if proj.name.lower() != wanted.lower():
                raise RuntimeError(
                    f"HOPSWORKS_PROJECT={wanted!r} not found; this API key "
                    f"belongs to project {proj.name!r}.") from e
            logger.warning("HOPSWORKS_PROJECT case mismatch: %r -> %r. "
                           "Fix .env to silence this.", wanted, proj.name)
            _PROJECT = proj
            return _PROJECT
    raise RuntimeError(f"Hopsworks login failed after {retries} attempts: {last}")
