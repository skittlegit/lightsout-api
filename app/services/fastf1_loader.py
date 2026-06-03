"""Offline FastF1 wrappers used only by ml/build_dataset.

This module is NOT imported by the API runtime — keeps cold start fast.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# Seconds to sleep between consecutive API calls to stay under 200 calls/h
_CALL_DELAY = 0.5
# Max retries on RateLimitExceededError before giving up on a session
_MAX_RETRIES = 5
    
def init_cache(cache_dir: Path) -> None:
    import fastf1
    import fastf1.ergast.interface as ergast_interface

    from app.config import get_settings

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    # FastF1 3.4.4 sources race classification (Position/GridPosition/Points/
    # Status) from the Ergast API, but ergast.com has been shut down. Without
    # this, session.results returns the entry list with NaN positions and the
    # dataset builder silently defaults every finish to P20 — poisoning the
    # models. Repoint the backend at Jolpica, the Ergast-compatible replacement.
    ergast_interface.BASE_URL = get_settings().jolpica_base_url.rstrip("/")


def _with_backoff(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) retrying on RateLimitExceededError."""
    delay = 20  # initial backoff seconds
    for attempt in range(_MAX_RETRIES):
        try:
            time.sleep(_CALL_DELAY)
            return fn(*args, **kwargs)
        except Exception as e:
            if "RateLimitExceeded" in type(e).__name__ or "rate" in str(e).lower():
                wait = delay * (2 ** attempt)
                log.warning("Rate limit hit — waiting %ds before retry %d/%d", wait, attempt + 1, _MAX_RETRIES)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Exceeded {_MAX_RETRIES} retries due to rate limiting")


def load_session(season: int, round_: int, kind: str):
    """kind: 'R' (race), 'Q' (qualifying).

    Qualifying sessions need laps=True AND messages=True so FastF1 can
    identify deleted laps and correctly compute Q1/Q2/Q3 best times
    in session.results.
    """
    import fastf1

    needs_laps = kind == "Q"
    needs_messages = kind == "Q"

    def _load():
        session = fastf1.get_session(season, round_, kind)
        session.load(laps=needs_laps, telemetry=False, weather=False, messages=needs_messages)
        return session

    return _with_backoff(_load)


def race_results(season: int, round_: int) -> Optional[pd.DataFrame]:
    try:
        s = load_session(season, round_, "R")
        df = s.results.copy()
        df["season"] = season
        df["round"] = round_
        df["race_name"] = s.event.get("EventName", "")
        df["circuit"] = s.event.get("Location", "")
        return df
    except Exception as e:  # noqa: BLE001
        log.warning("race_results(%s, %s) failed: %s", season, round_, e)
        return None


def quali_results(season: int, round_: int) -> Optional[pd.DataFrame]:
    try:
        s = load_session(season, round_, "Q")
        df = s.results.copy()
        df["season"] = season
        df["round"] = round_
        return df
    except Exception as e:  # noqa: BLE001
        log.warning("quali_results(%s, %s) failed: %s", season, round_, e)
        return None


def season_schedule(season: int) -> Optional[pd.DataFrame]:
    import fastf1

    try:
        return _with_backoff(fastf1.get_event_schedule, season, include_testing=False)
    except Exception as e:  # noqa: BLE001
        log.warning("season_schedule(%s) failed: %s", season, e)
        return None
