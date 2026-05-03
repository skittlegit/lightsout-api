"""Offline FastF1 wrappers used only by ml/build_dataset.

This module is NOT imported by the API runtime — keeps cold start fast.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


def init_cache(cache_dir: Path) -> None:
    import fastf1

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def load_session(season: int, round_: int, kind: str):
    """kind: 'R' (race), 'Q' (qualifying)."""
    import fastf1

    session = fastf1.get_session(season, round_, kind)
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    return session


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
        return fastf1.get_event_schedule(season, include_testing=False)
    except Exception as e:  # noqa: BLE001
        log.warning("season_schedule(%s) failed: %s", season, e)
        return None
