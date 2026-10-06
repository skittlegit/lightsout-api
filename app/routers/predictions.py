"""Prediction endpoints.

GET  /next                       resolves the next round and returns prediction
GET  /{round}?season=2026        full prediction (pre + post-quali if available)
POST /{round}/refresh            invalidate cache, requires X-API-Key
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Optional

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from app.cache import (
    current_form_cache,
    invalidate_prediction,
    predictions_cache,
    predictions_key,
)
from app.auth import require_api_key
from app.config import REPO_ROOT, get_settings
from app.schemas.predictions import (
    ModePrediction,
    PredictionResponse,
    RefreshResponse,
)
from app.services.jolpica import jolpica
from app.services.predictor import predictor
from ml.features import DriverContext, RaceContext, grid_features

router = APIRouter()
log = logging.getLogger(__name__)

_Season = Query(default=None, ge=1950, le=2100)

# One lock per (season, round) so concurrent cold requests compute once and
# the rest are served from the cache the first request fills.
_compute_locks: dict[str, asyncio.Lock] = {}


# ---------------------------------------------------------------------------
# Lazy-loaded historical context for inference
# ---------------------------------------------------------------------------
# Raw checkpoint history retains championship points and qualifying positions.
# Reload when either history file changes; engineered training rows are a fallback.
_history_cache: dict = {"mtime": None, "races": None, "quali": None}


def _load_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = REPO_ROOT / "ml" / "data"
    # Raw results retain points, which engineered training rows do not contain.
    races_path = data_dir / "_checkpoint.parquet"
    quali_path = data_dir / "_checkpoint_quali.parquet"
    if not races_path.exists():
        races_path = data_dir / "training.parquet"
    if not quali_path.exists():
        quali_path = data_dir / "quali_training.parquet"

    if not races_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    mtime = (races_path.stat().st_mtime_ns, quali_path.stat().st_mtime_ns if quali_path.exists() else None)
    if _history_cache["mtime"] != mtime:
        log.info("Loading historical context from %s", races_path)
        df = pd.read_parquet(races_path)
        races = df.drop_duplicates(["season", "round", "driver_code"], keep="last").copy()
        races = races[races["finish_position"].notna() & (races["finish_position"] > 0)]
        if "points" not in races:
            races["points"] = 0.0
        quali = pd.read_parquet(quali_path) if quali_path.exists() else pd.DataFrame()
        if not quali.empty:
            quali = quali.drop_duplicates(["season", "round", "driver_code"], keep="last")
        _history_cache.update({"mtime": mtime, "races": races, "quali": quali})

    return _history_cache["races"], _history_cache["quali"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _resolve_race(season: int, round_: int) -> dict:
    races = await jolpica.schedule(season)
    for r in races:
        if r["round"] == round_:
            return r
    raise HTTPException(status_code=404, detail=f"Race {season} round {round_} not found")


async def _current_season_frames(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Live current-season race + quali results from Jolpica, shaped like the
    training history. Cached briefly so we don't re-fetch on every cache miss.

    This is what makes the model reflect *this* season's form: without it the
    only context is the static 2018–2025 parquet, so a dominant new-regulation
    pairing (e.g. 2026 Mercedes / Antonelli) would never surface.
    """
    key = f"form:{season}"
    cached = current_form_cache.get(key)
    if cached is not None:
        return cached
    try:
        races, quali = await asyncio.gather(
            jolpica.season_results(season), jolpica.season_qualifying(season),
        )
    except httpx.HTTPError as e:  # noqa: BLE001
        log.warning("current-season form fetch failed for %s: %s", season, e)
        return pd.DataFrame(), pd.DataFrame()
    # Training form uses Grand Prix results. Mixing sprints into only inference
    # changes the meaning of last-three-races and double-counts round keys.
    frames = (pd.DataFrame(races), pd.DataFrame(quali))
    current_form_cache[key] = frames
    return frames


async def _build_driver_contexts(
    season: int, round_: int, quali_rows: list[dict], results: pd.DataFrame,
) -> list[DriverContext]:
    # Standings contain every driver who appeared this season, including
    # substitutes. Use this race's qualifying roster or the latest known field.
    rows = quali_rows
    if not rows and not results.empty:
        eligible = results[(results["season"] == season) & (results["round"] <= round_)]
        if not eligible.empty:
            rows = eligible[eligible["round"] == eligible["round"].max()].to_dict("records")
    if not rows:
        rows = await jolpica.driver_standings(season)
    return [
        DriverContext(
            driver_code=s["driver_code"],
            driver_name=s["driver_name"],
            team=s["team"],
            team_tenure_months=12.0,  # TODO: derive from contract data
        )
        for s in rows
    ]


def _merge_prior(history: pd.DataFrame, live: pd.DataFrame, season: int, round_: int) -> pd.DataFrame:
    frame = pd.concat([history, live], ignore_index=True)
    if frame.empty:
        return frame
    prior = (frame["season"] < season) | ((frame["season"] == season) & (frame["round"] < round_))
    return frame[prior].drop_duplicates(["season", "round", "driver_code"], keep="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/next", response_model=PredictionResponse)
async def predict_next(season: int = _Season):
    season = season or date.today().year
    try:
        races = await jolpica.schedule(season)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream Jolpica error: {e}") from e
    nxt = next((r for r in races if r["is_next"]), None)
    if nxt is None:
        raise HTTPException(status_code=404, detail="No upcoming race in season")
    return await predict_round(round_=nxt["round"], season=season)


@router.get("/{round_}", response_model=PredictionResponse)
async def predict_round(round_: int, season: int = _Season):
    season = season or date.today().year
    cache_key = predictions_key(season, round_)
    if cache_key in predictions_cache:
        return predictions_cache[cache_key]
    lock = _compute_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        if cache_key in predictions_cache:
            return predictions_cache[cache_key]
        return await _compute_prediction(season, round_, cache_key)


async def _compute_prediction(season: int, round_: int, cache_key: str) -> PredictionResponse:
    try:
        # Probe whether qualifying has run for this round alongside the rest.
        race, quali_rows, (cur_races, cur_quali) = await asyncio.gather(
            _resolve_race(season, round_),
            jolpica.qualifying(season, round_),
            _current_season_frames(season),
        )
        history_races, history_quali = _load_history()
        roster = pd.concat([history_races, cur_races], ignore_index=True)
        if not roster.empty:
            roster = roster.drop_duplicates(["season", "round", "driver_code"], keep="last")
        drivers = await _build_driver_contexts(season, round_, quali_rows, roster)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream Jolpica error: {e}") from e

    if not drivers:
        raise HTTPException(status_code=503, detail="No driver standings available yet")

    if not predictor.pre_loaded and not predictor.post_loaded:
        return PredictionResponse(
            season=season,
            round=round_,
            race_name=race["race_name"],
            circuit=race["circuit"],
            race_date=race["race_date"],
            status="model_unavailable",
            message="Model artifacts not loaded — train and deploy them first.",
        )

    prior_races = _merge_prior(history_races, cur_races, season, round_)
    prior_quali = _merge_prior(history_quali, cur_quali, season, round_)

    settings = get_settings()
    race_ctx = RaceContext(
        season=season,
        round=round_,
        circuit=race["circuit"],
        round_in_season=round_,
        weather_rain_prob=0.1,  # TODO: hook to Open-Meteo
        weather_temp_c=22.0,
    )

    # LightGBM + Monte Carlo is CPU-bound (~1s); keep it off the event loop so
    # standings/calendar requests aren't stalled behind a cold forecast.
    # Pre-quali always
    pre = await run_in_threadpool(
        predictor.predict_mode,
        mode="pre_quali",
        drivers=drivers,
        race=race_ctx,
        prior_race_results=prior_races,
        prior_quali_results=prior_quali,
        n_simulations=settings.mc_simulations,
    )

    # Post-quali only if quali has actually been run
    post: Optional[ModePrediction] = None
    if quali_rows and predictor.post_loaded:
        grid = grid_features(quali_rows)
        post = await run_in_threadpool(
            predictor.predict_mode,
            mode="post_quali",
            drivers=drivers,
            race=race_ctx,
            prior_race_results=prior_races,
            prior_quali_results=prior_quali,
            grid=grid,
            n_simulations=settings.mc_simulations,
        )

    response = PredictionResponse(
        season=season,
        round=round_,
        race_name=race["race_name"],
        circuit=race["circuit"],
        race_date=race["race_date"],
        pre_quali=pre,
        post_quali=post,
    )
    predictions_cache[cache_key] = response
    return response


@router.post("/{round_}/refresh", response_model=RefreshResponse)
async def refresh_round(
    round_: int,
    season: int = _Season,
    _: str = Depends(require_api_key),
):
    season = season or date.today().year
    invalidated = invalidate_prediction(season, round_)
    current_form_cache.pop(f"form:{season}", None)
    return RefreshResponse(season=season, round=round_, invalidated=invalidated)
