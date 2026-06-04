"""Prediction endpoints.

GET  /next                       resolves the next round and returns prediction
GET  /{round}?season=2026        full prediction (pre + post-quali if available)
POST /{round}/refresh            invalidate cache, requires X-API-Key
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.cache import (
    current_form_cache,
    invalidate_prediction,
    predictions_cache,
    predictions_key,
)
from app.config import get_settings
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


# ---------------------------------------------------------------------------
# Lazy-loaded historical context for inference
# ---------------------------------------------------------------------------
# We need prior race + quali results so feature-builder can compute form and
# track-history. Loaded from ml/data/training.parquet on first use; reload
# whenever the file's mtime changes.
_history_cache: dict = {"mtime": None, "races": None, "quali": None}


def _load_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    races_path = Path("ml/data/training.parquet")
    quali_path = Path("ml/data/quali_training.parquet")

    if not races_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    mtime = races_path.stat().st_mtime
    if _history_cache["mtime"] != mtime:
        log.info("Loading historical context from %s", races_path)
        df = pd.read_parquet(races_path)
        # training.parquet has both modes; for context we only need the labels.
        # Use mode == 'post_quali' rows since they include both finish & quali info.
        races = df[df["mode"] == "post_quali"].copy() if "mode" in df.columns else df.copy()
        quali = pd.read_parquet(quali_path) if quali_path.exists() else pd.DataFrame()
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
        races = await jolpica.season_results(season)
        sprints = await jolpica.season_sprint_results(season)
        quali = await jolpica.season_qualifying(season)
    except httpx.HTTPError as e:  # noqa: BLE001
        log.warning("current-season form fetch failed for %s: %s", season, e)
        return pd.DataFrame(), pd.DataFrame()
    # Sprint results share the schema and count toward form + championship
    # points, so merge them into the race-results frame.
    race_rows = races + sprints
    frames = (pd.DataFrame(race_rows), pd.DataFrame(quali))
    current_form_cache[key] = frames
    return frames


async def _build_driver_contexts(season: int) -> list[DriverContext]:
    standings = await jolpica.driver_standings(season)
    if not standings:
        return []
    return [
        DriverContext(
            driver_code=s["driver_code"],
            driver_name=s["driver_name"],
            team=s["team"],
            team_tenure_months=12.0,  # TODO: derive from contract data
        )
        for s in standings
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/next", response_model=PredictionResponse)
async def predict_next(season: int = Query(default=None)):
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
async def predict_round(round_: int, season: int = Query(default=None)):
    season = season or date.today().year
    cache_key = predictions_key(season, round_)
    if cache_key in predictions_cache:
        return predictions_cache[cache_key]

    try:
        race = await _resolve_race(season, round_)
        drivers = await _build_driver_contexts(season)
        # Probe whether qualifying has run for this round
        quali_rows = await jolpica.qualifying(season, round_)
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

    prior_races, prior_quali = _load_history()

    # Fold in this season's completed rounds (strictly before the target round
    # to avoid leakage) so form/points features reflect current-season pace.
    cur_races, cur_quali = await _current_season_frames(season)
    if not cur_races.empty:
        cur_races = cur_races[cur_races["round"] < round_]
        prior_races = pd.concat([prior_races, cur_races], ignore_index=True)
    if not cur_quali.empty:
        cur_quali = cur_quali[cur_quali["round"] < round_]
        prior_quali = pd.concat([prior_quali, cur_quali], ignore_index=True)

    settings = get_settings()
    race_ctx = RaceContext(
        season=season,
        round=round_,
        circuit=race["circuit"],
        round_in_season=round_,
        weather_rain_prob=0.1,  # TODO: hook to Open-Meteo
        weather_temp_c=22.0,
    )

    # Pre-quali always
    pre = predictor.predict_mode(
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
        post = predictor.predict_mode(
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


def _require_api_key(x_api_key: str = Header(default="")) -> str:
    expected = get_settings().retrain_api_key
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return x_api_key


@router.post("/{round_}/refresh", response_model=RefreshResponse)
async def refresh_round(
    round_: int,
    season: int = Query(default=None),
    _: str = Depends(_require_api_key),
):
    season = season or date.today().year
    invalidated = invalidate_prediction(season, round_)
    return RefreshResponse(season=season, round=round_, invalidated=invalidated)
