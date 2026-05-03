"""Driver and constructor standings — Jolpica proxy with TTL cache."""
from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Query

from app.cache import standings_cache
from app.schemas.predictions import ConstructorStanding, DriverStanding
from app.services.jolpica import jolpica

router = APIRouter()


@router.get("/drivers", response_model=list[DriverStanding])
async def driver_standings(season: int = Query(..., ge=1950, le=2100)):
    key = f"drivers:{season}"
    if key in standings_cache:
        return standings_cache[key]
    try:
        rows = await jolpica.driver_standings(season)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream Jolpica error: {e}") from e
    standings_cache[key] = rows
    return rows


@router.get("/constructors", response_model=list[ConstructorStanding])
async def constructor_standings(season: int = Query(..., ge=1950, le=2100)):
    key = f"constructors:{season}"
    if key in standings_cache:
        return standings_cache[key]
    try:
        rows = await jolpica.constructor_standings(season)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream Jolpica error: {e}") from e
    standings_cache[key] = rows
    return rows
