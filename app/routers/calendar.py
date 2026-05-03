"""Season calendar — Jolpica proxy with 24h cache."""
from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Query

from app.cache import calendar_cache
from app.schemas.predictions import CalendarResponse
from app.services.jolpica import jolpica

router = APIRouter()


@router.get("", response_model=CalendarResponse)
async def get_calendar(season: int = Query(..., ge=1950, le=2100)):
    key = f"calendar:{season}"
    if key in calendar_cache:
        return calendar_cache[key]
    try:
        races = await jolpica.schedule(season)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream Jolpica error: {e}") from e
    payload = {"season": season, "races": races}
    calendar_cache[key] = payload
    return payload
