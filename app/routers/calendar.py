"""Season calendar — Jolpica proxy.

The raw schedule is cached inside the Jolpica client; is_next / is_completed
are recomputed per request so a finished race never lingers as "next".
"""
from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Query

from app.schemas.predictions import CalendarResponse
from app.services.jolpica import jolpica

router = APIRouter()


@router.get("", response_model=CalendarResponse)
async def get_calendar(season: int = Query(..., ge=1950, le=2100)):
    try:
        races = await jolpica.schedule(season)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream Jolpica error: {e}") from e
    return {"season": season, "races": races}
