"""Async client for Jolpica (Ergast-compatible) F1 data.

Wraps standings, schedule, and qualifying endpoints. All public methods raise
``httpx.HTTPError`` on upstream failure — the routers translate that to a
502 Bad Gateway. We never hide upstream failures as 500.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from app.config import get_settings


_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


class JolpicaClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or get_settings().jolpica_base_url).rstrip("/")

    async def _get_json(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(url, headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json()

    # ---------- standings ----------
    async def driver_standings(self, season: int) -> list[dict]:
        data = await self._get_json(f"{season}/driverStandings.json")
        lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        if not lists:
            return []
        out: list[dict] = []
        for row in lists[0].get("DriverStandings", []):
            d = row.get("Driver", {})
            ctors = row.get("Constructors", [{}])
            out.append({
                "position": int(row.get("position", 0)),
                "driver_code": d.get("code") or _fallback_code(d),
                "driver_name": f"{d.get('givenName', '')} {d.get('familyName', '')}".strip(),
                "team": ctors[0].get("name", "") if ctors else "",
                "points": float(row.get("points", 0)),
                "wins": int(row.get("wins", 0)),
            })
        return out

    async def constructor_standings(self, season: int) -> list[dict]:
        data = await self._get_json(f"{season}/constructorStandings.json")
        lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        if not lists:
            return []
        out: list[dict] = []
        for row in lists[0].get("ConstructorStandings", []):
            c = row.get("Constructor", {})
            out.append({
                "position": int(row.get("position", 0)),
                "team": c.get("name", ""),
                "points": float(row.get("points", 0)),
                "wins": int(row.get("wins", 0)),
            })
        return out

    # ---------- schedule ----------
    async def schedule(self, season: int) -> list[dict]:
        data = await self._get_json(f"{season}.json")
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        today = date.today().isoformat()
        out: list[dict] = []
        for r in races:
            circuit = r.get("Circuit", {})
            out.append({
                "season": int(r.get("season", season)),
                "round": int(r.get("round", 0)),
                "race_name": r.get("raceName", ""),
                "circuit": circuit.get("circuitName", ""),
                "country": circuit.get("Location", {}).get("country", ""),
                "race_date": r.get("date", ""),
                "is_completed": bool(r.get("date", "")) and r["date"] < today,
                "is_next": False,  # filled in below
            })
        # mark the first not-yet-completed race as next
        for race in out:
            if not race["is_completed"]:
                race["is_next"] = True
                break
        return out

    # ---------- qualifying ----------
    async def qualifying(self, season: int, round_: int) -> list[dict]:
        """Return per-driver qualifying results, or [] if not yet run."""
        try:
            data = await self._get_json(f"{season}/{round_}/qualifying.json")
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 404:
                return []
            raise
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            return []
        out: list[dict] = []
        for q in races[0].get("QualifyingResults", []):
            d = q.get("Driver", {})
            ctors = q.get("Constructor", {})
            out.append({
                "position": int(q.get("position", 0)),
                "driver_code": d.get("code") or _fallback_code(d),
                "driver_name": f"{d.get('givenName', '')} {d.get('familyName', '')}".strip(),
                "team": ctors.get("name", ""),
                "q1": q.get("Q1"),
                "q2": q.get("Q2"),
                "q3": q.get("Q3"),
            })
        return out

    async def has_qualifying(self, season: int, round_: int) -> bool:
        results = await self.qualifying(season, round_)
        return len(results) > 0


def _fallback_code(driver: dict) -> str:
    fam = driver.get("familyName", "")
    return (fam[:3] or "UNK").upper()


jolpica = JolpicaClient()
