"""Race-window weather from Open-Meteo, shared by training and inference.

Training uses what actually happened (archive API): ``rain`` is the share of
race-window hours with measurable precipitation. Inference uses the forecast
API's precipitation probability for the same window, so both land on a 0–1
"how wet is the race" scale. Races beyond the forecast horizon fall back to
that circuit's climatology from the backfilled table.

Backfill / refresh the table (run before ``ml.build_dataset``)::

    python -m ml.weather            # 2018 → current season, only missing races
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd

log = logging.getLogger(__name__)

WEATHER_PATH = Path(__file__).resolve().parent / "data" / "weather.parquet"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_HORIZON_DAYS = 15

# Neutral priors, identical to the constants the models used before weather.
DEFAULT_RAIN = 0.1
DEFAULT_TEMP_C = 22.0

_RACE_HOURS = 2  # sample the start hour plus the next two
_WET_MM = 0.1    # hourly precipitation that counts as a wet hour
_TIMEOUT = httpx.Timeout(20.0, connect=5.0)


def race_start(race_date: str, race_time: Optional[str]) -> datetime:
    """UTC start; Jolpica times look like "13:00:00Z". Defaults to 13:00 UTC."""
    hhmm = (race_time or "13:00:00Z").rstrip("Z")[:5]
    return datetime.fromisoformat(f"{race_date}T{hhmm}:00").replace(tzinfo=timezone.utc)


def _window(hourly: dict, start: datetime, key: str) -> list[float]:
    end = start + timedelta(hours=_RACE_HOURS)
    lo = start.replace(minute=0)
    out = []
    for stamp, value in zip(hourly.get("time", []), hourly.get(key, [])):
        t = datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)
        if lo <= t <= end and value is not None:
            out.append(float(value))
    return out


def _params(lat: float, lon: float, start: datetime, hourly: str) -> dict:
    day = start.date().isoformat()
    return {"latitude": lat, "longitude": lon, "start_date": day, "end_date": day,
            "hourly": hourly, "timezone": "UTC"}


def parse_observed(payload: dict, start: datetime) -> Optional[tuple[float, float]]:
    hourly = payload.get("hourly", {})
    precip = _window(hourly, start, "precipitation")
    temps = _window(hourly, start, "temperature_2m")
    if not precip or not temps:
        return None  # archive lags real time by a few days
    rain = sum(p >= _WET_MM for p in precip) / len(precip)
    return rain, sum(temps) / len(temps)


def parse_forecast(payload: dict, start: datetime) -> Optional[tuple[float, float]]:
    hourly = payload.get("hourly", {})
    probs = _window(hourly, start, "precipitation_probability")
    temps = _window(hourly, start, "temperature_2m")
    if not probs or not temps:
        return None
    return max(probs) / 100.0, sum(temps) / len(temps)


# ---------------------------------------------------------------------------
# Table + climatology
# ---------------------------------------------------------------------------
_table_cache: dict = {"mtime": None, "frame": None}


def load_table(path: Path = WEATHER_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["season", "round", "circuit_id", "rain", "temp_c"])
    mtime = path.stat().st_mtime_ns
    if _table_cache["mtime"] != mtime:
        _table_cache.update({"mtime": mtime, "frame": pd.read_parquet(path)})
    return _table_cache["frame"]


def lookup(table: pd.DataFrame, season: int, round_: int, circuit_id: str = "") -> tuple[float, float]:
    """Observed weather for a past race, else circuit climatology, else priors."""
    return _lookup(table, season, round_, circuit_id)[:2]


def _lookup(table: pd.DataFrame, season: int, round_: int, circuit_id: str) -> tuple[float, float, str]:
    if not table.empty:
        row = table[(table["season"] == season) & (table["round"] == round_)]
        if not row.empty:
            return float(row["rain"].iloc[0]), float(row["temp_c"].iloc[0]), "observed"
    return (*climatology(table, circuit_id), "climatology")


def climatology(table: pd.DataFrame, circuit_id: str) -> tuple[float, float]:
    rows = table[table["circuit_id"] == circuit_id] if circuit_id and not table.empty else table.iloc[0:0]
    if rows.empty:
        return DEFAULT_RAIN, DEFAULT_TEMP_C
    return float(rows["rain"].mean()), float(rows["temp_c"].mean())


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
async def race_weather(race: dict, table: Optional[pd.DataFrame] = None) -> tuple[float, float, str]:
    """(rain 0–1, temp °C, source) for a schedule row from ``JolpicaClient.schedule``.

    Never raises: any upstream failure degrades to climatology.
    """
    table = load_table() if table is None else table
    season, round_ = int(race["season"]), int(race["round"])
    circuit_id = race.get("circuit_id", "")
    fallback = _lookup(table, season, round_, circuit_id)
    start = race_start(race["race_date"], race.get("race_time"))
    days_out = (start.date() - date.today()).days
    if not (0 <= days_out <= FORECAST_HORIZON_DAYS) or race.get("lat") is None:
        return fallback
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(FORECAST_URL, params=_params(
                race["lat"], race["long"], start, "temperature_2m,precipitation_probability"))
            r.raise_for_status()
            parsed = parse_forecast(r.json(), start)
            return (*parsed, "forecast") if parsed else fallback
    except httpx.HTTPError as e:
        log.warning("weather forecast failed for %s round %s: %s", season, round_, e)
        return fallback


# ---------------------------------------------------------------------------
# Backfill (offline, used by CI before build_dataset)
# ---------------------------------------------------------------------------
def _schedule(client: httpx.Client, base_url: str, season: int) -> list[dict]:
    r = client.get(f"{base_url}/{season}.json", params={"limit": 100})
    r.raise_for_status()
    return r.json().get("MRData", {}).get("RaceTable", {}).get("Races", [])


def backfill(seasons: range, path: Path = WEATHER_PATH) -> pd.DataFrame:
    from app.config import get_settings

    base_url = get_settings().jolpica_base_url.rstrip("/")
    existing = load_table(path)
    done = set(zip(existing["season"].astype(int), existing["round"].astype(int))) if not existing.empty else set()
    rows: list[dict] = []
    today = date.today().isoformat()
    with httpx.Client(timeout=_TIMEOUT) as client:
        for season in seasons:
            for race in _schedule(client, base_url, season):
                key = (int(race["season"]), int(race["round"]))
                if key in done or race.get("date", "9999") >= today:
                    continue
                loc = race["Circuit"]["Location"]
                start = race_start(race["date"], race.get("time"))
                r = client.get(ARCHIVE_URL, params=_params(
                    float(loc["lat"]), float(loc["long"]), start, "temperature_2m,precipitation"))
                if r.status_code == 429:
                    log.warning("Open-Meteo rate limit — stopping; rerun to resume")
                    break
                r.raise_for_status()
                parsed = parse_observed(r.json(), start)
                if parsed is None:
                    log.info("no archive data yet for %s round %s", *key)
                    continue
                rows.append({"season": key[0], "round": key[1],
                             "circuit_id": race["Circuit"]["circuitId"],
                             "rain": parsed[0], "temp_c": parsed[1]})
                time.sleep(0.2)
    if not rows:
        log.info("weather table up to date (%d races)", len(existing))
        return existing
    fresh = pd.DataFrame(rows)
    merged = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
    merged = merged.drop_duplicates(["season", "round"], keep="last").sort_values(["season", "round"])
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    log.info("added %d races → %s (%d total)", len(rows), path, len(merged))
    return merged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Backfill race-window weather from Open-Meteo")
    p.add_argument("--start", type=int, default=2018)
    p.add_argument("--end", type=int, default=date.today().year)
    args = p.parse_args()
    backfill(range(args.start, args.end + 1))


if __name__ == "__main__":
    main()
