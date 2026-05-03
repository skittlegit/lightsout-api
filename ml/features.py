"""SHARED feature definitions and builders.

Used identically by ``ml.build_dataset`` (training) and
``app.services.predictor`` (inference). Same code, same outputs — this is the
single most important file for avoiding train/inference skew. Do not duplicate
this logic anywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

PRE_QUALI_FEATURES: list[str] = [
    "driver_form_last3",
    "driver_form_last5",
    "driver_season_pts_pct",
    "team_form_last3",
    "team_season_pts_pct",
    "driver_track_history_avg",
    "driver_track_visits",
    "track_type_street",
    "track_type_permanent",
    "track_overtaking_score",
    "weather_rain_prob",
    "weather_temp_c",
    "round_in_season",
    "driver_team_tenure_months",
]

POST_QUALI_EXTRA: list[str] = [
    "grid_position",
    "quali_gap_to_pole_s",
    "front_row",
    "reached_q3",
    "teammate_quali_delta_s",
]
POST_QUALI_FEATURES: list[str] = PRE_QUALI_FEATURES + POST_QUALI_EXTRA

POLE_EXTRA: list[str] = [
    "driver_quali_form_last3",
    "team_quali_form_last3",
    "driver_track_quali_history",
]
POLE_FEATURES: list[str] = PRE_QUALI_FEATURES + POLE_EXTRA


# ---------------------------------------------------------------------------
# Track metadata (manual seed; refine from historical position-change data)
# ---------------------------------------------------------------------------
# overtaking_score: 0 = Monaco-impossible, 1 = Spa-easy
TRACK_META: dict[str, dict] = {
    "Monaco":             {"type": "street",    "overtaking": 0.05},
    "Singapore":          {"type": "street",    "overtaking": 0.20},
    "Baku":               {"type": "street",    "overtaking": 0.55},
    "Jeddah":             {"type": "street",    "overtaking": 0.45},
    "Miami":              {"type": "street",    "overtaking": 0.55},
    "Las Vegas":          {"type": "street",    "overtaking": 0.60},
    "Albert Park":        {"type": "hybrid",    "overtaking": 0.50},
    "Montreal":           {"type": "hybrid",    "overtaking": 0.65},
    "Spa":                {"type": "permanent", "overtaking": 1.00},
    "Monza":              {"type": "permanent", "overtaking": 0.90},
    "Silverstone":        {"type": "permanent", "overtaking": 0.85},
    "Bahrain":            {"type": "permanent", "overtaking": 0.75},
    "Sakhir":             {"type": "permanent", "overtaking": 0.75},
    "Imola":              {"type": "permanent", "overtaking": 0.30},
    "Barcelona":          {"type": "permanent", "overtaking": 0.45},
    "Red Bull Ring":      {"type": "permanent", "overtaking": 0.80},
    "Hungaroring":        {"type": "permanent", "overtaking": 0.25},
    "Zandvoort":          {"type": "permanent", "overtaking": 0.30},
    "Suzuka":             {"type": "permanent", "overtaking": 0.55},
    "Mexico City":        {"type": "permanent", "overtaking": 0.55},
    "Interlagos":         {"type": "permanent", "overtaking": 0.80},
    "Yas Marina":         {"type": "permanent", "overtaking": 0.40},
    "Lusail":             {"type": "permanent", "overtaking": 0.55},
    "COTA":               {"type": "permanent", "overtaking": 0.70},
    "Shanghai":           {"type": "permanent", "overtaking": 0.65},
    "Paul Ricard":        {"type": "permanent", "overtaking": 0.50},
    "Istanbul":           {"type": "permanent", "overtaking": 0.65},
    "Mugello":            {"type": "permanent", "overtaking": 0.40},
    "Nurburgring":        {"type": "permanent", "overtaking": 0.55},
    "Portimao":           {"type": "permanent", "overtaking": 0.55},
    "Sochi":              {"type": "permanent", "overtaking": 0.55},
    "Hockenheim":         {"type": "permanent", "overtaking": 0.65},
}


def lookup_track(circuit: str) -> dict:
    """Fuzzy-match a circuit name to TRACK_META; falls back to permanent/0.5."""
    if not circuit:
        return {"type": "permanent", "overtaking": 0.5}
    key = circuit.lower()
    for k, v in TRACK_META.items():
        if k.lower() in key or key in k.lower():
            return v
    return {"type": "permanent", "overtaking": 0.5}


# ---------------------------------------------------------------------------
# Form / history helpers — used by both training and inference
# ---------------------------------------------------------------------------

def driver_form(prior_results: pd.DataFrame, driver_code: str, n: int) -> float:
    """Mean finishing position over driver's prior n races. NaN-safe.

    ``prior_results`` MUST already be filtered to races strictly BEFORE the
    target race. Leakage protection is the caller's responsibility.
    """
    if prior_results.empty:
        return 10.5  # field midpoint prior
    rows = prior_results[prior_results["driver_code"] == driver_code]
    rows = rows.sort_values(["season", "round"]).tail(n)
    if rows.empty:
        return 10.5
    return float(rows["finish_position"].mean())


def team_form(prior_results: pd.DataFrame, team: str, n: int) -> float:
    if prior_results.empty:
        return 10.5
    rows = prior_results[prior_results["team"] == team]
    rows = rows.sort_values(["season", "round"]).tail(n * 2)  # 2 cars/team
    if rows.empty:
        return 10.5
    return float(rows["finish_position"].mean())


def season_points_pct(prior_results: pd.DataFrame, key: str, value: str, season: int) -> float:
    """Driver/team points / max points in the column, this season so far."""
    if prior_results.empty:
        return 0.0
    season_rows = prior_results[prior_results["season"] == season]
    if season_rows.empty:
        return 0.0
    pts = season_rows.groupby(key)["points"].sum()
    if pts.empty or pts.max() == 0:
        return 0.0
    return float(pts.get(value, 0.0) / pts.max())


def track_history(
    prior_results: pd.DataFrame, driver_code: str, circuit: str, n: int = 3
) -> tuple[float, int]:
    """(avg finish at this circuit over last n visits, total visits)."""
    if prior_results.empty:
        return 10.5, 0
    rows = prior_results[
        (prior_results["driver_code"] == driver_code)
        & (prior_results["circuit"].str.contains(circuit, case=False, na=False)
           if circuit else False)
    ]
    visits = len(rows)
    rows = rows.sort_values(["season", "round"]).tail(n)
    if rows.empty:
        return 10.5, visits
    return float(rows["finish_position"].mean()), visits


def driver_quali_form(prior_quali: pd.DataFrame, driver_code: str, n: int = 3) -> float:
    if prior_quali.empty:
        return 10.5
    rows = prior_quali[prior_quali["driver_code"] == driver_code]
    rows = rows.sort_values(["season", "round"]).tail(n)
    if rows.empty:
        return 10.5
    return float(rows["quali_position"].mean())


def team_quali_form(prior_quali: pd.DataFrame, team: str, n: int = 3) -> float:
    if prior_quali.empty:
        return 10.5
    rows = prior_quali[prior_quali["team"] == team]
    rows = rows.sort_values(["season", "round"]).tail(n * 2)
    if rows.empty:
        return 10.5
    return float(rows["quali_position"].mean())


def driver_track_quali_history(
    prior_quali: pd.DataFrame, driver_code: str, circuit: str, n: int = 3
) -> float:
    if prior_quali.empty or not circuit:
        return 10.5
    rows = prior_quali[
        (prior_quali["driver_code"] == driver_code)
        & (prior_quali["circuit"].str.contains(circuit, case=False, na=False))
    ]
    rows = rows.sort_values(["season", "round"]).tail(n)
    if rows.empty:
        return 10.5
    return float(rows["quali_position"].mean())


# ---------------------------------------------------------------------------
# Inference-time feature builder
# ---------------------------------------------------------------------------

@dataclass
class DriverContext:
    driver_code: str
    driver_name: str
    team: str
    team_tenure_months: float = 12.0


@dataclass
class RaceContext:
    season: int
    round: int
    circuit: str
    round_in_season: int
    weather_rain_prob: float = 0.1
    weather_temp_c: float = 22.0


def build_inference_features(
    drivers: list[DriverContext],
    race: RaceContext,
    prior_race_results: pd.DataFrame,
    prior_quali_results: pd.DataFrame,
    *,
    grid: Optional[dict[str, dict]] = None,
) -> pd.DataFrame:
    """Build a feature DataFrame, one row per driver.

    If ``grid`` is provided (post-quali), the post-quali extras are added.
    Otherwise only PRE_QUALI_FEATURES are populated (POLE_FEATURES likewise
    populated; the predictor selects the right column subset per model).
    """
    track = lookup_track(race.circuit)

    rows: list[dict] = []
    for d in drivers:
        f3 = driver_form(prior_race_results, d.driver_code, 3)
        f5 = driver_form(prior_race_results, d.driver_code, 5)
        tf3 = team_form(prior_race_results, d.team, 3)
        d_pts_pct = season_points_pct(prior_race_results, "driver_code", d.driver_code, race.season)
        t_pts_pct = season_points_pct(prior_race_results, "team", d.team, race.season)
        th_avg, th_visits = track_history(prior_race_results, d.driver_code, race.circuit)

        row: dict = {
            "driver_code": d.driver_code,
            "driver_name": d.driver_name,
            "team": d.team,
            "driver_form_last3": f3,
            "driver_form_last5": f5,
            "driver_season_pts_pct": d_pts_pct,
            "team_form_last3": tf3,
            "team_season_pts_pct": t_pts_pct,
            "driver_track_history_avg": th_avg,
            "driver_track_visits": th_visits,
            "track_type_street": int(track["type"] == "street"),
            "track_type_permanent": int(track["type"] == "permanent"),
            "track_overtaking_score": float(track["overtaking"]),
            "weather_rain_prob": float(race.weather_rain_prob),
            "weather_temp_c": float(race.weather_temp_c),
            "round_in_season": int(race.round_in_season),
            "driver_team_tenure_months": float(d.team_tenure_months),
            # pole-model extras
            "driver_quali_form_last3": driver_quali_form(prior_quali_results, d.driver_code),
            "team_quali_form_last3": team_quali_form(prior_quali_results, d.team),
            "driver_track_quali_history": driver_track_quali_history(
                prior_quali_results, d.driver_code, race.circuit
            ),
        }

        if grid is not None:
            g = grid.get(d.driver_code, {})
            row.update({
                "grid_position": int(g.get("grid_position", 20)),
                "quali_gap_to_pole_s": float(g.get("quali_gap_to_pole_s", 2.0)),
                "front_row": int(g.get("grid_position", 99) <= 2),
                "reached_q3": int(g.get("reached_q3", False)),
                "teammate_quali_delta_s": float(g.get("teammate_quali_delta_s", 0.0)),
            })

        rows.append(row)

    return pd.DataFrame(rows)


def grid_features(quali_results: list[dict]) -> dict[str, dict]:
    """Convert Jolpica qualifying results into a per-driver grid-feature dict.

    ``quali_results`` items must have: position, driver_code, team, q1, q2, q3.
    """
    if not quali_results:
        return {}

    def _sec(s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        try:
            if ":" in s:
                m, rest = s.split(":", 1)
                return int(m) * 60 + float(rest)
            return float(s)
        except (ValueError, TypeError):
            return None

    # best lap per driver across Q1/Q2/Q3
    best: dict[str, float] = {}
    for q in quali_results:
        times = [t for t in (_sec(q.get("q1")), _sec(q.get("q2")), _sec(q.get("q3"))) if t]
        if times:
            best[q["driver_code"]] = min(times)

    pole_time = min(best.values()) if best else 0.0

    # team grouping for teammate delta
    team_drivers: dict[str, list[str]] = {}
    for q in quali_results:
        team_drivers.setdefault(q["team"], []).append(q["driver_code"])

    out: dict[str, dict] = {}
    for q in quali_results:
        code = q["driver_code"]
        my = best.get(code)
        teammates = [c for c in team_drivers.get(q["team"], []) if c != code]
        tm = next((best[c] for c in teammates if c in best), None)
        out[code] = {
            "grid_position": int(q.get("position", 20)),
            "quali_gap_to_pole_s": (my - pole_time) if (my is not None and pole_time) else 2.0,
            "reached_q3": _sec(q.get("q3")) is not None,
            "teammate_quali_delta_s": (my - tm) if (my is not None and tm is not None) else 0.0,
        }
    return out


def fill_missing(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    """Make sure every requested feature column exists; fill NaN with neutral priors."""
    df = df.copy()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[list(feature_cols)] = df[list(feature_cols)].fillna(0.0)
    return df
