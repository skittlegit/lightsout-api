"""Build the training parquet from FastF1 data, 2018-2025.

Produces:
  ml/data/training.parquet         — two rows per (driver, race): mode='pre_quali' and mode='post_quali'
  ml/data/quali_training.parquet   — one row per (driver, race), target = gap_to_pole_s

CRITICAL: when computing form features, only use races with (season, round) strictly
LESS THAN the target race. Leakage here will inflate backtest metrics and silently
mislead model selection.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import get_settings
from app.services import fastf1_loader as f1
from ml.features import (
    POLE_FEATURES,
    POST_QUALI_FEATURES,
    PRE_QUALI_FEATURES,
    DriverContext,
    RaceContext,
    build_inference_features,
    fill_missing,
    grid_features,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Helpers — extract clean per-driver records from FastF1 results frames
# ---------------------------------------------------------------------------

def _race_records(df: pd.DataFrame) -> list[dict]:
    """Normalize a FastF1 race-results frame into our internal schema."""
    out: list[dict] = []
    for _, row in df.iterrows():
        try:
            out.append({
                "season": int(row["season"]),
                "round": int(row["round"]),
                "race_name": row.get("race_name", ""),
                "circuit": row.get("circuit", ""),
                "driver_code": str(row.get("Abbreviation", "")) or _abbr(row),
                "driver_name": f"{row.get('FirstName', '')} {row.get('LastName', '')}".strip(),
                "team": str(row.get("TeamName", "")),
                "grid_position": _safe_int(row.get("GridPosition"), 20),
                "finish_position": _safe_int(row.get("Position"), 20),
                "points": float(row.get("Points") or 0.0),
                "status": str(row.get("Status", "")),
            })
        except Exception as e:  # noqa: BLE001
            log.warning("skip race row: %s", e)
    return out


def _quali_records(df: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    for _, row in df.iterrows():
        try:
            q1 = _td_seconds(row.get("Q1"))
            q2 = _td_seconds(row.get("Q2"))
            q3 = _td_seconds(row.get("Q3"))
            best = min([t for t in (q1, q2, q3) if t is not None], default=None)
            out.append({
                "season": int(row["season"]),
                "round": int(row["round"]),
                "circuit": row.get("circuit", "") or row.get("EventName", ""),
                "driver_code": str(row.get("Abbreviation", "")) or _abbr(row),
                "team": str(row.get("TeamName", "")),
                "quali_position": _safe_int(row.get("Position"), 20),
                "q1_s": q1, "q2_s": q2, "q3_s": q3,
                "best_quali_s": best,
                "reached_q3": q3 is not None,
            })
        except Exception as e:  # noqa: BLE001
            log.warning("skip quali row: %s", e)
    return out


def _safe_int(v, default: int) -> int:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return int(v)
    except (ValueError, TypeError):
        return default


def _td_seconds(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, pd.Timedelta):
        return float(v.total_seconds()) if not pd.isna(v) else None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _abbr(row) -> str:
    last = str(row.get("LastName", ""))[:3].upper()
    return last or "UNK"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build(seasons: range, out_dir: Path) -> None:
    settings = get_settings()
    f1.init_cache(settings.fastf1_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pull every race + quali we can from FastF1
    all_race_rows: list[dict] = []
    all_quali_rows: list[dict] = []

    for season in seasons:
        sched = f1.season_schedule(season)
        if sched is None:
            continue
        rounds = [int(r) for r in sched["RoundNumber"].dropna().tolist() if r >= 1]
        for rnd in rounds:
            log.info("Pulling %s round %s", season, rnd)
            race_df = f1.race_results(season, rnd)
            if race_df is not None and not race_df.empty:
                all_race_rows.extend(_race_records(race_df))
            quali_df = f1.quali_results(season, rnd)
            if quali_df is not None and not quali_df.empty:
                all_quali_rows.extend(_quali_records(quali_df))

    races_df = pd.DataFrame(all_race_rows)
    quali_df = pd.DataFrame(all_quali_rows)

    if races_df.empty:
        raise RuntimeError("No race data pulled — check FastF1 cache and connectivity.")

    races_df = races_df.sort_values(["season", "round"]).reset_index(drop=True)
    quali_df = quali_df.sort_values(["season", "round"]).reset_index(drop=True) if not quali_df.empty else quali_df

    # 2) For each (season, round) build features using ONLY data strictly prior.
    pre_rows: list[dict] = []
    post_rows: list[dict] = []
    quali_train_rows: list[dict] = []

    keys = races_df[["season", "round"]].drop_duplicates().values.tolist()

    for season, rnd in keys:
        prior_mask = (races_df["season"] < season) | (
            (races_df["season"] == season) & (races_df["round"] < rnd)
        )
        prior_races = races_df[prior_mask]
        prior_quali = (
            quali_df[(quali_df["season"] < season) | ((quali_df["season"] == season) & (quali_df["round"] < rnd))]
            if not quali_df.empty else pd.DataFrame()
        )

        race_rows = races_df[(races_df["season"] == season) & (races_df["round"] == rnd)]
        if race_rows.empty:
            continue

        circuit = str(race_rows.iloc[0]["circuit"])
        race_name = str(race_rows.iloc[0]["race_name"])

        # Round-in-season: position of this round in the season's chronological order
        season_rounds = sorted(races_df[races_df["season"] == season]["round"].unique().tolist())
        round_in_season = season_rounds.index(rnd) + 1 if rnd in season_rounds else rnd

        # Driver contexts for this race
        drivers = [
            DriverContext(
                driver_code=str(r["driver_code"]),
                driver_name=str(r["driver_name"]),
                team=str(r["team"]),
                team_tenure_months=12.0,
            )
            for _, r in race_rows.iterrows()
        ]
        race_ctx = RaceContext(
            season=int(season),
            round=int(rnd),
            circuit=circuit,
            round_in_season=round_in_season,
            weather_rain_prob=0.1,
            weather_temp_c=22.0,
        )

        # Quali results for this race for grid_features
        race_quali_rows = (
            quali_df[(quali_df["season"] == season) & (quali_df["round"] == rnd)]
            if not quali_df.empty else pd.DataFrame()
        )
        grid: dict[str, dict] = {}
        if not race_quali_rows.empty:
            # build a Jolpica-shaped list to reuse grid_features()
            shaped = []
            for _, qr in race_quali_rows.iterrows():
                shaped.append({
                    "position": int(qr["quali_position"]),
                    "driver_code": str(qr["driver_code"]),
                    "team": str(qr["team"]),
                    "q1": str(qr["q1_s"]) if qr["q1_s"] is not None else None,
                    "q2": str(qr["q2_s"]) if qr["q2_s"] is not None else None,
                    "q3": str(qr["q3_s"]) if qr["q3_s"] is not None else None,
                })
            grid = grid_features(shaped)

        # Pre-quali features
        pre_df = build_inference_features(drivers, race_ctx, prior_races, prior_quali)
        pre_df = fill_missing(pre_df, PRE_QUALI_FEATURES)

        # Post-quali features (only if we have grid)
        post_df = None
        if grid:
            post_df = build_inference_features(drivers, race_ctx, prior_races, prior_quali, grid=grid)
            post_df = fill_missing(post_df, POST_QUALI_FEATURES)

        # Pole features (pre_quali + extras)
        pole_df = build_inference_features(drivers, race_ctx, prior_races, prior_quali)
        pole_df = fill_missing(pole_df, POLE_FEATURES)

        # Map driver_code -> finishing position label
        finish_by_code = {
            str(r["driver_code"]): int(r["finish_position"])
            for _, r in race_rows.iterrows()
        }

        # Pole label: gap-to-pole in seconds
        pole_label_by_code: dict[str, float] = {}
        if not race_quali_rows.empty:
            best = race_quali_rows.dropna(subset=["best_quali_s"])
            if not best.empty:
                pole_t = float(best["best_quali_s"].min())
                for _, qr in best.iterrows():
                    pole_label_by_code[str(qr["driver_code"])] = float(qr["best_quali_s"]) - pole_t

        for _, frow in pre_df.iterrows():
            code = frow["driver_code"]
            if code not in finish_by_code:
                continue
            row = frow.to_dict()
            row.update({
                "season": int(season), "round": int(rnd),
                "race_name": race_name, "circuit": circuit,
                "mode": "pre_quali",
                "finish_position": finish_by_code[code],
            })
            pre_rows.append(row)

        if post_df is not None:
            for _, frow in post_df.iterrows():
                code = frow["driver_code"]
                if code not in finish_by_code:
                    continue
                row = frow.to_dict()
                row.update({
                    "season": int(season), "round": int(rnd),
                    "race_name": race_name, "circuit": circuit,
                    "mode": "post_quali",
                    "finish_position": finish_by_code[code],
                })
                post_rows.append(row)

        if pole_label_by_code:
            for _, frow in pole_df.iterrows():
                code = frow["driver_code"]
                if code not in pole_label_by_code:
                    continue
                row = frow.to_dict()
                row.update({
                    "season": int(season), "round": int(rnd),
                    "circuit": circuit,
                    "gap_to_pole_s": pole_label_by_code[code],
                    "quali_position": int(
                        race_quali_rows[race_quali_rows["driver_code"] == code]["quali_position"].iloc[0]
                    ),
                })
                quali_train_rows.append(row)

    training = pd.DataFrame(pre_rows + post_rows)
    quali_training = pd.DataFrame(quali_train_rows)

    training_path = out_dir / "training.parquet"
    quali_path = out_dir / "quali_training.parquet"
    training.to_parquet(training_path, index=False)
    quali_training.to_parquet(quali_path, index=False)
    log.info("Wrote %s (%d rows)", training_path, len(training))
    log.info("Wrote %s (%d rows)", quali_path, len(quali_training))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2018)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--out-dir", type=Path, default=Path("ml/data"))
    args = p.parse_args()
    build(range(args.start, args.end + 1), args.out_dir)


if __name__ == "__main__":
    main()
