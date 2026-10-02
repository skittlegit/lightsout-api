"""Refresh raw race classifications directly from Jolpica before rebuilding features."""
from __future__ import annotations

import argparse
import asyncio
from datetime import date
from pathlib import Path

import pandas as pd

from app.services.jolpica import jolpica
from ml.build_dataset import _clean_races


async def refresh(season: int, out_dir: Path) -> None:
    fresh = _clean_races(pd.DataFrame(await jolpica.season_results(season)))
    if fresh.empty:
        schedule = await jolpica.schedule(season)
        if any(race["is_completed"] for race in schedule):
            raise RuntimeError(f"No valid race results for {season}; existing checkpoint is unchanged")
        print(f"No completed races in {season} yet; existing checkpoint is unchanged")
        return
    path = out_dir / "_checkpoint.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if not existing.empty:
        # Replace whole published rounds so withdrawn/substitute entries from
        # older snapshots do not survive a correction to the official field.
        replaced = (existing["season"] == season) & existing["round"].isin(fresh["round"].unique())
        existing = existing[~replaced]
    merged = _clean_races(pd.concat([existing, fresh], ignore_index=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / "_checkpoint.refresh.parquet"
    merged.to_parquet(temporary, index=False)
    temporary.replace(path)
    print(f"Refreshed {season}: {fresh['round'].nunique()} rounds, {len(fresh)} driver results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--out-dir", type=Path, default=Path("ml/data"))
    args = parser.parse_args()
    asyncio.run(refresh(args.season, args.out_dir))


if __name__ == "__main__":
    main()
