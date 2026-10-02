import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

from app.routers import predictions
from ml.build_dataset import _clean_races
from ml.features import DriverContext, RaceContext, build_inference_features, grid_features
from ml.train import _fit_bundle
from ml.refresh_results import refresh


class DataRefreshTests(unittest.TestCase):
    def test_empty_refresh_preserves_checkpoint_and_detects_missing_completed_results(self):
        with TemporaryDirectory() as directory:
            data = Path(directory)
            checkpoint = data / "_checkpoint.parquet"
            checkpoint.write_bytes(b"existing checkpoint")
            with patch("ml.refresh_results.jolpica.season_results", new=AsyncMock(return_value=[])), patch("ml.refresh_results.jolpica.schedule", new=AsyncMock(return_value=[{"is_completed": False}])):
                asyncio.run(refresh(2026, data))
            self.assertEqual(checkpoint.read_bytes(), b"existing checkpoint")
            with patch("ml.refresh_results.jolpica.season_results", new=AsyncMock(return_value=[])), patch("ml.refresh_results.jolpica.schedule", new=AsyncMock(return_value=[{"is_completed": True}])):
                with self.assertRaises(RuntimeError):
                    asyncio.run(refresh(2026, data))
            self.assertEqual(checkpoint.read_bytes(), b"existing checkpoint")

    def test_refresh_replaces_published_round_without_stale_driver(self):
        with TemporaryDirectory() as directory:
            data = Path(directory)
            pd.DataFrame([
                {"season": 2026, "round": 1, "driver_code": "OLD", "finish_position": 1},
                {"season": 2026, "round": 1, "driver_code": "AAA", "finish_position": 2},
            ]).to_parquet(data / "_checkpoint.parquet")
            published = [
                {"season": 2026, "round": 1, "driver_code": "NEW", "finish_position": 1},
                {"season": 2026, "round": 1, "driver_code": "AAA", "finish_position": 2},
            ]
            with patch("ml.refresh_results.jolpica.season_results", new=AsyncMock(return_value=published)):
                asyncio.run(refresh(2026, data))
            self.assertEqual(set(pd.read_parquet(data / "_checkpoint.parquet").driver_code), {"NEW", "AAA"})

    def test_invalid_checkpoint_round_remains_retryable(self):
        rows = pd.DataFrame([
            {"season": 2026, "round": rnd, "driver_code": code, "finish_position": finish}
            for rnd, finishes in [(1, [1, 2]), (2, [20, 20]), (3, [None, None])]
            for code, finish in zip(["AAA", "BBB"], finishes)
        ])
        self.assertEqual(_clean_races(rows)["round"].unique().tolist(), [1])

    def test_live_history_overwrites_duplicates_and_excludes_target(self):
        old = pd.DataFrame([
            {"season": 2026, "round": 1, "driver_code": "AAA", "points": 18},
            {"season": 2026, "round": 2, "driver_code": "AAA", "points": 25},
            {"season": 2027, "round": 1, "driver_code": "AAA", "points": 25},
        ])
        live = pd.DataFrame([{"season": 2026, "round": 1, "driver_code": "AAA", "points": 25}])
        merged = predictions._merge_prior(old, live, 2026, 2)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["points"], 25)

    def test_raw_history_preserves_points_and_quali_only_change_reloads(self):
        with TemporaryDirectory() as directory, patch.object(predictions, "REPO_ROOT", Path(directory)), patch.dict(predictions._history_cache, {"mtime": None}):
            data = Path(directory) / "ml" / "data"
            data.mkdir(parents=True)
            pd.DataFrame([{"season": 2026, "round": 1, "driver_code": "AAA", "finish_position": 1, "points": 25}]).to_parquet(data / "_checkpoint.parquet")
            quali = data / "_checkpoint_quali.parquet"
            pd.DataFrame([{"season": 2026, "round": 1, "driver_code": "AAA", "quali_position": 2}]).to_parquet(quali)
            races, _ = predictions._load_history()
            self.assertEqual(races.iloc[0]["points"], 25)
            pd.DataFrame([{"season": 2026, "round": 1, "driver_code": "AAA", "quali_position": 1}]).to_parquet(quali)
            # Guarantee a different timestamp even on coarse filesystems.
            import os
            stat = quali.stat()
            os.utime(quali, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            _, q = predictions._load_history()
            self.assertEqual(q.iloc[0]["quali_position"], 1)

    def test_roster_uses_latest_field_instead_of_season_substitutes(self):
        rows = pd.DataFrame([
            {"season": 2026, "round": rnd, "driver_code": code, "driver_name": code, "team": "Mercedes"}
            for rnd, codes in [(1, ["OLD", "AAA"]), (2, ["NEW", "AAA"])] for code in codes
        ])
        with patch.object(predictions.jolpica, "driver_standings", new=AsyncMock()) as standings:
            drivers = asyncio.run(predictions._build_driver_contexts(2026, 3, [], rows))
            self.assertEqual({d.driver_code for d in drivers}, {"NEW", "AAA"})
            standings.assert_not_called()
            qualified = [{"driver_code": "BBB", "driver_name": "BBB", "team": "Ferrari"}]
            drivers = asyncio.run(predictions._build_driver_contexts(2026, 3, qualified, rows))
            self.assertEqual([d.driver_code for d in drivers], ["BBB"])

    def test_training_and_live_names_share_track_and_team_features(self):
        races = pd.DataFrame([{"season": 2026, "round": 1, "driver_code": "AAA", "team": "Red Bull Racing", "circuit": "Melbourne", "finish_position": 1, "points": 25}])
        quali = races.assign(quali_position=1)
        frame = build_inference_features([DriverContext("AAA", "A", "Red Bull")], RaceContext(2026, 2, "Albert Park Grand Prix Circuit", 2), races, quali)
        self.assertEqual(frame.iloc[0]["team_form_last3"], 1)
        self.assertEqual(frame.iloc[0]["driver_track_visits"], 1)
        self.assertEqual(frame.iloc[0]["driver_track_quali_history"], 1)

    def test_nan_qualifying_times_do_not_poison_other_drivers(self):
        grid = grid_features([
            {"driver_code": "AAA", "team": "A", "position": 1, "q1": "nan", "q3": "80.0"},
            {"driver_code": "BBB", "team": "B", "position": 2, "q1": "81.0", "q3": "nan"},
        ])
        self.assertEqual(grid["BBB"]["quali_gap_to_pole_s"], 1)
        self.assertFalse(grid["BBB"]["reached_q3"])

    def test_deployed_bundle_refits_with_refreshed_season(self):
        old = pd.DataFrame({"season": [2025], "round": [24], "feature": [1.0], "finish_position": [1]})
        new = pd.DataFrame({"season": [2026], "round": [15], "feature": [2.0], "finish_position": [2]})
        with patch("ml.train._fit_quantile") as fit, patch("ml.train._evaluate_finish", return_value={"mae": 1}):
            bundle, _ = _fit_bundle(old, new, ["feature"], "finish_position", "test")
        self.assertEqual([len(call.args[0]) for call in fit.call_args_list], [1, 1, 1, 2, 2, 2])
        self.assertEqual(bundle["training_through"], {"season": 2026, "round": 15})


if __name__ == "__main__":
    unittest.main()
