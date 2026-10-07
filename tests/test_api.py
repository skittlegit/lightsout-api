import asyncio
import unittest
from datetime import date
from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi.testclient import TestClient

from app.routers import admin, predictions
from app.services import jolpica as jolpica_module
from app.services.jolpica import jolpica
from ml.monte_carlo import run_simulation


def _schedule_payload(dates):
    return {"MRData": {"RaceTable": {"Races": [
        {"season": "2026", "round": str(i + 1), "raceName": f"R{i + 1}", "date": d, "time": "13:00:00Z",
         "Circuit": {"circuitName": "C", "Location": {"country": "X"}}}
        for i, d in enumerate(dates)
    ]}}}


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        jolpica_module._schedule_cache.clear()

    def test_cached_schedule_recomputes_next_race_each_call(self):
        fetch = AsyncMock(return_value=_schedule_payload(["2026-10-04", "2026-10-11", "2026-10-18"]))
        with patch.object(jolpica, "_get_json", new=fetch), patch.object(jolpica_module, "date") as fake_date:
            fake_date.today.return_value = date(2026, 10, 6)
            first = asyncio.run(jolpica.schedule(2026))
            fake_date.today.return_value = date(2026, 10, 12)
            second = asyncio.run(jolpica.schedule(2026))
        self.assertEqual(fetch.await_count, 1)
        self.assertEqual([r["round"] for r in first if r["is_next"]], [2])
        self.assertEqual([r["round"] for r in second if r["is_next"]], [3])
        self.assertEqual(first[0]["race_time"], "13:00:00Z")


class PredictionConcurrencyTests(unittest.TestCase):
    def test_concurrent_cold_requests_compute_once(self):
        calls = 0

        async def compute(season, round_, key):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            predictions.predictions_cache[key] = "result"
            return "result"

        async def run():
            return await asyncio.gather(*(predictions.predict_round(round_=99, season=2026) for _ in range(5)))

        predictions.predictions_cache.pop("2026:99", None)
        with patch.object(predictions, "_compute_prediction", new=compute):
            results = asyncio.run(run())
        predictions.predictions_cache.pop("2026:99", None)
        self.assertEqual(calls, 1)
        self.assertEqual(results, ["result"] * 5)


class MonteCarloTests(unittest.TestCase):
    def test_matrix_is_doubly_stochastic(self):
        prob = run_simulation(np.arange(22.0), np.full(22, 2.0), n_sims=2_000, rng=np.random.default_rng(0))
        self.assertTrue(np.allclose(prob.sum(axis=0), 1.0))
        self.assertTrue(np.allclose(prob.sum(axis=1), 1.0))
        self.assertEqual(int(np.argmax(prob[:, 0])), 0)


class AdminTests(unittest.TestCase):
    def test_retrain_requires_key_and_rejects_overlap(self):
        from app.main import app

        client = TestClient(app)
        with patch("app.auth.get_settings") as settings, patch.object(admin, "run_retrain"):
            settings.return_value.retrain_api_key = "secret"
            self.assertEqual(client.post("/api/retrain", headers={"X-API-Key": "nope"}).status_code, 401)
            self.assertTrue(admin.try_start_retrain())
            try:
                self.assertEqual(client.post("/api/retrain", headers={"X-API-Key": "secret"}).status_code, 409)
            finally:
                admin._retrain_lock.release()


if __name__ == "__main__":
    unittest.main()
