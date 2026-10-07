import asyncio
import unittest
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd

from ml import weather


def _hourly(day, key, values):
    return {"hourly": {"time": [f"{day}T{h:02d}:00" for h in range(24)], "temperature_2m": [20.0] * 24, key: values}}


class WeatherTests(unittest.TestCase):
    def test_observed_rain_is_share_of_wet_race_hours(self):
        precip = [0.0] * 24
        precip[12], precip[13] = 1.5, 0.05  # wet start, dry-ish next hour
        start = weather.race_start("2025-10-05", "12:00:00Z")
        rain, temp = weather.parse_observed(_hourly("2025-10-05", "precipitation", precip), start)
        self.assertAlmostEqual(rain, 1 / 3)
        self.assertEqual(temp, 20.0)

    def test_forecast_uses_peak_probability_in_window(self):
        probs = [0] * 24
        probs[14] = 70
        start = weather.race_start("2026-10-11", "13:00:00Z")
        self.assertEqual(weather.parse_forecast(_hourly("2026-10-11", "precipitation_probability", probs), start)[0], 0.7)

    def test_far_future_race_uses_circuit_climatology_without_network(self):
        table = pd.DataFrame([
            {"season": 2024, "round": 1, "circuit_id": "spa", "rain": 1.0, "temp_c": 15.0},
            {"season": 2025, "round": 1, "circuit_id": "spa", "rain": 0.0, "temp_c": 19.0},
        ])
        race = {"season": 2026, "round": 1, "circuit_id": "spa", "lat": 50.4, "long": 5.9,
                "race_date": (date.today() + timedelta(days=60)).isoformat(), "race_time": "13:00:00Z"}
        with patch("ml.weather.httpx.AsyncClient") as client:
            result = asyncio.run(weather.race_weather(race, table))
        client.assert_not_called()
        self.assertEqual(result, (0.5, 17.0, "climatology"))

    def test_past_race_uses_observed_row(self):
        table = pd.DataFrame([{"season": 2025, "round": 13, "circuit_id": "spa", "rain": 1.0, "temp_c": 17.7}])
        self.assertEqual(weather.lookup(table, 2025, 13), (1.0, 17.7))
        self.assertEqual(weather.lookup(pd.DataFrame(), 2025, 13), (weather.DEFAULT_RAIN, weather.DEFAULT_TEMP_C))


if __name__ == "__main__":
    unittest.main()
