"""Predictor service: loads the three quantile bundles, builds features, runs MC.

A bundle pickle is a dict::

    {
        "version":   "preq-v1.2",
        "features":  ["driver_form_last3", ...],
        "models":    {"q10": <booster>, "q50": <booster>, "q90": <booster>},
        "trained_at": "2026-04-01T12:00:00Z",
    }
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from app.schemas.predictions import (
    DriverPrediction,
    ModePrediction,
    PredictedPole,
)
from ml.features import (
    POLE_FEATURES,
    POST_QUALI_FEATURES,
    PRE_QUALI_FEATURES,
    DriverContext,
    RaceContext,
    build_inference_features,
    fill_missing,
)
from ml.monte_carlo import derive_scalars, run_simulation

log = logging.getLogger(__name__)

# Approx z-score range between q10 and q90 of a Normal: 2 * 1.2816 ≈ 2.5631
_NORMAL_Q90_Q10_Z = 2.5631


class _Bundle:
    def __init__(self, version: str, features: list[str], models: dict) -> None:
        self.version = version
        self.features = features
        self.models = models  # {'q10': ..., 'q50': ..., 'q90': ...}

    @classmethod
    def from_path(cls, path: Path) -> Optional["_Bundle"]:
        if not path.exists():
            log.warning("Model artifact missing: %s", path)
            return None
        try:
            obj = joblib.load(path)
            return cls(
                version=obj.get("version", path.stem),
                features=obj["features"],
                models=obj["models"],
            )
        except Exception as e:  # noqa: BLE001
            log.exception("Failed to load %s: %s", path, e)
            return None

    def predict_quantiles(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = X[self.features].to_numpy()
        return (
            self.models["q10"].predict(x),
            self.models["q50"].predict(x),
            self.models["q90"].predict(x),
        )


class Predictor:
    """Singleton holding the three loaded model bundles."""

    def __init__(self) -> None:
        self._pre: Optional[_Bundle] = None
        self._post: Optional[_Bundle] = None
        self._pole: Optional[_Bundle] = None

    # ---------- lifecycle ----------
    def load(self, *, pre_quali_path: Path, post_quali_path: Path, pole_path: Path) -> None:
        self._pre = _Bundle.from_path(pre_quali_path)
        self._post = _Bundle.from_path(post_quali_path)
        self._pole = _Bundle.from_path(pole_path)

    def loaded_models(self) -> list[str]:
        out = []
        if self._pre: out.append("pre_quali")
        if self._post: out.append("post_quali")
        if self._pole: out.append("pole")
        return out

    @property
    def pre_loaded(self) -> bool: return self._pre is not None

    @property
    def post_loaded(self) -> bool: return self._post is not None

    @property
    def pole_loaded(self) -> bool: return self._pole is not None

    # ---------- inference ----------
    def predict_mode(
        self,
        *,
        mode: str,  # 'pre_quali' or 'post_quali'
        drivers: list[DriverContext],
        race: RaceContext,
        prior_race_results: pd.DataFrame,
        prior_quali_results: pd.DataFrame,
        grid: Optional[dict[str, dict]] = None,
        n_simulations: int = 10_000,
    ) -> Optional[ModePrediction]:
        bundle = self._pre if mode == "pre_quali" else self._post
        if bundle is None:
            return None

        feat_cols = PRE_QUALI_FEATURES if mode == "pre_quali" else POST_QUALI_FEATURES
        df = build_inference_features(
            drivers, race, prior_race_results, prior_quali_results, grid=grid,
        )
        df = fill_missing(df, feat_cols)

        q10, q50, q90 = bundle.predict_quantiles(df)
        mu = np.asarray(q50, dtype=np.float64)
        sigma = np.clip((q90 - q10) / _NORMAL_Q90_Q10_Z, 0.5, None)

        prob = run_simulation(mu, sigma, n_sims=n_simulations)
        # Pad to 20 columns if fewer drivers (frontend always expects len 20)
        n = prob.shape[0]
        n_pad = max(0, 20 - n)
        if n_pad:
            prob = np.hstack([prob, np.zeros((n, n_pad))])

        scalars = derive_scalars(prob[:, :n])  # use unpadded for scalars

        # Predicted pole only sensible at pre-quali; pole model is separate
        pole_pred = self._predict_pole(drivers, race, prior_race_results, prior_quali_results) \
            if mode == "pre_quali" else None

        # Build per-driver records and sort by expected_position ascending
        records: list[DriverPrediction] = []
        for i, d in enumerate(drivers):
            records.append(DriverPrediction(
                driver_code=d.driver_code,
                driver_name=d.driver_name,
                team=d.team,
                expected_position=float(scalars["expected_position"][i]),
                win_probability=float(scalars["win_probability"][i]),
                podium_probability=float(scalars["podium_probability"][i]),
                points_probability=float(scalars["points_probability"][i]),
                position_distribution=[float(x) for x in prob[i].tolist()],
            ))
        records.sort(key=lambda r: r.expected_position)

        return ModePrediction(
            generated_at=datetime.now(timezone.utc),
            model_version=bundle.version,
            n_simulations=n_simulations,
            predicted_pole=pole_pred,
            drivers=records,
        )

    # ---------- pole model ----------
    def _predict_pole(
        self,
        drivers: list[DriverContext],
        race: RaceContext,
        prior_race_results: pd.DataFrame,
        prior_quali_results: pd.DataFrame,
    ) -> Optional[PredictedPole]:
        if self._pole is None:
            return None

        df = build_inference_features(
            drivers, race, prior_race_results, prior_quali_results,
        )
        df = fill_missing(df, POLE_FEATURES)
        _, q50, _ = self._pole.predict_quantiles(df)

        # Lower predicted gap-to-pole = faster.
        # Win-prob via softmax on negative gap; confidence = P1 − P2 margin.
        scores = -np.asarray(q50, dtype=np.float64)
        scores = scores - scores.max()
        weights = np.exp(scores * 4.0)  # temperature; tune empirically
        probs = weights / weights.sum()

        order = np.argsort(-probs)
        top = order[0]
        runner_up = order[1] if len(order) > 1 else top
        confidence = float(probs[top] - probs[runner_up])

        d = drivers[top]
        return PredictedPole(
            driver_code=d.driver_code,
            driver_name=d.driver_name,
            team=d.team,
            confidence=confidence,
        )


# Module-level singleton, loaded by the FastAPI lifespan
predictor = Predictor()
