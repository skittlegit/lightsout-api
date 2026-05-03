"""Pydantic response schemas for the predictions API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DriverStanding(BaseModel):
    position: int
    driver_code: str
    driver_name: str
    team: str
    points: float
    wins: int


class ConstructorStanding(BaseModel):
    position: int
    team: str
    points: float
    wins: int


class Race(BaseModel):
    season: int
    round: int
    race_name: str
    circuit: str
    country: str
    race_date: str  # ISO date
    is_next: bool = False
    is_completed: bool = False


class CalendarResponse(BaseModel):
    season: int
    races: list[Race]


class PredictedPole(BaseModel):
    driver_code: str
    driver_name: str
    team: str
    confidence: float = Field(..., description="Win-prob margin over P2")


class DriverPrediction(BaseModel):
    driver_code: str
    driver_name: str
    team: str
    expected_position: float
    win_probability: float
    podium_probability: float
    points_probability: float
    position_distribution: list[float] = Field(
        ..., description="Length 20, sums to 1.0; index k = P(finish in P(k+1))"
    )


class ModePrediction(BaseModel):
    generated_at: datetime
    model_version: str
    n_simulations: int
    predicted_pole: Optional[PredictedPole] = None
    drivers: list[DriverPrediction]


class PredictionResponse(BaseModel):
    season: int
    round: int
    race_name: str
    circuit: str
    race_date: str
    pre_quali: Optional[ModePrediction] = None
    post_quali: Optional[ModePrediction] = None
    status: Literal["ok", "model_unavailable"] = "ok"
    message: Optional[str] = None


class RetrainAccepted(BaseModel):
    job_id: str
    status: Literal["accepted"] = "accepted"
    message: str = "Retrain started in background"


class RefreshResponse(BaseModel):
    season: int
    round: int
    invalidated: bool
