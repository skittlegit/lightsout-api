"""FastAPI app entry point.

Lifespan loads the three model pickles. If any are missing the app still starts —
predictions endpoints will report `model_unavailable` so the dashboard's other
panes (standings, calendar) keep working.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import admin, calendar, predictions, standings
from app.services.predictor import predictor

log = logging.getLogger("lightsout")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("Loading model artifacts...")
    predictor.load(
        pre_quali_path=settings.model_pre_quali_path,
        post_quali_path=settings.model_post_quali_path,
        pole_path=settings.model_pole_path,
    )
    log.info("Models loaded: %s", predictor.loaded_models())
    yield
    # nothing to tear down


app = FastAPI(
    title="lightsout-api",
    version="1.0.0",
    description="F1 2026 race-result & pole-sitter prediction backend.",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": predictor.loaded_models(),
    }


app.include_router(standings.router, prefix="/api/standings", tags=["standings"])
app.include_router(calendar.router, prefix="/api/calendar", tags=["calendar"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(admin.router, prefix="/api", tags=["admin"])
