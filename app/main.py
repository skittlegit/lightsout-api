"""FastAPI app entry point.

Lifespan loads the three model pickles and starts the weekly auto-retrain
scheduler. If any artifacts are missing the app still starts — predictions
endpoints report `model_unavailable` so the dashboard's other panes
(standings, calendar) keep working.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import admin, calendar, predictions, standings
from app.services.predictor import predictor

log = logging.getLogger("lightsout")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_scheduler = BackgroundScheduler(timezone="UTC")


def _auto_retrain_job() -> None:
    """Runs in background thread: rebuild dataset then retrain models."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    log.info("[auto-retrain] starting weekly retrain")
    settings = get_settings()
    artifacts_dir = Path("ml/artifacts")
    tmp_dir = artifacts_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for cmd, label in [
        ([sys.executable, "-m", "ml.build_dataset"], "build_dataset"),
        ([sys.executable, "-m", "ml.train", "--out-dir", str(tmp_dir)], "train"),
    ]:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("[auto-retrain] %s failed:\n%s", label, result.stderr[-2000:])
            return
        log.info("[auto-retrain] %s complete", label)

    for pkl in tmp_dir.glob("*.pkl"):
        shutil.move(str(pkl), str(artifacts_dir / pkl.name))

    predictor.load(
        pre_quali_path=settings.model_pre_quali_path,
        post_quali_path=settings.model_post_quali_path,
        pole_path=settings.model_pole_path,
    )
    log.info("[auto-retrain] done — models reloaded: %s", predictor.loaded_models())


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

    # Start weekly auto-retrain scheduler if enabled
    cron = settings.auto_retrain_cron.strip()
    if cron:
        _scheduler.add_job(
            _auto_retrain_job,
            CronTrigger.from_crontab(cron, timezone="UTC"),
            id="auto_retrain",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        _scheduler.start()
        log.info("Auto-retrain scheduler started (cron: '%s' UTC)", cron)
    else:
        log.info("Auto-retrain disabled (AUTO_RETRAIN_CRON is empty)")

    yield

    if _scheduler.running:
        _scheduler.shutdown(wait=False)


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
    next_run = None
    job = _scheduler.get_job("auto_retrain")
    if job and job.next_run_time:
        next_run = job.next_run_time.isoformat()
    return {
        "status": "ok",
        "models_loaded": predictor.loaded_models(),
        "next_auto_retrain": next_run,
    }


app.include_router(standings.router, prefix="/api/standings", tags=["standings"])
app.include_router(calendar.router, prefix="/api/calendar", tags=["calendar"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(admin.router, prefix="/api", tags=["admin"])
