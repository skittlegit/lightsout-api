"""Admin endpoints: /retrain.

Runs training as a FastAPI BackgroundTask so the request returns immediately.
Training writes new artifacts to ml/artifacts/.tmp/ then atomically renames
into ml/artifacts/ — the running predictor keeps serving old models until the
swap succeeds, then we re-load.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, status

from app.config import get_settings
from app.schemas.predictions import RetrainAccepted
from app.services.predictor import predictor

router = APIRouter()
log = logging.getLogger(__name__)


def _require_api_key(x_api_key: str = Header(default="")) -> str:
    expected = get_settings().retrain_api_key
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return x_api_key


def _run_retrain(job_id: str) -> None:
    """Background task: invoke ml.train as a subprocess.

    Using a subprocess (rather than calling main() in-process) keeps the API
    event loop responsive even during heavy LightGBM fits.
    """
    log.info("[retrain %s] starting", job_id)
    settings = get_settings()
    artifacts_dir = Path("ml/artifacts")
    tmp_dir = artifacts_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "ml.train", "--out-dir", str(tmp_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.error("[retrain %s] failed:\n%s", job_id, result.stderr)
            return

        # Atomic-ish swap: rename each *.pkl from tmp_dir into artifacts_dir
        for pkl in tmp_dir.glob("*.pkl"):
            target = artifacts_dir / pkl.name
            shutil.move(str(pkl), str(target))

        log.info("[retrain %s] artifacts swapped, reloading models", job_id)
        predictor.load(
            pre_quali_path=settings.model_pre_quali_path,
            post_quali_path=settings.model_post_quali_path,
            pole_path=settings.model_pole_path,
        )
        log.info("[retrain %s] complete", job_id)
    except Exception as e:  # noqa: BLE001
        log.exception("[retrain %s] exception: %s", job_id, e)


@router.post("/retrain", response_model=RetrainAccepted, status_code=status.HTTP_202_ACCEPTED)
async def retrain(
    background: BackgroundTasks,
    _: str = Depends(_require_api_key),
):
    job_id = uuid.uuid4().hex[:12]
    background.add_task(_run_retrain, job_id)
    return RetrainAccepted(job_id=job_id)
