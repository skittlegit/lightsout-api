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
import threading
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.auth import require_api_key
from app.config import REPO_ROOT, get_settings
from app.schemas.predictions import RetrainAccepted
from app.services.predictor import predictor

router = APIRouter()
log = logging.getLogger(__name__)

# Manual and scheduled retrains share ml/artifacts/.tmp, so only one may run.
_retrain_lock = threading.Lock()


def run_retrain(job_id: str, *, rebuild_dataset: bool = False) -> None:
    """Invoke the ml pipeline as subprocesses, swap artifacts, reload models.

    Using subprocesses (rather than calling main() in-process) keeps the API
    event loop responsive even during heavy LightGBM fits. Callers must hold
    ``_retrain_lock``; it is released here when the run ends.
    """
    try:
        log.info("[retrain %s] starting", job_id)
        settings = get_settings()
        artifacts_dir = REPO_ROOT / "ml" / "artifacts"
        tmp_dir = artifacts_dir / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        steps = [([sys.executable, "-m", "ml.train", "--out-dir", str(tmp_dir)], "train")]
        if rebuild_dataset:
            steps.insert(0, ([sys.executable, "-m", "ml.build_dataset"], "build_dataset"))
        for cmd, label in steps:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=REPO_ROOT)
            if result.returncode != 0:
                log.error("[retrain %s] %s failed:\n%s", job_id, label, result.stderr[-2000:])
                return
            log.info("[retrain %s] %s complete", job_id, label)

        # Atomic-ish swap: rename each *.pkl from tmp_dir into artifacts_dir
        for pkl in tmp_dir.glob("*.pkl"):
            shutil.move(str(pkl), str(artifacts_dir / pkl.name))

        log.info("[retrain %s] artifacts swapped, reloading models", job_id)
        predictor.load(
            pre_quali_path=settings.model_pre_quali_path,
            post_quali_path=settings.model_post_quali_path,
            pole_path=settings.model_pole_path,
        )
        from app.cache import current_form_cache, predictions_cache
        predictions_cache.clear()
        current_form_cache.clear()
        log.info("[retrain %s] complete — models loaded: %s", job_id, predictor.loaded_models())
    except Exception as e:  # noqa: BLE001
        log.exception("[retrain %s] exception: %s", job_id, e)
    finally:
        _retrain_lock.release()


def try_start_retrain() -> bool:
    """Claim the retrain slot; False if a retrain is already running."""
    return _retrain_lock.acquire(blocking=False)


@router.post("/retrain", response_model=RetrainAccepted, status_code=status.HTTP_202_ACCEPTED)
async def retrain(
    background: BackgroundTasks,
    _: str = Depends(require_api_key),
):
    if not try_start_retrain():
        raise HTTPException(status_code=409, detail="A retrain is already running")
    job_id = uuid.uuid4().hex[:12]
    background.add_task(run_retrain, job_id)
    return RetrainAccepted(job_id=job_id)
