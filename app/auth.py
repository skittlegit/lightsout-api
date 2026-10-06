"""Shared X-API-Key dependency for admin-only endpoints."""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from app.config import get_settings


def require_api_key(x_api_key: str = Header(default="")) -> str:
    expected = get_settings().retrain_api_key
    # Constant-time compare so the key can't be recovered via response timing.
    if not expected or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return x_api_key
