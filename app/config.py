"""Typed application configuration loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # CORS — stored as raw string so pydantic-settings never JSON-decodes it;
    # accepts: empty/"" (→ default), JSON array, or comma-separated list.
    cors_origins_raw: str = Field(
        default="",
        alias="CORS_ORIGINS",
    )

    @property
    def cors_origins(self) -> list[str]:
        import json as _json
        v = self.cors_origins_raw.strip()
        if not v:
            return ["http://localhost:3000"]
        if v.startswith("["):
            try:
                return _json.loads(v)
            except Exception:
                pass
        return [s.strip() for s in v.split(",") if s.strip()]

    # Upstream
    jolpica_base_url: str = Field(
        default="https://api.jolpi.ca/ergast/f1",
        alias="JOLPICA_BASE_URL",
    )

    # Monte Carlo
    mc_simulations: int = Field(default=10_000, alias="MC_SIMULATIONS")

    # Models
    model_pre_quali_path: Path = Field(
        default=REPO_ROOT / "ml" / "artifacts" / "pre_quali_finish.pkl",
        alias="MODEL_PRE_QUALI_PATH",
    )
    model_post_quali_path: Path = Field(
        default=REPO_ROOT / "ml" / "artifacts" / "post_quali_finish.pkl",
        alias="MODEL_POST_QUALI_PATH",
    )
    model_pole_path: Path = Field(
        default=REPO_ROOT / "ml" / "artifacts" / "pole.pkl",
        alias="MODEL_POLE_PATH",
    )

    fastf1_cache_dir: Path = Field(
        default=REPO_ROOT / "fastf1_cache",
        alias="FASTF1_CACHE_DIR",
    )

    retrain_api_key: str = Field(default="", alias="RETRAIN_API_KEY")

    # Weekly auto-retrain schedule (cron syntax, UTC).
    # Default: every Tuesday at 03:00 UTC (after most race weekends).
    # Set AUTO_RETRAIN_CRON="" to disable.
    auto_retrain_cron: str = Field(default="0 3 * * 2", alias="AUTO_RETRAIN_CRON")


@lru_cache
def get_settings() -> Settings:
    return Settings()
