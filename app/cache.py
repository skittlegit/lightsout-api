"""In-memory TTL caches. Three namespaces with different TTLs.

When we scale horizontally these can be swapped for Redis behind the same
get/set API; for now per-process is fine.
"""
from __future__ import annotations

from typing import Any

from cachetools import TTLCache


# 1h for standings (changes after each race)
standings_cache: TTLCache[str, Any] = TTLCache(maxsize=64, ttl=60 * 60)

# 24h for calendar (rarely changes mid-season)
calendar_cache: TTLCache[str, Any] = TTLCache(maxsize=16, ttl=60 * 60 * 24)

# 6h for predictions; manually invalidated by /refresh after qualifying
predictions_cache: TTLCache[str, Any] = TTLCache(maxsize=128, ttl=60 * 60 * 6)


def predictions_key(season: int, round_: int) -> str:
    return f"{season}:{round_}"


def invalidate_prediction(season: int, round_: int) -> bool:
    key = predictions_key(season, round_)
    return predictions_cache.pop(key, None) is not None
