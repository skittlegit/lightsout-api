"""Shared names for FastF1 training data and Jolpica inference data."""
from __future__ import annotations

import unicodedata
from functools import lru_cache


TEAM_ALIASES = {
    "Red Bull Racing": "Red Bull",
    "Alpine": "Alpine F1 Team",
    "Racing Bulls": "RB F1 Team",
    "RB": "RB F1 Team",
    "Cadillac": "Cadillac F1 Team",
}

CIRCUIT_ALIASES = {
    "Albert Park": ("melbourne",),
    "Montreal": ("montreal", "gilles villeneuve"),
    "Monaco": ("monte carlo",),
    "Miami": ("miami gardens",),
    "Singapore": ("marina bay",),
    "Red Bull Ring": ("spielberg",),
    "Hungaroring": ("budapest",),
    "Yas Marina": ("yas island",),
    "COTA": ("austin", "circuit of the americas"),
    "Interlagos": ("sao paulo", "jose carlos pace"),
    "Paul Ricard": ("le castellet",),
    "Mexico City": ("hermanos rodriguez",),
    "Madrid": ("madring",),
    "Sepang": ("sepang international circuit",),
}


def _plain(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c)).lower()


@lru_cache(maxsize=1024)
def canonical_circuit(value: str) -> str:
    key = _plain(str(value))
    for name, aliases in CIRCUIT_ALIASES.items():
        if any(_plain(alias) in key for alias in (name, *aliases)):
            return name
    # Common names embedded in Jolpica's full circuit names.
    for name in ("Spa", "Monza", "Silverstone", "Suzuka", "Barcelona", "Zandvoort",
                 "Baku", "Jeddah", "Shanghai", "Imola", "Sakhir", "Lusail",
                 "Las Vegas", "Hockenheim", "Nurburgring", "Portimao", "Mugello",
                 "Istanbul", "Sochi"):
        if _plain(name) in key:
            return name
    return str(value)


def canonical_team(value: str) -> str:
    return TEAM_ALIASES.get(value, value)
