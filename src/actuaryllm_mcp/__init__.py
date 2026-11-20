"""TurkActuary MCP sunucusu. Sürümün tek kaynağı `__version__`dır (açık paketin pyproject'i buradan okur)."""
from __future__ import annotations

from pathlib import Path

__version__ = "0.1.3"

# Tekerlekte (wheel) veri bu klasöre gömülür: data/, artifacts/, evidence/ depodaki göreli düzenle.
BUNDLE = Path(__file__).resolve().parent / "_bundle"


def default_root() -> Path | None:
    """Kurulu pakette gömülü veri kökü; git kopyasında None (kök paketin bulunduğu depodur)."""
    return BUNDLE if (BUNDLE / "data/sources.json").is_file() else None
