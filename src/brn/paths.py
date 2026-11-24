"""Locations of the bundled network data and of new experiment outputs.

``BRN_DATA`` overrides the TNTP directory and ``BRN_OUTPUTS`` the output directory. Archived
paper records live in ``results/`` and are never written by the experiment scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = Path(os.environ.get("BRN_OUTPUTS", REPO_ROOT / "outputs"))
OUTPUTS.mkdir(parents=True, exist_ok=True)


def tntp_dir() -> Path:
    """Directory with ``<network>_net.tntp`` and ``<network>_trips.tntp``."""
    return Path(os.environ.get("BRN_DATA", REPO_ROOT / "data" / "tntp"))


def upstream_root() -> Path:
    """Location of the vendored equilibrium solver (``brn.sue``)."""
    return Path(__file__).resolve().parent / "sue"


def ensure_upstream_on_path() -> Path:
    """Kept for compatibility with the research scripts; the solver is vendored."""
    return upstream_root()
