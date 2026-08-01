"""Shared paths for the skill package and its external research runs."""

from __future__ import annotations

import os
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path(
    os.environ.get(
        "VWAP_RESEARCH_OUTPUT_ROOT",
        str(SKILL_ROOT.parent / f"{SKILL_ROOT.name}-runs"),
    )
).expanduser().resolve()
DATASET_MANIFEST = Path(
    os.environ.get(
        "VWAP_RESEARCH_DATASET_MANIFEST",
        str(RUNS_ROOT / "index_mtf" / "frozen_dataset" / "dataset_manifest.json"),
    )
).expanduser().resolve()


def run_path(*parts: str) -> Path:
    """Return a path under the external research-run directory."""

    return RUNS_ROOT.joinpath(*parts)
