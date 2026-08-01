"""Run the package regression suite with cache output outside the skill package."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(
    os.environ.get(
        "VWAP_RESEARCH_OUTPUT_ROOT",
        str(SKILL_ROOT.parent / f"{SKILL_ROOT.name}-runs"),
    )
).expanduser().resolve()
CACHE_DIR = RUNS_ROOT / "pytest_cache"


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(SKILL_ROOT),
            "-q",
            "-o",
            f"cache_dir={CACHE_DIR}",
        ],
        cwd=RUNS_ROOT,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
