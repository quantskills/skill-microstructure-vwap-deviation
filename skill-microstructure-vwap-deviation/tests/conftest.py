"""Make the external SSQuant skill runtime available to package tests."""

import sys
from pathlib import Path


SSQUANT_SKILL_ROOT = Path.home() / ".codex" / "skills" / "ssquant-backtest"
if str(SSQUANT_SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SSQUANT_SKILL_ROOT))
