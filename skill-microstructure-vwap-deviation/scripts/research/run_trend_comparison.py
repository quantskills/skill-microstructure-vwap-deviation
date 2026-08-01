"""Run the fixed IM/IC/IF representative set with the multifactor trend filter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.project_paths import DATASET_MANIFEST, RUNS_ROOT

RUNNER = ROOT / "run_index_mtf_formal.py"
VERIFY = Path.home() / ".codex" / "skills" / "ssquant-backtest" / "scripts" / "verify_formal_run.py"
DATASET = DATASET_MANIFEST
OUTPUT_ROOT = RUNS_ROOT / "index_mtf" / "formal_trend_multifactor"

COMBOS = (
    ("IM888", "5m", "120m", 30, 2.25, 0.5),
    ("IC888", "5m", "30m", 60, 2.5, 0.25),
    ("IF888", "5m", "30m", 30, 2.0, 0.25),
)


def run_one(combo: tuple[str, str, str, int, float, float]) -> dict:
    symbol, base, higher, hold, entry_z, exit_z = combo
    output_dir = OUTPUT_ROOT / f"hold_{hold}m_entry_{entry_z:g}_exit_{exit_z:g}" / symbol / f"{base}_{higher}"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "SSQUANT_DATA_MODE": "frozen",
            "SSQUANT_DATASET_MANIFEST": str(DATASET.resolve()),
            "SSQUANT_SYMBOL": symbol,
            "SSQUANT_BASE_PERIOD": base,
            "SSQUANT_HIGHER_PERIOD": higher,
            "SSQUANT_MAX_HOLD_MINUTES": str(hold),
            "SSQUANT_ENTRY_Z": str(entry_z),
            "SSQUANT_EXIT_Z": str(exit_z),
            "SSQUANT_OUTPUT_DIR": str(output_dir.resolve()),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", str(RUNNER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (output_dir / "runner.log").write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
        encoding="utf-8",
    )
    result = {
        "symbol": symbol,
        "base_period": base,
        "higher_period": higher,
        "max_hold_minutes": hold,
        "entry_z": entry_z,
        "exit_z": exit_z,
        "output_dir": str(output_dir.resolve()),
    }
    if completed.returncode != 0:
        result.update({"status": "FAIL", "returncode": completed.returncode})
        return result

    verify = subprocess.run(
        [sys.executable, "-X", "utf8", str(VERIFY), str(output_dir)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if verify.returncode != 0:
        result.update({"status": "VERIFY_FAIL", "verify_output": verify.stdout + verify.stderr})
        return result

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    performance = manifest.get("performance", {})
    result.update(
        {
            "status": "PASS",
            "bars_hash": manifest.get("bars_hash"),
            "strategy_version": manifest.get("strategy_version"),
            "strategy_file_hash": manifest.get("strategy_file_hash"),
            "total_return": performance.get("total_return"),
            "max_drawdown_pct": performance.get("max_drawdown_pct"),
            "sharpe_ratio": performance.get("sharpe_ratio"),
            "win_rate": performance.get("win_rate"),
            "trade_count": manifest.get("trade_count"),
            "round_trades": int(manifest.get("trade_count") or 0) // 2,
        }
    )
    return result


def main() -> int:
    if not DATASET.is_file():
        raise RuntimeError(f"frozen dataset manifest not found: {DATASET}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_one, combo): combo for combo in COMBOS}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=True))
    results.sort(key=lambda row: row["symbol"])
    summary = OUTPUT_ROOT / "trend_summary.json"
    summary.write_text(json.dumps(results, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    failures = [row for row in results if row["status"] != "PASS"]
    print(f"[TREND] {len(results) - len(failures)}/{len(results)} PASS -> {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
