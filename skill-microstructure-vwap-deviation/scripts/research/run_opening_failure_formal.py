"""Run governed IM opening-failure experiments on one frozen dataset."""

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
OUTPUT_ROOT = RUNS_ROOT / "index_mtf" / "formal_im_opening_failure_v1"

SCENARIOS = (
    ("baseline_core_only", True, False),
    ("opening_failure_short_only", False, True),
    ("core_plus_opening_failure_short", True, True),
)


def run_one(scenario: tuple[str, bool, bool]) -> dict:
    name, enable_core, enable_opening_failure = scenario
    output_dir = OUTPUT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "SSQUANT_DATA_MODE": "frozen",
            "SSQUANT_DATASET_MANIFEST": str(DATASET.resolve()),
            "SSQUANT_OUTPUT_DIR": str(output_dir.resolve()),
            "SSQUANT_SYMBOL": "IM888",
            "SSQUANT_BASE_PERIOD": "5m",
            "SSQUANT_HIGHER_PERIOD": "120m",
            "SSQUANT_START_DATE": "2024-01-02",
            "SSQUANT_END_DATE": "2026-04-30",
            "SSQUANT_INITIAL_CAPITAL": "300000",
            "SSQUANT_USE_REAL_IM_PARAMS": "1",
            "SSQUANT_ENTRY_Z": "2.25",
            "SSQUANT_EXIT_Z": "0.5",
            "SSQUANT_MAX_HOLD_MINUTES": "30",
            "SSQUANT_ENTRY_CONFIRMATION_BARS": "0",
            "SSQUANT_ENABLE_CORE_MEAN_REVERSION": "1" if enable_core else "0",
            "SSQUANT_ENABLE_OPENING_FAILURE_SHORT": (
                "1" if enable_opening_failure else "0"
            ),
            "SSQUANT_ENABLE_TREND_PULLBACK": "0",
            "SSQUANT_ENABLE_TREND_BREAKOUT": "0",
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
        "scenario": name,
        "enable_core_mean_reversion": enable_core,
        "enable_opening_failure_short": enable_opening_failure,
        "output_dir": str(output_dir.resolve()),
    }
    if completed.returncode != 0:
        return {**result, "status": "FAIL", "returncode": completed.returncode}

    verified = subprocess.run(
        [sys.executable, "-X", "utf8", str(VERIFY), str(output_dir)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (output_dir / "verify.log").write_text(
        verified.stdout + verified.stderr,
        encoding="utf-8",
    )
    if verified.returncode != 0:
        return {**result, "status": "VERIFY_FAIL"}

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    performance = manifest["performance"]
    return {
        **result,
        "status": "PASS",
        "bars_hash": manifest["bars_hash"],
        "adjustment_mode": manifest["adjustment_mode"],
        "strategy_file_hash": manifest["strategy_file_hash"],
        "total_return": performance.get("total_return"),
        "max_drawdown_pct": performance.get("max_drawdown_pct"),
        "sharpe_ratio": performance.get("sharpe_ratio"),
        "win_rate": performance.get("win_rate"),
        "profit_factor": performance.get("trade_stats", {}).get("profit_factor"),
        "round_trades": int(manifest.get("trade_count", 0)) // 2,
    }


def main() -> int:
    if not DATASET.is_file():
        raise RuntimeError(f"frozen dataset manifest not found: {DATASET}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=len(SCENARIOS)) as pool:
        futures = {pool.submit(run_one, scenario): scenario for scenario in SCENARIOS}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
    order = {scenario[0]: index for index, scenario in enumerate(SCENARIOS)}
    results.sort(key=lambda row: order[row["scenario"]])
    summary_path = OUTPUT_ROOT / "batch_summary.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    failures = [result for result in results if result["status"] != "PASS"]
    print(f"[FORMAL] {len(results) - len(failures)}/{len(results)} PASS -> {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
