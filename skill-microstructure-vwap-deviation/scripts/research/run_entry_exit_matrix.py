"""Run a governed entry/exit threshold matrix on frozen index data."""

from __future__ import annotations

import argparse
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
COMBOS = (
    ("IM888", "5m", "120m", 30.0, (2.25, 2.5, 2.75)),
    ("IC888", "5m", "30m", 60.0, (2.25, 2.5, 2.75)),
    ("IF888", "5m", "30m", 30.0, (2.0, 2.25, 2.5)),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen entry/exit threshold matrix")
    parser.add_argument(
        "--dataset-manifest",
        default=str(DATASET_MANIFEST),
    )
    parser.add_argument(
        "--output-root",
        default=str(RUNS_ROOT / "index_mtf" / "formal_entry_exit"),
    )
    parser.add_argument("--exit-z", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--max-workers", type=int, default=3)
    return parser.parse_args()


def _run_variant(
    symbol: str,
    base_period: str,
    higher_period: str,
    hold_minutes: float,
    entry_z: float,
    exit_z: float,
    manifest: Path,
    output_root: Path,
) -> dict:
    label = f"hold_{int(hold_minutes)}m_entry_{entry_z:g}_exit_{exit_z:g}"
    output_dir = output_root / label / symbol / f"{base_period}_{higher_period}"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "SSQUANT_DATA_MODE": "frozen",
            "SSQUANT_DATASET_MANIFEST": str(manifest.resolve()),
            "SSQUANT_SYMBOL": symbol,
            "SSQUANT_BASE_PERIOD": base_period,
            "SSQUANT_HIGHER_PERIOD": higher_period,
            "SSQUANT_MAX_HOLD_MINUTES": str(int(hold_minutes)),
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
        "base_period": base_period,
        "higher_period": higher_period,
        "max_hold_minutes": int(hold_minutes),
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
    run_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    performance = run_manifest.get("performance", {})
    result.update(
        {
            "status": "PASS",
            "bars_hash": run_manifest.get("bars_hash"),
            "strategy_file_hash": run_manifest.get("strategy_file_hash"),
            "total_return": performance.get("total_return"),
            "max_drawdown_pct": performance.get("max_drawdown_pct"),
            "sharpe_ratio": performance.get("sharpe_ratio"),
            "win_rate": performance.get("win_rate"),
            "trade_count": run_manifest.get("trade_count"),
            "round_trades": int(run_manifest.get("trade_count") or 0) // 2,
        }
    )
    return result


def main() -> int:
    args = _parse_args()
    manifest = Path(args.dataset_manifest).expanduser().resolve()
    if not manifest.is_file():
        raise RuntimeError(f"Dataset manifest not found: {manifest}")
    output_root = Path(args.output_root).expanduser().resolve()
    jobs = [
        (symbol, base, higher, hold, entry_z, exit_z)
        for symbol, base, higher, hold, entry_values in COMBOS
        for entry_z in entry_values
        for exit_z in args.exit_z
    ]
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {
            pool.submit(_run_variant, symbol, base, higher, hold, entry_z, exit_z, manifest, output_root):
            (symbol, base, higher, hold, entry_z, exit_z)
            for symbol, base, higher, hold, entry_z, exit_z in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=True))
    results.sort(key=lambda row: (row["symbol"], row["entry_z"], row["exit_z"]))
    summary = output_root / "entry_exit_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    failures = [row for row in results if row["status"] != "PASS"]
    print(f"[ENTRY_EXIT] {len(results) - len(failures)}/{len(results)} PASS -> {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
