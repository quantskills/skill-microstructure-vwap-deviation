"""Run governed index MTF combinations and verify each formal directory."""

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

from research.index_matrix_contract import validate_period_pair
from research.project_paths import DATASET_MANIFEST, RUNS_ROOT


RUNNER = ROOT / "run_index_mtf_formal.py"
VERIFY = Path.home() / ".codex" / "skills" / "ssquant-backtest" / "scripts" / "verify_formal_run.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen index MTF matrix")
    parser.add_argument("--symbols", nargs="+", default=["IM888", "IF888", "IC888"])
    parser.add_argument("--base-periods", nargs="+", default=["5m"])
    parser.add_argument("--higher-periods", nargs="+", default=["30m", "60m", "90m", "120m"])
    parser.add_argument(
        "--dataset-manifest",
        default=str(DATASET_MANIFEST),
    )
    parser.add_argument(
        "--output-root",
        default=str(RUNS_ROOT / "index_mtf" / "formal"),
    )
    parser.add_argument("--max-workers", type=int, default=3)
    return parser.parse_args()


def _run_one(symbol: str, base: str, higher: str, manifest: Path, output_root: Path) -> dict:
    output_dir = output_root / symbol / f"{base}_{higher}"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "SSQUANT_DATA_MODE": "frozen",
            "SSQUANT_DATASET_MANIFEST": str(manifest.resolve()),
            "SSQUANT_SYMBOL": symbol,
            "SSQUANT_BASE_PERIOD": base,
            "SSQUANT_HIGHER_PERIOD": higher,
            "SSQUANT_OUTPUT_DIR": str(output_dir.resolve()),
        }
    )
    command = [sys.executable, "-X", "utf8", str(RUNNER)]
    completed = subprocess.run(
        command,
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
    if completed.returncode != 0:
        return {
            "symbol": symbol,
            "base_period": base,
            "higher_period": higher,
            "status": "FAIL",
            "returncode": completed.returncode,
            "output_dir": str(output_dir.resolve()),
        }
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
        return {
            "symbol": symbol,
            "base_period": base,
            "higher_period": higher,
            "status": "VERIFY_FAIL",
            "output_dir": str(output_dir.resolve()),
            "verify_output": verify.stdout + verify.stderr,
        }
    manifest_data = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    performance = manifest_data.get("performance", {})
    return {
        "symbol": symbol,
        "base_period": base,
        "higher_period": higher,
        "status": "PASS",
        "output_dir": str(output_dir.resolve()),
        "bars_hash": manifest_data.get("bars_hash"),
        "total_return": performance.get("total_return"),
        "max_drawdown_pct": performance.get("max_drawdown_pct"),
        "sharpe_ratio": performance.get("sharpe_ratio"),
        "win_rate": performance.get("win_rate"),
        "trade_count": manifest_data.get("trade_count"),
    }


def main() -> int:
    args = _parse_args()
    manifest = Path(args.dataset_manifest).expanduser().resolve()
    if not manifest.is_file():
        raise RuntimeError(f"Dataset manifest not found: {manifest}")
    output_root = Path(args.output_root).expanduser().resolve()
    jobs = [
        (symbol.upper(), base, higher)
        for symbol in args.symbols
        for base in args.base_periods
        for higher in args.higher_periods
        if validate_period_pair(base, higher)
    ]
    if not jobs:
        raise RuntimeError("No valid MTF combinations")

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {
            pool.submit(_run_one, symbol, base, higher, manifest, output_root): (symbol, base, higher)
            for symbol, base, higher in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=True))

    results.sort(key=lambda row: (row["symbol"], row["base_period"], row["higher_period"]))
    summary_path = output_root / "matrix_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    failed = [row for row in results if row["status"] != "PASS"]
    print(f"[MATRIX] {len(results) - len(failed)}/{len(results)} PASS -> {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
