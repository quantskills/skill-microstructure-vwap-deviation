"""Plot the previously selected representative equity curves for IM/IF/IC."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from research.project_paths import RUNS_ROOT

PROJECT_ROOT = RUNS_ROOT
OUTPUT_DIR = RUNS_ROOT / "index_mtf" / "best_equity_curves"

# These are the representative highest-return combinations from the prior grid.
RUNS = {
    "IM888": PROJECT_ROOT
    / "index_mtf/formal_entry_exit/hold_30m_entry_2.25_exit_0.5/IM888/5m_120m",
    "IC888": PROJECT_ROOT
    / "index_mtf/formal_entry_exit/hold_60m_entry_2.5_exit_0.25/IC888/5m_30m",
    "IF888": PROJECT_ROOT
    / "index_mtf/formal_entry_exit/hold_30m_entry_2_exit_0.25/IF888/5m_30m",
}


def load_run(symbol: str, run_dir: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = run_dir / "run_manifest.json"
    curve_path = run_dir / "equity_curve.csv"
    if not manifest_path.exists() or not curve_path.exists():
        raise FileNotFoundError(f"missing governed artifacts for {symbol}: {run_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "backtest_summary.json").read_text(encoding="utf-8"))
    curve = pd.read_csv(curve_path, parse_dates=["datetime"])
    curve = curve.sort_values("datetime").drop_duplicates("datetime")
    initial_capital = float(manifest["initial_capital"])
    curve_start_equity = float(curve["equity"].iloc[0])
    if curve_start_equity <= 0:
        raise ValueError(f"non-positive starting equity for {manifest['symbol']}")
    # Formal SSQuant curves in this matrix start at 150,000 while the manifest
    # capital is 300,000; use the actual curve base so every line starts at 1.0.
    curve["net_value"] = curve["equity"] / curve_start_equity
    curve["symbol"] = symbol
    manifest["summary"] = summary
    return curve, manifest


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    curves: list[pd.DataFrame] = []
    records = []
    initial_capitals = set()
    for symbol, run_dir in RUNS.items():
        curve, manifest = load_run(symbol, run_dir)
        curves.append(curve[["datetime", "symbol", "equity", "net_value"]])
        manifest_initial_capital = float(manifest["initial_capital"])
        curve_start_equity = float(curve["equity"].iloc[0])
        initial_capitals.add(manifest_initial_capital)
        summary = manifest["performance"]
        records.append(
            {
                "symbol": symbol,
                "run_dir": str(run_dir),
                "bars_hash": manifest["bars_hash"],
                "start_date": manifest["start_date"],
                "end_date": manifest["end_date"],
                "manifest_initial_capital": manifest_initial_capital,
                "curve_start_equity": curve_start_equity,
                "base_period": manifest["base_period"],
                "higher_period": manifest["higher_period"],
                "max_hold_minutes": manifest["strategy_params"]["max_hold_minutes"],
                "entry_z": manifest["strategy_params"]["entry_z"],
                "exit_z": manifest["strategy_params"]["exit_z"],
                "total_return_pct": summary["total_return"],
                "max_drawdown_pct": summary["max_drawdown_pct"],
                "sharpe_ratio": summary["sharpe_ratio"],
                "trade_count": manifest["trade_count"],
            }
        )

    if len(initial_capitals) != 1:
        raise ValueError(f"selected runs use different initial capital: {initial_capitals}")

    long_curve = pd.concat(curves, ignore_index=True)
    long_curve.to_csv(OUTPUT_DIR / "best_equity_curves.csv", index=False)
    pd.DataFrame(records).to_json(
        OUTPUT_DIR / "plot_manifest.json", orient="records", force_ascii=True, indent=2
    )

    colors = {"IM888": "#1f77b4", "IC888": "#d62728", "IF888": "#2ca02c"}
    fig, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    for record in records:
        symbol = record["symbol"]
        curve = long_curve[long_curve["symbol"] == symbol]
        label = (
            f"{symbol} | return {record['total_return_pct']:.2f}% | "
            f"DD {record['max_drawdown_pct']:.2f}%"
        )
        axis.plot(
            curve["datetime"],
            curve["net_value"],
            color=colors[symbol],
            linewidth=1.1,
            label=label,
        )

    axis.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--")
    axis.set_title("Selected IM888 / IC888 / IF888 Equity Curves")
    axis.set_xlabel("Datetime")
    axis.set_ylabel("Normalized Net Value")
    axis.grid(alpha=0.22)
    axis.legend(loc="upper left", frameon=True)
    fig.savefig(OUTPUT_DIR / "best_equity_curves.png", dpi=180)
    plt.close(fig)

    print(OUTPUT_DIR / "best_equity_curves.png")
    print(OUTPUT_DIR / "best_equity_curves.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
