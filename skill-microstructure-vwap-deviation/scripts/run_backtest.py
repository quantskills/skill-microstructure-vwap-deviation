"""CLI for the standalone VWAP deviation research run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backtest import BacktestConfig, run_backtest
from data_loader import freeze_dataset, load_minute_bars, normalize_minute_bars
from vwap_strategy import VWAPConfig
from research.project_paths import RUNS_ROOT


def make_synthetic_bars(periods: int = 600, seed: int = 42) -> pd.DataFrame:
    """Create deterministic pipeline-test bars; never use for research conclusions."""

    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-05 09:30", periods=periods, freq="min")
    base = 100 + np.cumsum(rng.normal(0, 0.05, periods))
    oscillation = 1.8 * np.sin(np.arange(periods) / 18.0)
    close = base + oscillation
    open_ = close + rng.normal(0, 0.03, periods)
    high = np.maximum(open_, close) + rng.uniform(0, 0.08, periods)
    low = np.minimum(open_, close) - rng.uniform(0, 0.08, periods)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100, 1000, periods),
        },
        index=index,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def write_artifacts(result: dict, output_dir: str | Path, manifest: dict) -> dict:
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    equity_path = target / "equity_curve.csv"
    trades_path = target / "trades.csv"
    report_path = target / "report.md"
    plot_path = target / "equity_curve.png"
    manifest_path = target / "run_manifest.json"

    result["equity_curve"].reset_index().to_csv(equity_path, index=False)
    result["trades"].to_csv(trades_path, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10, 4.8))
    curve = result["equity_curve"]
    if len(curve):
        axis.plot(curve.index, curve["equity"], color="#1f77b4", linewidth=1.2)
    axis.set_title("Minute VWAP Deviation Equity Curve")
    axis.set_xlabel("Datetime")
    axis.set_ylabel("Equity")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary = result["summary"]
    report_lines = [
        "# Minute VWAP Deviation Run",
        "",
        f"- Mode: `{manifest['backtest_mode']}`",
        f"- Symbol: `{manifest['symbol']}`",
        f"- Data source: `{manifest['data_source']}`",
        f"- Adjustment mode: `{manifest['adjustment_mode']}`",
        f"- Bars hash: `{manifest['bars_hash']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        report_lines.append(f"| {key} | {value} |")
    report_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Equity curve: `{equity_path}`",
            f"- Trades: `{trades_path}`",
            f"- Plot: `{plot_path}`",
        ]
    )
    if manifest.get("debug_only"):
        report_lines.extend(["", "> This run is debug-only and must not be used as a research conclusion."])
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = dict(manifest)
    manifest.update(
        {
            "artifacts": {
                "equity_curve_csv": str(equity_path),
                "equity_curve_png": str(plot_path),
                "trades_csv": str(trades_path),
                "report_md": str(report_path),
                "run_manifest_json": str(manifest_path),
            },
            "summary": summary,
            "strategy_config": result["config"]["strategy"],
            "backtest_config": result["config"]["backtest"],
        }
    )
    manifest_path.write_text(json.dumps(_jsonable(manifest), ensure_ascii=True, indent=2), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--symbol", default="SYNTH")
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default="20260131")
    parser.add_argument("--frequency", choices=("1m", "5m", "15m", "60m"), default="1m")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RUNS_ROOT / "standalone")
    parser.add_argument("--vwap-minutes", type=int, choices=(30, 60), default=30)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--no-trend-filter", action="store_true")
    parser.add_argument("--session-end", default="15:00")
    parser.add_argument("--tail-lock-minutes", type=int, default=30)
    parser.add_argument("--adjustment-mode", choices=("official", "raw", "synthetic"), default="official")
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=300_000.0,
        help="Initial account cash used by the standalone research harness.",
    )
    parser.add_argument(
        "--quantity",
        type=int,
        default=1,
        help="Units opened per signal.",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0003,
        help="Commission rate applied to entry and exit notional.",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=1.0,
        help="Slippage in basis points applied against the trade direction.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.input_csv and args.synthetic:
        raise SystemExit("--input-csv and --synthetic are mutually exclusive")
    if args.input_csv:
        bars = normalize_minute_bars(pd.read_csv(args.input_csv))
        data_source = "local_csv"
        debug_only = args.adjustment_mode != "official"
    elif args.synthetic:
        bars = make_synthetic_bars()
        data_source = "synthetic"
        debug_only = True
        args.adjustment_mode = "synthetic"
    else:
        bars = load_minute_bars(
            args.symbol,
            args.start_date,
            args.end_date,
            frequency=args.frequency,
        )
        data_source = bars.attrs.get("data_source", "unknown")
        debug_only = False

    if bars.empty:
        raise RuntimeError("no bars available; no backtest artifacts were published")
    frozen = freeze_dataset(bars, Path(args.output_dir).resolve() / "frozen_dataset", args.symbol, args.adjustment_mode)
    strategy_config = VWAPConfig(
        vwap_minutes=args.vwap_minutes,
        bar_minutes=int(args.frequency.rstrip("m")),
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        trend_filter=not args.no_trend_filter,
        session_end=args.session_end,
        tail_lock_minutes=args.tail_lock_minutes,
    )
    result = run_backtest(
        bars,
        strategy_config,
        BacktestConfig(
            initial_cash=args.initial_cash,
            quantity=args.quantity,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
        ),
    )
    manifest = {
        "backtest_mode": "standalone_research",
        "debug_only": debug_only,
        "symbol": args.symbol,
        "frequency": args.frequency,
        "data_source": data_source,
        "adjustment_mode": args.adjustment_mode,
        "bars_hash": frozen["bars_hash"],
        "dataset_manifest": str(Path(frozen["bars_path"]).with_name("dataset_manifest.json")),
    }
    final = write_artifacts(result, args.output_dir, manifest)
    print(json.dumps(_jsonable(final), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
