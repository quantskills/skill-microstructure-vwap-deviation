"""Build governed comparison artifacts for baseline vs multifactor trend runs."""

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

ROOT = RUNS_ROOT
OUTPUT_DIR = RUNS_ROOT / "index_mtf" / "comparison_trend_multifactor"

RUNS = {
    "IM888": {
        "baseline": ROOT / "index_mtf/formal_entry_exit/hold_30m_entry_2.25_exit_0.5/IM888/5m_120m",
        "multifactor": ROOT / "index_mtf/formal_trend_multifactor/hold_30m_entry_2.25_exit_0.5/IM888/5m_120m",
    },
    "IC888": {
        "baseline": ROOT / "index_mtf/formal_entry_exit/hold_60m_entry_2.5_exit_0.25/IC888/5m_30m",
        "multifactor": ROOT / "index_mtf/formal_trend_multifactor/hold_60m_entry_2.5_exit_0.25/IC888/5m_30m",
    },
    "IF888": {
        "baseline": ROOT / "index_mtf/formal_entry_exit/hold_30m_entry_2_exit_0.25/IF888/5m_30m",
        "multifactor": ROOT / "index_mtf/formal_trend_multifactor/hold_30m_entry_2_exit_0.25/IF888/5m_30m",
    },
}


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    curve = pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["datetime"])
    curve = curve.sort_values("datetime").drop_duplicates("datetime")
    curve["net_value"] = curve["equity"] / float(curve["equity"].iloc[0])
    return curve[["datetime", "net_value"]], manifest


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_frames = []
    rows = []
    for symbol, variants in RUNS.items():
        loaded = {name: load_run(path) for name, path in variants.items()}
        baseline_curve, baseline_manifest = loaded["baseline"]
        multifactor_curve, multifactor_manifest = loaded["multifactor"]
        if baseline_manifest["bars_hash"] != multifactor_manifest["bars_hash"]:
            raise ValueError(f"bars hash mismatch for {symbol}")
        if baseline_manifest["initial_capital"] != multifactor_manifest["initial_capital"]:
            raise ValueError(f"initial capital mismatch for {symbol}")

        frame = baseline_curve.rename(columns={"net_value": "baseline_net_value"}).merge(
            multifactor_curve.rename(columns={"net_value": "multifactor_net_value"}),
            on="datetime",
            how="outer",
        )
        frame["symbol"] = symbol
        comparison_frames.append(frame[["datetime", "symbol", "baseline_net_value", "multifactor_net_value"]])

        base_perf = baseline_manifest["performance"]
        multi_perf = multifactor_manifest["performance"]
        rows.append(
            {
                "symbol": symbol,
                "baseline_run": str(variants["baseline"].resolve()),
                "multifactor_run": str(variants["multifactor"].resolve()),
                "bars_hash": baseline_manifest["bars_hash"],
                "baseline_version": baseline_manifest["strategy_version"],
                "multifactor_version": multifactor_manifest["strategy_version"],
                "baseline_return_pct": base_perf["total_return"],
                "multifactor_return_pct": multi_perf["total_return"],
                "return_delta_pct": multi_perf["total_return"] - base_perf["total_return"],
                "baseline_max_drawdown_pct": base_perf["max_drawdown_pct"],
                "multifactor_max_drawdown_pct": multi_perf["max_drawdown_pct"],
                "drawdown_delta_pct": multi_perf["max_drawdown_pct"] - base_perf["max_drawdown_pct"],
                "baseline_sharpe": base_perf["sharpe_ratio"],
                "multifactor_sharpe": multi_perf["sharpe_ratio"],
                "sharpe_delta": multi_perf["sharpe_ratio"] - base_perf["sharpe_ratio"],
                "baseline_round_trades": int(baseline_manifest["trade_count"]) // 2,
                "multifactor_round_trades": int(multifactor_manifest["trade_count"]) // 2,
            }
        )

    comparison = pd.concat(comparison_frames, ignore_index=True)
    comparison.to_csv(OUTPUT_DIR / "compare_curve.csv", index=False)
    compare_manifest = {
        "schema_version": 1,
        "comparison_type": "baseline_vs_multifactor_trend",
        "data_identity_verified": True,
        "time_range": {"start": "2024-01-02", "end": "2026-04-30"},
        "comparison_curve": str((OUTPUT_DIR / "compare_curve.csv").resolve()),
        "comparison_plot": str((OUTPUT_DIR / "compare_equity.png").resolve()),
        "runs": rows,
    }
    (OUTPUT_DIR / "compare_manifest.json").write_text(
        json.dumps(compare_manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False, constrained_layout=True)
    colors = {"baseline_net_value": "#7f7f7f", "multifactor_net_value": "#1f77b4"}
    for axis, symbol in zip(axes, RUNS):
        frame = comparison[comparison["symbol"] == symbol]
        axis.plot(frame["datetime"], frame["baseline_net_value"], color=colors["baseline_net_value"], label="baseline")
        axis.plot(frame["datetime"], frame["multifactor_net_value"], color=colors["multifactor_net_value"], label="multifactor trend")
        axis.axhline(1.0, color="#555555", linewidth=0.7, linestyle="--")
        axis.set_title(symbol)
        axis.set_ylabel("Net value")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper left")
    axes[-1].set_xlabel("Datetime")
    fig.suptitle("Baseline vs Multifactor Trend: IM888 / IC888 / IF888")
    fig.savefig(OUTPUT_DIR / "compare_equity.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Baseline vs Multifactor Trend Comparison",
        "",
        "- Input window: `2024-01-02` to `2026-04-30`",
        "- Base and higher timeframe, entry/exit thresholds, hold limits, execution model and frozen data are held constant per symbol.",
        "- Baseline and multifactor runs use the same bars hash per symbol; strategy version is intentionally different.",
        "- Net value curves are normalized by each run's first recorded equity value.",
        "",
        "| Symbol | Baseline return % | Multifactor return % | Return delta % | Baseline DD % | Multifactor DD % | DD delta % | Baseline Sharpe | Multifactor Sharpe | Sharpe delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['symbol']} | {row['baseline_return_pct']:.2f} | {row['multifactor_return_pct']:.2f} | "
            f"{row['return_delta_pct']:.2f} | {row['baseline_max_drawdown_pct']:.2f} | "
            f"{row['multifactor_max_drawdown_pct']:.2f} | {row['drawdown_delta_pct']:.2f} | "
            f"{row['baseline_sharpe']:.2f} | {row['multifactor_sharpe']:.2f} | {row['sharpe_delta']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Trend parameters",
            "",
            "`fast_bars=3`, `slow_bars=8`, `slope_bars=3`, `efficiency_threshold=0.35`.",
            "",
            "## Artifacts",
            "",
            "- `compare_curve.csv`",
            "- `compare_equity.png`",
            "- `compare_manifest.json`",
        ]
    )
    (OUTPUT_DIR / "compare_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_DIR / "compare_equity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
