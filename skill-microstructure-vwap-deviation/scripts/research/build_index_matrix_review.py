"""Build governed comparison artifacts from completed index MTF runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from research.project_paths import RUNS_ROOT

ROOT = RUNS_ROOT
REQUIRED = ("equity_curve.csv", "equity_curve.png", "trades.csv", "report.md", "run_manifest.json")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summaries",
        nargs="+",
        default=[
            str(RUNS_ROOT / "index_mtf" / "formal_5m" / "matrix_summary.json"),
            str(RUNS_ROOT / "index_mtf" / "formal_15m" / "matrix_summary.json"),
            str(RUNS_ROOT / "index_mtf" / "formal_30m" / "matrix_summary.json"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        default=str(RUNS_ROOT / "index_mtf" / "comparison"),
    )
    return parser.parse_args()


def _load_rows(summary_paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for summary_path in summary_paths:
        path = Path(summary_path).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"Summary not found: {path}")
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("status") != "PASS":
                continue
            run_dir = Path(row["output_dir"]).expanduser().resolve()
            missing = [name for name in REQUIRED if not (run_dir / name).is_file()]
            if missing:
                raise RuntimeError(f"Formal artifacts missing in {run_dir}: {missing}")
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 2 or manifest.get("data_mode") != "frozen":
                raise RuntimeError(f"Non-schema-v2 frozen run: {run_dir}")
            if not manifest.get("frozen_input_used"):
                raise RuntimeError(f"Run does not declare frozen input: {run_dir}")
            subset = json.loads(
                (run_dir / "frozen_dataset" / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            sources = {source["source_id"]: source for source in subset["sources"]}
            base_id, higher_id = manifest["source_ids"]
            if base_id not in sources or higher_id not in sources:
                raise RuntimeError(f"Subset manifest/source mismatch: {run_dir}")
            row = dict(row)
            row.update(
                {
                    "run_dir": str(run_dir),
                    "strategy_file_hash": manifest["strategy_file_hash"],
                    "base_bars_hash": sources[base_id]["bars_hash"],
                    "higher_bars_hash": sources[higher_id]["bars_hash"],
                    "pipeline_id": subset["pipeline_id"],
                    "adjustment_mode": subset["adjustment_mode"],
                    "round_trades": int(row.get("trade_count") or 0) // 2,
                }
            )
            rows.append(row)
    if not rows:
        raise RuntimeError("No PASS formal runs found")
    return rows


def _validate_comparison(rows: list[dict]) -> None:
    strategy_hashes = {row["strategy_file_hash"] for row in rows}
    if len(strategy_hashes) != 1:
        raise RuntimeError("Comparison mixes strategy file hashes")
    base_hashes: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["symbol"], row["base_period"])
        previous = base_hashes.setdefault(key, row["base_bars_hash"])
        if previous != row["base_bars_hash"]:
            raise RuntimeError(f"Base bars differ within comparison group: {key}")


def _read_curves(rows: list[dict]) -> pd.DataFrame:
    curves: list[pd.Series] = []
    for row in rows:
        frame = pd.read_csv(Path(row["run_dir"]) / "equity_curve.csv", parse_dates=["datetime"])
        if frame.empty or "equity" not in frame:
            raise RuntimeError(f"Empty equity curve: {row['run_dir']}")
        hold_suffix = f"_hold{row['max_hold_minutes']}m" if row.get("max_hold_minutes") else ""
        threshold_suffix = ""
        if row.get("entry_z") is not None and row.get("exit_z") is not None:
            threshold_suffix = f"_entry{row['entry_z']:g}_exit{row['exit_z']:g}"
        key = f"{row['symbol']}_{row['base_period']}_{row['higher_period']}{hold_suffix}{threshold_suffix}"
        series = frame.set_index("datetime")["equity"].rename(key)
        curves.append(series)
    return pd.concat(curves, axis=1).sort_index().rename_axis("datetime").reset_index()


def _best_by_symbol(rows: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for symbol in sorted({row["symbol"] for row in rows}):
        candidates = [row for row in rows if row["symbol"] == symbol]
        active = [row for row in candidates if int(row.get("round_trades") or 0) > 0]
        if active:
            candidates = active
        result[symbol] = max(
            candidates,
            key=lambda row: (
                float(row.get("sharpe_ratio") or 0.0),
                float(row.get("total_return") or 0.0),
                -float(row.get("max_drawdown_pct") or 0.0),
            ),
        )
    return result


def _write_plot(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    best = _best_by_symbol(rows)
    figure, axis = plt.subplots(figsize=(13, 6))
    for symbol, row in sorted(best.items()):
        frame = pd.read_csv(Path(row["run_dir"]) / "equity_curve.csv", parse_dates=["datetime"])
        hold_suffix = f" hold={row['max_hold_minutes']}m" if row.get("max_hold_minutes") else ""
        threshold_suffix = ""
        if row.get("entry_z") is not None and row.get("exit_z") is not None:
            threshold_suffix = f" entry={row['entry_z']:g} exit={row['exit_z']:g}"
        axis.plot(
            frame["datetime"],
            frame["equity"],
            linewidth=1.2,
            label=f"{symbol} {row['base_period']}->{row['higher_period']}{hold_suffix}{threshold_suffix}",
        )
    axis.set_title("Index VWAP MTF representative formal equity curves")
    axis.set_xlabel("datetime")
    axis.set_ylabel("equity")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _write_report(rows: list[dict], output: Path, master_manifest: Path) -> dict[str, dict]:
    best = _best_by_symbol(rows)
    lines = [
        "# IM/IF/IC VWAP 偏离交易：多周期正式验证复盘",
        "",
        "## 数据与比较口径",
        "",
        f"- 冻结数据清单：`{master_manifest}`",
        "- 正式输入：SSQuant 官方数据管线、schema-v2 冻结 CSV、adjust_type_1。",
        "- 区间：2024-01-02 至 2026-04-30。",
        "- 所有运行均为下一根 bar 开盘成交，手续费和滑点沿用 SSQuant 配置。",
        "- 同一品种/基础周期内，基础周期 bars_hash 已核对一致；高周期是有意变化的实验变量，因此各运行的组合 bars_hash 不同。",
        "",
        "## 运行结果",
        "",
        "| 品种 | 基础周期 | 高周期 | 持仓上限 min | entry_z | exit_z | 收益率 % | 最大回撤 % | Sharpe | 胜率 % | 完整交易 | 状态 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda item: (item["symbol"], item["base_period"], item["higher_period"])):
        lines.append(
            f"| {row['symbol']} | {row['base_period']} | {row['higher_period']} | "
            f"{row.get('max_hold_minutes', '-')} | {row.get('entry_z', '-')} | {row.get('exit_z', '-')} | "
            f"{_fmt(row.get('total_return'))} | {_fmt(row.get('max_drawdown_pct'))} | "
            f"{_fmt(row.get('sharpe_ratio'))} | {_fmt(row.get('win_rate'))} | "
            f"{row.get('round_trades', 0)} | PASS |"
        )
    lines.extend(["", "## 品种结论", ""])
    for symbol, row in sorted(best.items()):
        sharpe = float(row.get("sharpe_ratio") or 0.0)
        rounds = int(row.get("round_trades") or 0)
        if sharpe > 0 and rounds >= 10:
            conclusion = "可作为候选组合，仍需滚动样本外和成本敏感性验证。"
        elif sharpe > 0:
            conclusion = "收益为正但交易样本不足，暂不能作为稳定结论。"
        else:
            conclusion = "当前参数下未形成可接受的风险调整收益，暂不实盘。"
        lines.append(
            f"- **{symbol}**：代表组合 `{row['base_period']} -> {row['higher_period']}`，"
            f"收益 {_fmt(row.get('total_return'))}%、回撤 {_fmt(row.get('max_drawdown_pct'))}%、"
            f"Sharpe {_fmt(row.get('sharpe_ratio'))}、约 {rounds} 次完整交易；{conclusion}"
        )
    lines.extend(
        [
            "",
            "## 优化建议",
            "",
            "1. 执行周期优先保留 5m；15m 的交易数量明显下降，30m 全部零交易，不应直接作为当前策略执行周期。",
            "2. IM 优先验证 5m -> 120m 与 5m -> 90m；IC 优先验证 5m -> 30m 与 5m -> 60m；IF 当前没有正 Sharpe 组合，应先做趋势过滤和成本敏感性分析。",
            "3. 不把 15m/30m 的少交易或零交易结果当作有效优化证据；下一轮应使用滚动训练/验证区间，并加入交易次数下限。",
            "4. 参数优化先固定基础周期和高周期，再做 entry_z、exit_z、max_hold_bars 的小网格，避免同时优化周期和参数造成选择偏差。",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return best


def main() -> int:
    args = _args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.summaries)
    _validate_comparison(rows)
    curves = _read_curves(rows)
    curves.to_csv(output_dir / "compare_curve.csv", index=False, lineterminator="\n")
    _write_plot(rows, output_dir / "compare_equity.png")
    master_manifest = (ROOT / "index_mtf" / "frozen_dataset" / "dataset_manifest.json").resolve()
    best = _write_report(rows, output_dir / "compare_summary.md", master_manifest)
    manifest = {
        "schema_version": 2,
        "comparison_type": "formal-matrix-comparison",
        "input_summaries": [str(Path(path).expanduser().resolve()) for path in args.summaries],
        "run_count": len(rows),
        "symbols": sorted({row["symbol"] for row in rows}),
        "base_periods": sorted({row["base_period"] for row in rows}),
        "higher_periods": sorted({row["higher_period"] for row in rows}),
        "hold_minutes": sorted({row["max_hold_minutes"] for row in rows if row.get("max_hold_minutes")}),
        "strategy_file_hash": rows[0]["strategy_file_hash"],
        "master_dataset_manifest": str(master_manifest),
        "pipeline_id": rows[0]["pipeline_id"],
        "adjustment_mode": rows[0]["adjustment_mode"],
        "comparison_note": "Base bars are identical within each symbol/base-period group; higher bars intentionally differ.",
        "best_by_symbol": {
            symbol: {
                "base_period": row["base_period"],
                "higher_period": row["higher_period"],
                "bars_hash": row["bars_hash"],
            }
            for symbol, row in best.items()
        },
        "artifact_files": ["compare_curve.csv", "compare_equity.png", "compare_summary.md", "compare_manifest.json"],
    }
    (output_dir / "compare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "run_count": len(rows), "output_dir": str(output_dir)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
