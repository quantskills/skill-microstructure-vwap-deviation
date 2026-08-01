# -*- coding: utf-8 -*-
"""Run the VWAP-deviation strategy through the official SSQuant runner."""

import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


STRATEGY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STRATEGY_DIR.parent
CODEX_SKILL_ROOT = Path.home() / ".codex" / "skills" / "ssquant-backtest"
ENGINE_ROOT = Path(r"D:\代码\Quant&AI\ssquant-main\ssquant")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))
if str(CODEX_SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEX_SKILL_ROOT))
from research.project_paths import RUNS_ROOT

RUNS_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(RUNS_ROOT)
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

# Reports are mandatory for formal runs.
os.environ.pop("NO_VISUALIZATION", None)
os.environ.pop("NO_CONSOLE_LOG", None)

from shared.preflight_check import run as _preflight
from shared.data_fallback import inject, verify_inject_active
from ssquant.config.trading_config import RunMode, get_config
from ssquant.backtest.unified_runner import UnifiedStrategyRunner


STRATEGY_MODULE = "vwap_deviation_strategy"
STRATEGY_FILE = STRATEGY_DIR / f"{STRATEGY_MODULE}.py"
_preflight(str(STRATEGY_FILE), __file__)
strat = importlib.import_module(STRATEGY_MODULE)

STRATEGY_VERSION = "1.0.0"
ADJUST_TYPE = "1"
DEFAULT_START = "2024-01-02"
DEFAULT_END = "2026-04-30"
DEFAULT_PERIOD = "5m"
DEFAULT_CAPITAL = 300000.0
DEFAULT_SLIPPAGE_TICKS = 1


def _product_prefix(symbol):
    letters = ""
    for ch in symbol.upper():
        if ch.isalpha():
            letters += ch
        else:
            break
    return letters


def _parse_args():
    parser = argparse.ArgumentParser(description="SSQuant VWAP deviation backtest")
    parser.add_argument("--symbol", default="IM888")
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--kline-period", default=DEFAULT_PERIOD)
    parser.add_argument("--output-dir", default=str(RUNS_ROOT / "single_symbol" / "baseline"))
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--slippage-ticks", type=int, default=DEFAULT_SLIPPAGE_TICKS)
    parser.add_argument("--vwap-minutes", type=int, default=30)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--max-volume", type=int, default=1)
    return parser.parse_args()


def _canonicalize_bars(df):
    if df is None or len(df) == 0:
        raise RuntimeError("SSQuant returned no bars; formal run stopped")
    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        for candidate in ("datetime", "datetime_local", "date", "time"):
            if candidate in work.columns:
                work[candidate] = pd.to_datetime(work[candidate], errors="coerce")
                work = work.set_index(candidate)
                break
    if not isinstance(work.index, pd.DatetimeIndex):
        raise RuntimeError("SSQuant bars have no datetime index")
    work.index = pd.to_datetime(work.index, errors="coerce")
    work = work.loc[~work.index.isna()].sort_index()
    columns = [
        c for c in ("open", "high", "low", "close", "volume", "amount", "openint", "_adjust_factor")
        if c in work.columns
    ]
    if not {"open", "high", "low", "close", "volume"}.issubset(columns):
        raise RuntimeError(f"SSQuant bars miss required OHLCV columns: {list(work.columns)}")
    frozen = work[columns].copy()
    frozen.insert(0, "datetime", frozen.index.strftime("%Y-%m-%dT%H:%M:%S"))
    frozen = frozen.reset_index(drop=True)
    return frozen


def _freeze_dataset(symbol, start_date, end_date, kline_period, output_dir):
    """Fetch through the injected official entry point and hash the exact bars."""
    import ssquant.data.api_data_fetcher as api_data_fetcher

    df = api_data_fetcher.get_futures_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        kline_period=kline_period,
        adjust_type=ADJUST_TYPE,
        use_cache=True,
        save_data=True,
    )
    source_markers = [str(df.attrs.get(k, "")).lower() for k in ("source", "data_source", "_source")]
    if any("tqsdk" in marker for marker in source_markers):
        raise RuntimeError("data_fallback selected TqSdk; SSQuant-only formal run stopped")
    frozen = _canonicalize_bars(df)
    payload = frozen.to_csv(index=False, lineterminator="\n", float_format="%.15g").encode("utf-8")
    bars_hash = hashlib.sha256(payload).hexdigest()
    freeze_dir = output_dir / "frozen_dataset"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    bars_path = freeze_dir / "bars.csv"
    bars_path.write_bytes(payload)
    manifest = {
        "symbol": symbol,
        "kline_period": kline_period,
        "start_date": start_date,
        "end_date": end_date,
        "adjust_type": ADJUST_TYPE,
        "adjustment_mode": "ssquant_adjust_type_1",
        "data_source": "ssquant",
        "data_pipeline": "data_fallback.inject -> SSQuant get_futures_data",
        "rows": int(len(frozen)),
        "first_datetime": frozen["datetime"].iloc[0],
        "last_datetime": frozen["datetime"].iloc[-1],
        "bars_sha256": bars_hash,
        "bars_file": str(bars_path),
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
    }
    (freeze_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[FROZEN] {symbol} rows={len(frozen)} hash={bars_hash[:16]}... range={manifest['first_datetime']}~{manifest['last_datetime']}")
    return manifest


def _extract_trades(results):
    rows = []
    for key, value in results.items():
        if isinstance(value, dict) and isinstance(value.get("trades"), list):
            for trade in value["trades"]:
                row = dict(trade)
                row["_ds"] = value.get("symbol", key)
                rows.append(row)
    return rows


def _extract_equity(results):
    candidates = []
    top = results.get("equity_curve")
    if isinstance(top, pd.Series):
        candidates.append(top)
    for value in results.values():
        if isinstance(value, dict) and isinstance(value.get("equity_curve"), pd.Series):
            candidates.append(value["equity_curve"])
    if not candidates:
        raise RuntimeError("SSQuant returned no equity curve; formal artifact generation stopped")
    return candidates[0].dropna()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _contract_params(config):
    keys = (
        "contract_multiplier",
        "price_tick",
        "margin_rate",
        "commission",
        "commission_close",
        "commission_close_today",
        "commission_per_lot",
        "commission_close_per_lot",
        "commission_close_today_per_lot",
        "slippage_ticks",
    )
    return {key: config.get(key) for key in keys if key in config}


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Injection and verification must happen before any official data access.
    inject()
    verify_inject_active()
    dataset = _freeze_dataset(args.symbol, args.start_date, args.end_date, args.kline_period, output_dir)

    strategy_params = {
        "vwap_minutes": args.vwap_minutes,
        "bar_minutes": 5,
        "entry_z": args.entry_z,
        "exit_z": args.exit_z,
        "max_volume": args.max_volume,
        "symbol_prefix": _product_prefix(args.symbol),
    }
    config = get_config(
        RunMode.BACKTEST,
        symbol=args.symbol,
        kline_period=args.kline_period,
        adjust_type=ADJUST_TYPE,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        slippage_ticks=args.slippage_ticks,
    )
    contract_params = _contract_params(config)
    runner = UnifiedStrategyRunner(mode=RunMode.BACKTEST)
    runner.set_config(config)
    results = runner.run(
        strategy=strat.handle_bar,
        initialize=strat.initialize,
        strategy_params=strategy_params,
    )

    html_report = results.get("html_report_path")
    text_report = results.get("report_path")
    report_path = html_report or text_report
    if not report_path or not os.path.exists(report_path):
        raise RuntimeError("SSQuant report was not generated")

    trades = _extract_trades(results)
    trades_path = output_dir / "trades.csv"
    pd.DataFrame(trades).to_csv(trades_path, index=False, encoding="utf-8-sig")

    equity = _extract_equity(results)
    equity_frame = pd.DataFrame({"datetime": equity.index.astype(str), "equity": equity.values})
    equity_csv = output_dir / "equity_curve.csv"
    equity_frame.to_csv(equity_csv, index=False, encoding="utf-8-sig")
    equity_png = output_dir / "equity_curve.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pd.to_datetime(equity.index), equity.values, linewidth=1.0)
    ax.set_title(f"{args.symbol} VWAP deviation equity curve")
    ax.set_xlabel("datetime")
    ax.set_ylabel("equity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(equity_png, dpi=150)
    plt.close(fig)

    performance = results.get("performance", {})
    report_md = output_dir / "report.md"
    report_md.write_text(
        "\n".join(
            [
                f"# VWAP Deviation Backtest: {args.symbol}",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| Data source | SSQuant official runner via `data_fallback.inject` |",
                f"| Period | {args.start_date} to {args.end_date} |",
                f"| K-line | {args.kline_period} |",
                f"| Adjustment | `adjust_type={ADJUST_TYPE}` |",
                f"| Frozen rows | {dataset['rows']} |",
                f"| Bars SHA-256 | `{dataset['bars_sha256']}` |",
                f"| Trades | {len(trades)} |",
                f"| Strategy version | {STRATEGY_VERSION} |",
                "",
                "## Parameters",
                "",
                "```json",
                json.dumps(_json_safe(strategy_params), ensure_ascii=False, indent=2),
                "```",
                "",
                "## Execution Costs",
                "",
                "```json",
                json.dumps(_json_safe(contract_params), ensure_ascii=False, indent=2),
                "```",
                "",
                "## Performance",
                "",
                "```json",
                json.dumps(_json_safe(performance), ensure_ascii=False, indent=2),
                "```",
                "",
                f"Framework report: `{report_path}`",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "formal": True,
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "kline_period": args.kline_period,
        "adjust_type": ADJUST_TYPE,
        "adjustment_mode": "ssquant_adjust_type_1",
        "data_source": "ssquant",
        "data_pipeline": "SSQuant official runner + data_fallback.inject + verify_inject_active",
        "fallback_forbidden": True,
        "strategy": "microstructure-vwap-deviation",
        "strategy_version": STRATEGY_VERSION,
        "strategy_params": _json_safe(strategy_params),
        "contract_params": _json_safe(contract_params),
        "execution_params": {
            "initial_capital": args.initial_capital,
            "slippage_ticks": args.slippage_ticks,
            "quantity_cap": args.max_volume,
            "fee_model": "SSQuant auto_params / contract metadata",
            "slippage_model": "slippage_ticks * price_tick * volume * contract_multiplier",
        },
        "dataset": dataset,
        "artifacts": {
            "equity_curve_csv": str(equity_csv),
            "equity_curve_png": str(equity_png),
            "trades_csv": str(trades_path),
            "report_md": str(report_md),
            "framework_report": str(report_path),
        },
        "performance": _json_safe(performance),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "time_range": {"start": args.start_date, "end": args.end_date},
        "time_frame": args.kline_period,
        "code_version": STRATEGY_VERSION,
        "equity_chart": str(equity_png),
        "trade_count": len(trades),
        "performance": _json_safe(performance),
    }
    (output_dir / "backtest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] {args.symbol} formal SSQuant backtest complete")
    print(f"[ARTIFACTS] {output_dir}")
    return results


if __name__ == "__main__":
    main()
