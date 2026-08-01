"""Formal frozen-data runner for index VWAP MTF validation."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODEX_SKILL_ROOT = Path.home() / ".codex" / "skills" / "ssquant-backtest"
sys.path.insert(0, str(CODEX_SKILL_ROOT))
sys.path.insert(1, str(PROJECT_ROOT / "scripts"))

from shared.formal_artifacts import write_formal_artifacts
from shared.frozen_data import load_dataset_manifest, sha256_file
from shared.preflight_check import run as _preflight
from shared.runtime_paths import find_project_root, resolve_engine_path
from research.index_matrix_contract import period_minutes, source_id, validate_period_pair
from research.formal_cost_adjustment import apply_close_today_costs
from research.project_paths import RUNS_ROOT

RUNS_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(RUNS_ROOT)


PROJECT_ROOT = find_project_root(PROJECT_ROOT)
ENGINE_PATH = resolve_engine_path(PROJECT_ROOT)
if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

STRATEGY_MODULE = "vwap_deviation_optimized_strategy"
STRATEGY_FILE = PROJECT_ROOT / "scripts" / f"{STRATEGY_MODULE}.py"
_preflight(str(STRATEGY_FILE), __file__)

# Formal mode reads only files already verified by shared.frozen_data;
# data_fallback is intentionally absent from this execution path.

from ssquant.backtest.unified_runner import UnifiedStrategyRunner
from ssquant.config.trading_config import RunMode, get_config

import importlib

strategy = importlib.import_module(STRATEGY_MODULE)

ADJUST_TYPE = "1"
STRATEGY_VERSION_BASE = "3.1.0-opening-failure-short"
DATA_MODE = os.environ.get("SSQUANT_DATA_MODE", "")
DATASET_MANIFEST_PATH = Path(os.environ.get("SSQUANT_DATASET_MANIFEST", "")).expanduser().resolve()
OUTPUT_DIR = Path(
    os.environ.get(
        "SSQUANT_OUTPUT_DIR",
        str(RUNS_ROOT / "index_mtf" / "formal_run"),
    )
).expanduser().resolve()
SYMBOL = os.environ.get("SSQUANT_SYMBOL", "IM888").upper()
BASE_PERIOD = os.environ.get("SSQUANT_BASE_PERIOD", "5m")
HIGHER_PERIOD = os.environ.get("SSQUANT_HIGHER_PERIOD", "60m")
START_DATE = os.environ.get("SSQUANT_START_DATE", "2024-01-02")
END_DATE = os.environ.get("SSQUANT_END_DATE", "2026-04-30")
INITIAL_CAPITAL = float(os.environ.get("SSQUANT_INITIAL_CAPITAL", "300000"))
USE_REAL_IM_PARAMS = os.environ.get("SSQUANT_USE_REAL_IM_PARAMS", "0") == "1"
STRATEGY_VERSION = (
    f"{STRATEGY_VERSION_BASE}-im-real-contract-costs"
    if USE_REAL_IM_PARAMS and SYMBOL == "IM888"
    else STRATEGY_VERSION_BASE
)


def _contract_overrides() -> dict:
    """Return only the explicitly requested IM overrides.

    Other symbols and ordinary formal runs continue to use SSQuant's automatic
    contract metadata.  The close-today field is retained in the config and
    manifest even though the current backtest result calculator does not apply
    it by trade date.
    """
    if not USE_REAL_IM_PARAMS:
        return {}
    if SYMBOL != "IM888":
        raise RuntimeError("SSQUANT_USE_REAL_IM_PARAMS=1 requires SSQUANT_SYMBOL=IM888")
    return {
        "contract_multiplier": 200,
        "price_tick": 0.2,
        "margin_rate": 0.12,
        "commission": 0.000023,
        "commission_close": 0.000023,
        "commission_close_today": 0.00023,
    }


def _load_source_frame(source: dict) -> pd.DataFrame:
    frame = pd.read_csv(source["file_path"], parse_dates=["datetime"])
    frame = frame.set_index("datetime").sort_index()
    if frame.empty:
        raise RuntimeError(f"Frozen source is empty: {source['source_id']}")
    return frame


def _select_sources(dataset: dict) -> list[dict]:
    if DATA_MODE != "frozen":
        raise RuntimeError("Formal index runs require SSQUANT_DATA_MODE=frozen")
    if not DATASET_MANIFEST_PATH.is_file():
        raise RuntimeError("SSQUANT_DATASET_MANIFEST is missing")
    if not validate_period_pair(BASE_PERIOD, HIGHER_PERIOD):
        raise RuntimeError(f"Invalid MTF period pair: {BASE_PERIOD} -> {HIGHER_PERIOD}")
    wanted = {
        source_id(SYMBOL, BASE_PERIOD, ADJUST_TYPE),
        source_id(SYMBOL, HIGHER_PERIOD, ADJUST_TYPE),
    }
    selected = [source for source in dataset["sources"] if source["source_id"] in wanted]
    if {source["source_id"] for source in selected} != wanted:
        raise RuntimeError(f"Frozen manifest is missing sources: {sorted(wanted)}")
    return sorted(selected, key=lambda item: period_minutes(item["kline_period"]))


def _install_frozen_fetch(selected: list[dict]):
    from ssquant.backtest.backtest_data import BacktestDataManager

    frames = {source["source_id"]: _load_source_frame(source) for source in selected}
    original_fetch = BacktestDataManager.fetch_data

    def fetch_frozen(_manager, symbols_and_periods, _symbol_configs, _base_config):
        requested = {
            source_id(item["symbol"], item["kline_period"], str(item["adjust_type"]))
            for item in symbols_and_periods
        }
        missing = requested - set(frames)
        if missing:
            raise RuntimeError(f"Frozen fetch requested unknown sources: {sorted(missing)}")
        return {key: frames[key].copy() for key in requested}

    BacktestDataManager.fetch_data = fetch_frozen
    return BacktestDataManager, original_fetch


def _write_subset_manifest(dataset: dict, selected: list[dict]) -> Path:
    path = OUTPUT_DIR / "frozen_dataset" / "dataset_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = dict(dataset)
    subset["bars_hash"] = _subset_hash(selected)
    subset["sources"] = selected
    subset["metadata"] = dict(dataset.get("metadata", {}))
    subset["metadata"]["subset_source_ids"] = [source["source_id"] for source in selected]
    path.write_text(json.dumps(subset, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return path


def _validate_formal_results(
    results: dict, selected: list[dict], runner: UnifiedStrategyRunner
) -> bool:
    expected = {source["source_id"]: source for source in selected}
    data_sources = getattr(getattr(runner, "backtester", None), "_last_multi_data_source", None)
    if data_sources is None:
        raise RuntimeError("SSQuant did not expose the completed multi-data-source state")
    reached = set()
    for data_source in data_sources.data_sources:
        key = source_id(data_source.symbol, data_source.kline_period, ADJUST_TYPE)
        if key not in expected:
            continue
        if data_source.data.empty or pd.Timestamp(data_source.data.index.max()) < pd.Timestamp(expected[key]["end"]):
            raise RuntimeError(f"Frozen source did not reach its end: {key}")
        reached.add(key)
    if reached != set(expected):
        raise RuntimeError(f"Completed engine sources differ from frozen sources: {sorted(reached)}")

    # The SSQuant result calculator omits data sources with no trades. The
    # higher timeframe is informational, so it is validated above through the
    # completed engine data state; the base source must have a result curve.
    base_key = source_id(SYMBOL, BASE_PERIOD, ADJUST_TYPE)
    matched = set()
    for key, result in results.items():
        if not isinstance(result, dict) or "equity_curve" not in result:
            continue
        symbol = str(result.get("symbol", SYMBOL)).upper()
        period = str(result.get("kline_period", result.get("period", "")))
        candidate = source_id(symbol, period, ADJUST_TYPE)
        if candidate not in expected:
            continue
        curve = result["equity_curve"]
        if not isinstance(curve, pd.Series) or curve.empty:
            raise RuntimeError(f"Empty equity curve for frozen source: {candidate}")
        actual_end = pd.Timestamp(curve.index.max())
        expected_end = pd.Timestamp(expected[candidate]["end"])
        if actual_end < expected_end:
            raise RuntimeError(
                f"Run stopped before frozen source end: {candidate}; "
                f"actual={actual_end}, expected={expected_end}"
            )
        matched.add(candidate)
    if base_key in matched:
        return False
    has_trades = any(
        isinstance(result, dict) and result.get("trades")
        for result in results.values()
    )
    if has_trades:
        raise RuntimeError(
            f"Formal results did not return the base source equity curve: {base_key}"
        )
    return True


def _inject_no_trade_result(results: dict, selected: list[dict], runner: UnifiedStrategyRunner) -> None:
    """Represent a valid engine run with no orders as a flat governed curve."""
    base_key = source_id(SYMBOL, BASE_PERIOD, ADJUST_TYPE)
    data_sources = runner.backtester._last_multi_data_source.data_sources
    base_data = next(
        data_source.data
        for data_source in data_sources
        if source_id(data_source.symbol, data_source.kline_period, ADJUST_TYPE) == base_key
    )
    results[base_key] = {
        "symbol": SYMBOL,
        "kline_period": BASE_PERIOD,
        "equity_curve": pd.Series(float(INITIAL_CAPITAL), index=base_data.index, dtype=float),
        "trades": [],
    }
    results["performance"] = {
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "total_return": 0.0,
        "annual_return": 0.0,
        "sharpe_ratio": 0.0,
        "win_rate": 0.0,
        "trade_stats": {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "profit_factor": 0.0,
        },
    }


def main() -> dict:
    dataset = load_dataset_manifest(DATASET_MANIFEST_PATH, formal=True)
    selected = _select_sources(dataset)
    subset_manifest_path = _write_subset_manifest(dataset, selected)
    base_source, higher_source = selected

    source_configs = [
        {
            "symbol": source["symbol"],
            "kline_period": source["kline_period"],
            "adjust_type": source["adjust_type"],
            "initial_capital": INITIAL_CAPITAL,
            **_contract_overrides(),
        }
        for source in selected
    ]
    config = get_config(
        RunMode.BACKTEST,
        data_sources=source_configs,
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        align_data=False,
        fill_method=None,
        use_cache=False,
        save_data=False,
        slippage_ticks=1,
    )

    strategy_params = {
        "vwap_minutes": int(os.environ.get("SSQUANT_VWAP_MINUTES", "60")),
        "local_vwap_minutes": int(os.environ.get("SSQUANT_LOCAL_VWAP_MINUTES", "30")),
        "bar_minutes": period_minutes(BASE_PERIOD),
        "entry_z": float(os.environ.get("SSQUANT_ENTRY_Z", "2.5")),
        "exit_z": float(os.environ.get("SSQUANT_EXIT_Z", "0.25")),
        "atr_window": int(os.environ.get("SSQUANT_ATR_WINDOW", "14")),
        "atr_stop_multiplier": float(os.environ.get("SSQUANT_ATR_STOP_MULTIPLIER", "1.5")),
        "max_hold_bars": int(os.environ.get("SSQUANT_MAX_HOLD_BARS", "6")),
        "max_hold_minutes": int(os.environ.get("SSQUANT_MAX_HOLD_MINUTES", "30")),
        "entry_confirmation_bars": int(os.environ.get("SSQUANT_ENTRY_CONFIRMATION_BARS", "1")),
        "enable_core_mean_reversion": os.environ.get(
            "SSQUANT_ENABLE_CORE_MEAN_REVERSION", "1"
        )
        == "1",
        "break_lock_minutes": int(os.environ.get("SSQUANT_BREAK_LOCK_MINUTES", "10")),
        "enable_trend_pullback": os.environ.get("SSQUANT_ENABLE_TREND_PULLBACK", "0") == "1",
        "enable_trend_breakout": os.environ.get("SSQUANT_ENABLE_TREND_BREAKOUT", "0") == "1",
        "enable_opening_failure_short": os.environ.get(
            "SSQUANT_ENABLE_OPENING_FAILURE_SHORT", "0"
        )
        == "1",
        "auxiliary_max_core_z": float(os.environ.get("SSQUANT_AUXILIARY_MAX_CORE_Z", "2.25")),
        "auxiliary_cooldown_bars": int(
            os.environ.get("SSQUANT_AUXILIARY_COOLDOWN_BARS", "7")
        ),
        "opening_failure_minimum_body": float(
            os.environ.get("SSQUANT_OPENING_FAILURE_MINIMUM_BODY", "0.20")
        ),
        "opening_failure_minimum_flow": float(
            os.environ.get("SSQUANT_OPENING_FAILURE_MINIMUM_FLOW", "0.05")
        ),
        "pullback_z": float(os.environ.get("SSQUANT_PULLBACK_Z", "0.8")),
        "pullback_reclaim_delta": float(
            os.environ.get("SSQUANT_PULLBACK_RECLAIM_DELTA", "0.35")
        ),
        "pullback_minimum_body": float(
            os.environ.get("SSQUANT_PULLBACK_MINIMUM_BODY", "0.15")
        ),
        "pullback_minimum_flow": float(
            os.environ.get("SSQUANT_PULLBACK_MINIMUM_FLOW", "0.05")
        ),
        "breakout_lookback_bars": int(os.environ.get("SSQUANT_BREAKOUT_LOOKBACK_BARS", "6")),
        "breakout_volume_lookback_bars": int(
            os.environ.get("SSQUANT_BREAKOUT_VOLUME_LOOKBACK_BARS", "12")
        ),
        "breakout_minimum_body": float(
            os.environ.get("SSQUANT_BREAKOUT_MINIMUM_BODY", "0.35")
        ),
        "breakout_minimum_flow": float(
            os.environ.get("SSQUANT_BREAKOUT_MINIMUM_FLOW", "0.08")
        ),
        "breakout_volume_multiplier": float(
            os.environ.get("SSQUANT_BREAKOUT_VOLUME_MULTIPLIER", "1.15")
        ),
        "higher_trend_bars": 3,
        "higher_trend_threshold": 0.005,
        "higher_fast_bars": 3,
        "higher_slow_bars": 8,
        "higher_slope_bars": 3,
        "higher_efficiency_threshold": 0.35,
        "higher_period_minutes": period_minutes(HIGHER_PERIOD),
        "evaluation_start": START_DATE,
        "symbol_prefix": "".join(ch for ch in SYMBOL if ch.isalpha()),
        "max_volume": 1,
    }

    runner = UnifiedStrategyRunner(mode=RunMode.BACKTEST)
    runner.set_config(config)
    data_manager_class, original_fetch = _install_frozen_fetch(selected)
    try:
        results = runner.run(
            strategy=strategy.handle_bar,
            initialize=strategy.initialize,
            strategy_params=strategy_params,
        )
    finally:
        data_manager_class.fetch_data = original_fetch

    no_trade_run = _validate_formal_results(results, selected, runner)
    if no_trade_run:
        _inject_no_trade_result(results, selected, runner)
    if USE_REAL_IM_PARAMS:
        fee_adjustment = apply_close_today_costs(
            results,
            contract_multiplier=200,
            regular_close_rate=0.000023,
            close_today_rate=0.00023,
        )
        runner.backtester.result_calculator.calculate_performance(results)
    else:
        fee_adjustment = {"applied": False}
    report_source = results.get("html_report_path") or results.get("report_path")
    if not report_source or not Path(report_source).is_file():
        raise RuntimeError("SSQuant engine report was not generated")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(report_source).suffix.lower()
    shutil.copy2(report_source, OUTPUT_DIR / ("engine_report.html" if suffix == ".html" else "engine_report.txt"))
    manifest = {
        "schema_version": 2,
        "run_type": "formal-backtest",
        "bars_hash": _subset_hash(selected),
        "adjustment_mode": dataset["adjustment_mode"],
        "strategy_id": "microstructure-vwap-deviation-optimized",
        "strategy_version": STRATEGY_VERSION,
        "strategy_file_hash": sha256_file(STRATEGY_FILE),
        "engine_path": str(ENGINE_PATH),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "initial_capital": INITIAL_CAPITAL,
        "symbol": SYMBOL,
        "base_period": BASE_PERIOD,
        "higher_period": HIGHER_PERIOD,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "source_ids": [source["source_id"] for source in selected],
        "dataset_manifest_path": str(subset_manifest_path),
        "data_mode": "frozen",
        "frozen_input_used": True,
        "data_pipeline": "official SSQuant get_futures_data -> schema-v2 frozen CSV -> SSQuant DataSource",
        "strategy_params": strategy_params,
        "execution_params": {
            "initial_capital": INITIAL_CAPITAL,
            "slippage_ticks": 1,
            "fee_model": (
                "IM explicit rates; close_today delta applied by governed formal runner"
                if USE_REAL_IM_PARAMS
                else "SSQuant contract metadata"
            ),
            **_contract_overrides(),
        },
        "fee_adjustment": fee_adjustment,
        "no_trade_run": no_trade_run,
    }
    artifacts = write_formal_artifacts(results, OUTPUT_DIR, run_manifest=manifest, config=config)
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    written_manifest["performance"] = results.get("performance", {})
    written_manifest["trade_count"] = sum(
        len(result.get("trades", []))
        for result in results.values()
        if isinstance(result, dict)
    )
    manifest_path.write_text(
        json.dumps(written_manifest, ensure_ascii=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    summary = {
        "time_range": {"start": START_DATE, "end": END_DATE},
        "time_frame": {"base": BASE_PERIOD, "higher": HIGHER_PERIOD},
        "code_version": STRATEGY_VERSION,
        "equity_chart": str(artifacts["equity_curve.png"]),
        "performance": results.get("performance", {}),
    }
    (OUTPUT_DIR / "backtest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"[FORMAL OK] {SYMBOL} {BASE_PERIOD}->{HIGHER_PERIOD}: {OUTPUT_DIR}")
    return results


def _subset_hash(selected: list[dict]) -> str:
    import hashlib

    payload = json.dumps(
        [(source["source_id"], source["bars_hash"]) for source in selected],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
