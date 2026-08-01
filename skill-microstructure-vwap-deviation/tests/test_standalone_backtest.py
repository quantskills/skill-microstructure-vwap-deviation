import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backtest import BacktestConfig, run_backtest
from run_backtest import _parse_args
from vwap_strategy import VWAPConfig


def test_entry_is_filled_on_next_bar_open():
    index = pd.date_range("2026-01-05 09:30", periods=7, freq="min")
    bars = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 999, 101],
            "high": [100, 100, 100, 100, 100, 120, 101],
            "low": [100, 100, 100, 100, 100, 120, 100],
            "close": [100, 100, 100, 100, 100, 120, 100],
            "volume": [1] * 7,
        },
        index=index,
    )

    result = run_backtest(
        bars,
        VWAPConfig(vwap_minutes=5, bar_minutes=1, trend_filter=False, session_end=None),
        BacktestConfig(fee_rate=0.0, slippage_bps=0.0),
    )

    assert result["signals"].iloc[5]["action"] == "enter_short"
    assert result["trades"].iloc[0]["entry_price"] == 101


def test_empty_trade_run_still_returns_equity_curve():
    index = pd.date_range("2026-01-05 09:30", periods=5, freq="min")
    bars = pd.DataFrame(
        {
            "open": [100] * 5,
            "high": [100] * 5,
            "low": [100] * 5,
            "close": [100] * 5,
            "volume": [1] * 5,
        },
        index=index,
    )

    result = run_backtest(
        bars,
        VWAPConfig(vwap_minutes=5, bar_minutes=1, trend_filter=False, session_end=None),
        BacktestConfig(),
    )

    assert result["trades"].empty
    assert len(result["equity_curve"]) == len(bars)
    assert result["summary"]["trade_count"] == 0


def test_entry_commission_is_counted_once():
    index = pd.date_range("2026-01-05 09:30", periods=7, freq="min")
    bars = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 999, 101],
            "high": [100, 100, 100, 100, 100, 120, 101],
            "low": [100, 100, 100, 100, 100, 120, 100],
            "close": [100, 100, 100, 100, 100, 120, 100],
            "volume": [1] * 7,
        },
        index=index,
    )
    result = run_backtest(
        bars,
        VWAPConfig(vwap_minutes=5, bar_minutes=1, trend_filter=False, session_end=None),
        BacktestConfig(fee_rate=0.001, slippage_bps=0.0),
    )

    trade = result["trades"].iloc[0]
    assert trade["commission"] == 0.201
    assert result["summary"]["total_commission"] == 0.201


def test_cli_exposes_cost_and_account_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_backtest.py",
            "--fee-rate",
            "0.001",
            "--slippage-bps",
            "2.5",
            "--initial-cash",
            "100000",
            "--quantity",
            "3",
        ],
    )

    args = _parse_args()

    assert args.fee_rate == 0.001
    assert args.slippage_bps == 2.5
    assert args.initial_cash == 100000.0
    assert args.quantity == 3
