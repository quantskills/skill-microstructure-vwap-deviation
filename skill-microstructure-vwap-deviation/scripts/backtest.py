"""Standalone research backtest for the signal engine.

This module is deliberately separate from SSQuant's governed AccountBridge. Its
outputs are suitable for offline strategy validation; formal SSQuant evaluation
must use the project's official runner and account layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from vwap_strategy import VWAPConfig, generate_signals


TRADE_COLUMNS = [
    "entry_time",
    "exit_time",
    "direction",
    "entry_price",
    "exit_price",
    "quantity",
    "gross_pnl",
    "commission",
    "net_pnl",
    "exit_reason",
]


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 300_000.0
    quantity: int = 1
    fee_rate: float = 0.0003
    slippage_bps: float = 1.0

    def validate(self) -> None:
        if self.initial_cash <= 0 or self.quantity <= 0:
            raise ValueError("initial_cash and quantity must be positive")
        if self.fee_rate < 0 or self.slippage_bps < 0:
            raise ValueError("fee_rate and slippage_bps cannot be negative")


def _fill_price(price: float, direction: int, slippage_bps: float) -> float:
    factor = 1.0 + slippage_bps / 10_000.0
    return price * (factor if direction > 0 else 2.0 - factor)


def _trade_row(position, exit_time, exit_price, config, exit_reason):
    direction = position["direction"]
    gross = direction * (exit_price - position["entry_price"]) * config.quantity
    commission = position["entry_commission"] + abs(exit_price) * config.quantity * config.fee_rate
    return {
        "entry_time": position["entry_time"],
        "exit_time": exit_time,
        "direction": direction,
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "quantity": config.quantity,
        "gross_pnl": gross,
        "commission": commission,
        "net_pnl": gross - commission,
        "exit_reason": exit_reason,
    }


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_COLUMNS)


def _session_key(timestamp: pd.Timestamp, row: pd.Series) -> object:
    if "session" in row and pd.notna(row["session"]):
        return row["session"]
    return timestamp.normalize()


def _summary(equity: pd.Series, trades: pd.DataFrame, config: BacktestConfig) -> dict:
    if equity.empty:
        return {"initial_cash": config.initial_cash, "final_equity": config.initial_cash, "trade_count": 0}
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    winners = trades.loc[trades["net_pnl"] > 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    losers = trades.loc[trades["net_pnl"] < 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    loss_total = abs(losers.sum())
    return {
        "initial_cash": config.initial_cash,
        "final_equity": float(equity.iloc[-1]),
        "total_return_pct": float((equity.iloc[-1] / config.initial_cash - 1.0) * 100),
        "max_drawdown_pct": float(drawdown.min() * 100),
        "sharpe": float(np.sqrt(252 * 390) * returns.mean() / returns.std()) if returns.std() else 0.0,
        "trade_count": int(len(trades)),
        "win_rate_pct": float((len(winners) / len(trades)) * 100) if len(trades) else 0.0,
        "profit_factor": float(winners.sum() / loss_total) if loss_total else None,
        "total_commission": float(trades["commission"].sum()) if not trades.empty else 0.0,
    }


def run_backtest(
    bars: pd.DataFrame,
    strategy_config: VWAPConfig | None = None,
    backtest_config: BacktestConfig | None = None,
) -> dict:
    """Run the standalone next-bar-open execution model on one symbol."""

    strategy_config = strategy_config or VWAPConfig()
    backtest_config = backtest_config or BacktestConfig()
    backtest_config.validate()
    if len(bars) == 0:
        return {
            "signals": generate_signals(bars, strategy_config),
            "equity_curve": pd.DataFrame(columns=["equity", "position"]),
            "trades": _empty_trades(),
            "summary": _summary(pd.Series(dtype=float), _empty_trades(), backtest_config),
            "config": {"strategy": asdict(strategy_config), "backtest": asdict(backtest_config)},
        }

    frame = bars.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    signals = generate_signals(frame, strategy_config)
    cash = backtest_config.initial_cash
    position = None
    trades = []
    equity_rows = []
    previous_session = None

    for index, (timestamp, row) in enumerate(frame.iterrows()):
        session = _session_key(timestamp, row)
        if previous_session is not None and session != previous_session and position is not None:
            exit_price = _fill_price(float(row["open"]), -position["direction"], backtest_config.slippage_bps)
            trade = _trade_row(position, timestamp, exit_price, backtest_config, "session_reset")
            cash += trade["gross_pnl"] - (trade["commission"] - position["entry_commission"])
            trades.append(trade)
            position = None
        previous_session = session

        if index > 0 and position is not None and signals.iloc[index - 1]["action"] == "exit":
            exit_price = _fill_price(float(row["open"]), -position["direction"], backtest_config.slippage_bps)
            trade = _trade_row(position, timestamp, exit_price, backtest_config, "signal")
            cash += trade["gross_pnl"] - (trade["commission"] - position["entry_commission"])
            trades.append(trade)
            position = None
        elif index > 0 and position is None and signals.iloc[index - 1]["action"] in {"enter_long", "enter_short"}:
            direction = int(signals.iloc[index - 1]["signal"])
            entry_price = _fill_price(float(row["open"]), direction, backtest_config.slippage_bps)
            entry_commission = abs(entry_price) * backtest_config.quantity * backtest_config.fee_rate
            cash -= entry_commission
            position = {
                "entry_time": timestamp,
                "entry_price": entry_price,
                "direction": direction,
                "entry_commission": entry_commission,
            }

        marked = cash
        if position is not None:
            marked += position["direction"] * (float(row["close"]) - position["entry_price"]) * backtest_config.quantity
        equity_rows.append({"datetime": timestamp, "equity": marked, "position": position["direction"] if position else 0})

    if position is not None:
        timestamp = frame.index[-1]
        exit_price = _fill_price(float(frame.iloc[-1]["close"]), -position["direction"], backtest_config.slippage_bps)
        trade = _trade_row(position, timestamp, exit_price, backtest_config, "forced_end")
        cash += trade["gross_pnl"] - (trade["commission"] - position["entry_commission"])
        trades.append(trade)
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["position"] = 0

    equity_curve = pd.DataFrame(equity_rows).set_index("datetime")
    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS) if trades else _empty_trades()
    return {
        "signals": signals,
        "equity_curve": equity_curve,
        "trades": trade_frame,
        "summary": _summary(equity_curve["equity"], trade_frame, backtest_config),
        "config": {"strategy": asdict(strategy_config), "backtest": asdict(backtest_config)},
    }
