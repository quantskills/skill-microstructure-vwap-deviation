"""Apply IM's close-today fee to governed SSQuant result objects."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


def _is_close_trade(trade: Mapping[str, Any]) -> bool:
    value = trade.get("net_profit")
    return value is not None and pd.notna(value)


def _trade_datetime(trade: Mapping[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(trade["datetime"])


def _recompute_result_metrics(result: dict[str, Any]) -> None:
    curve = result.get("equity_curve")
    if not isinstance(curve, pd.Series) or curve.empty:
        return
    curve = curve.astype(float)
    initial = float(result.get("initial_capital", curve.iloc[0]))
    final = float(curve.iloc[-1])
    closed = [trade for trade in result.get("trades", []) if _is_close_trade(trade)]
    net_profits = np.asarray([float(trade["net_profit"]) for trade in closed], dtype=float)
    wins = net_profits[net_profits > 0]
    losses = net_profits[net_profits <= 0]

    result["final_equity"] = final
    result["net_value"] = final / initial if initial else 0.0
    result["total_net_profit"] = final - initial
    result["total_trades"] = int(len(closed))
    result["win_trades"] = int(len(wins))
    result["loss_trades"] = int(len(losses))
    result["win_rate"] = float(len(wins) / len(net_profits)) if len(net_profits) else 0.0
    result["profit_factor"] = (
        float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else 0.0
    )
    result["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    result["avg_loss"] = float(losses.mean()) if len(losses) else 0.0

    cummax = curve.cummax()
    drawdown = cummax - curve
    result["max_drawdown"] = float(drawdown.max())
    result["max_drawdown_pct"] = float((drawdown / cummax).replace([np.inf, -np.inf], np.nan).max()) * 100

    daily_equity = curve.resample("D").last().dropna()
    annual_return = 0.0
    sharpe_ratio = 0.0
    if len(daily_equity) > 1:
        daily_returns = daily_equity.pct_change().dropna()
        total_return = daily_equity.iloc[-1] / daily_equity.iloc[0] - 1.0
        years = len(daily_equity) / 250.0
        annual_return = total_return / years * 100 if years > 0 else 0.0
        if len(daily_returns) and daily_returns.std() > 0:
            sharpe_ratio = (
                (daily_returns.mean() - 0.03 / 250.0)
                / daily_returns.std()
                * np.sqrt(250.0)
            )
    result["annual_return"] = float(annual_return)
    result["sharpe_ratio"] = float(sharpe_ratio)
    result["total_amount_profit"] = float(
        result["total_net_profit"]
        + float(result.get("total_commission", 0.0) or 0.0)
        + float(result.get("total_slippage", 0.0) or 0.0)
    )


def apply_close_today_costs(
    results: Mapping[str, Any],
    contract_multiplier: float,
    regular_close_rate: float,
    close_today_rate: float,
) -> dict[str, Any]:
    """Mutate engine results with the fee delta for same-day closing legs."""
    extra_rate = float(close_today_rate) - float(regular_close_rate)
    total_extra_fee = 0.0
    same_day_closes = 0
    by_dataset: dict[str, float] = {}

    for dataset, result in results.items():
        if not isinstance(result, dict) or not isinstance(result.get("trades"), list):
            continue
        pending_entry: Mapping[str, Any] | None = None
        events: list[tuple[pd.Timestamp, float]] = []
        for trade in sorted(result["trades"], key=_trade_datetime):
            if not _is_close_trade(trade):
                pending_entry = trade
                continue
            if pending_entry is None:
                continue
            entry_day: date = _trade_datetime(pending_entry).date()
            close_time = _trade_datetime(trade)
            if entry_day == close_time.date():
                close_price = float(trade.get("raw_price", trade.get("price", 0.0)))
                volume = float(trade.get("volume", 1.0) or 0.0)
                extra_fee = close_price * volume * float(contract_multiplier) * extra_rate
                trade["commission"] = float(trade.get("commission", 0.0) or 0.0) + extra_fee
                trade["net_profit"] = float(trade["net_profit"]) - extra_fee
                if "profit" in trade and trade["profit"] is not None:
                    trade["profit"] = float(trade["profit"]) - extra_fee
                events.append((close_time, extra_fee))
                total_extra_fee += extra_fee
                same_day_closes += 1
            pending_entry = None

        if events and isinstance(result.get("equity_curve"), pd.Series):
            corrected_curve = result["equity_curve"].astype(float).copy()
            for close_time, extra_fee in events:
                corrected_curve.loc[corrected_curve.index >= close_time] -= extra_fee
            result["equity_curve"] = corrected_curve
        result["total_commission"] = float(result.get("total_commission", 0.0) or 0.0) + sum(
            amount for _, amount in events
        )
        by_dataset[str(dataset)] = float(sum(amount for _, amount in events))
        _recompute_result_metrics(result)

    return {
        "regular_close_rate": float(regular_close_rate),
        "close_today_rate": float(close_today_rate),
        "extra_rate": extra_rate,
        "same_day_closes": same_day_closes,
        "total_extra_fee": float(total_extra_fee),
        "by_dataset": by_dataset,
    }
