# -*- coding: utf-8 -*-
"""Read-only attribution analysis for governed VWAP-deviation backtest outputs."""

from collections import deque
from datetime import timedelta
from pathlib import Path
import sys

import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from research.project_paths import RUNS_ROOT

ROOT = RUNS_ROOT
SYMBOLS = ("IM888", "AU888", "RB888", "CU888", "AG888")


def _session_key(dt):
    if dt.hour >= 21:
        return dt.date()
    if dt.hour < 4:
        return (dt - timedelta(days=1)).date()
    return dt.date()


def signal_features(symbol):
    """Reconstruct current-bar features for attribution only."""
    bars = pd.read_csv(ROOT / symbol / "frozen_dataset" / "bars.csv")
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    values = deque()
    trend = deque()
    sum_volume = sum_pv = sum_pv2 = 0.0
    previous_session = None
    rows = []
    for row in bars.itertuples(index=False):
        current_session = _session_key(row.datetime)
        if previous_session is not None and current_session != previous_session:
            values.clear()
            trend.clear()
            sum_volume = sum_pv = sum_pv2 = 0.0
        previous_session = current_session
        price = float(row.close)
        volume = max(float(row.volume), 0.0)
        if len(values) >= 6:
            old_price, old_volume = values.popleft()
            sum_volume -= old_volume
            sum_pv -= old_price * old_volume
            sum_pv2 -= old_price * old_price * old_volume
        values.append((price, volume))
        sum_volume += volume
        sum_pv += price * volume
        sum_pv2 += price * price * volume
        trend.append(price)
        while len(trend) > 30:
            trend.popleft()
        zscore = float("nan")
        if sum_volume > 0 and len(values) >= 6:
            vwap = sum_pv / sum_volume
            dispersion = max(sum_pv2 / sum_volume - vwap * vwap, 0.0) ** 0.5
            if dispersion > 1e-12:
                zscore = (price - vwap) / dispersion
        trend_state = "unknown"
        if len(trend) >= 30:
            change = trend[-1] / trend[0] - 1.0 if trend[0] else 0.0
            trend_state = "up" if change > 0.003 else "down" if change < -0.003 else "range"
        rows.append({"signal_time": row.datetime, "entry_z": zscore, "trend_state": trend_state})
    features = pd.DataFrame(rows)
    features["entry_time"] = features["signal_time"].shift(-1)
    return features[["entry_time", "entry_z", "trend_state"]].dropna()


def pair_trades(trades):
    rows = []
    pending = None
    for _, trade in trades.sort_values("datetime").iterrows():
        action = trade["action"]
        if action in ("开多", "开空"):
            pending = trade
            continue
        if action not in ("平多", "平空") or pending is None:
            continue
        entry = pending
        exit_trade = trade
        rows.append(
            {
                "entry_time": pd.to_datetime(entry["datetime"]),
                "exit_time": pd.to_datetime(exit_trade["datetime"]),
                "direction": "long" if entry["action"] == "开多" else "short",
                "entry_price": float(entry["raw_price"]),
                "exit_price": float(exit_trade["raw_price"]),
                "amount_profit": float(exit_trade.get("amount_profit", 0.0) or 0.0),
                "net_profit": float(exit_trade.get("net_profit", 0.0) or 0.0),
                "commission": float(entry.get("commission", 0.0) or 0.0)
                + float(exit_trade.get("commission", 0.0) or 0.0),
                "slippage": float(entry.get("slippage", 0.0) or 0.0)
                + float(exit_trade.get("slippage", 0.0) or 0.0),
            }
        )
        pending = None
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["hold_minutes"] = (result["exit_time"] - result["entry_time"]).dt.total_seconds() / 60.0
    result["entry_hour"] = result["entry_time"].dt.hour
    result["entry_month"] = result["entry_time"].dt.to_period("M").astype(str)
    result["exit_year"] = result["exit_time"].dt.year
    result["entry_session"] = result["entry_hour"].map(
        lambda hour: "night" if hour >= 21 or hour < 4 else "day"
    )
    result = result.merge(signal_features(symbol), on="entry_time", how="left")
    return result


def summarize(symbol):
    path = ROOT / symbol
    trades = pd.read_csv(path / "trades.csv")
    paired = pair_trades(trades)
    if paired.empty:
        return {"symbol": symbol, "closed_trades": 0}
    summary = {
        "symbol": symbol,
        "closed_trades": len(paired),
        "long_trades": int((paired["direction"] == "long").sum()),
        "short_trades": int((paired["direction"] == "short").sum()),
        "long_pnl": float(paired.loc[paired["direction"] == "long", "net_profit"].sum()),
        "short_pnl": float(paired.loc[paired["direction"] == "short", "net_profit"].sum()),
        "avg_pnl": float(paired["net_profit"].mean()),
        "win_rate": float((paired["net_profit"] > 0).mean()),
        "avg_win": float(paired.loc[paired["net_profit"] > 0, "net_profit"].mean()),
        "avg_loss": float(paired.loc[paired["net_profit"] <= 0, "net_profit"].mean()),
        "profit_factor": float(
            paired.loc[paired["net_profit"] > 0, "net_profit"].sum()
            / abs(paired.loc[paired["net_profit"] <= 0, "net_profit"].sum())
        ),
        "gross_profit": float(paired["amount_profit"].sum()),
        "commission": float(paired["commission"].sum()),
        "slippage": float(paired["slippage"].sum()),
        "entry_abs_z_mean": float(paired["entry_z"].abs().mean()),
        "entry_abs_z_median": float(paired["entry_z"].abs().median()),
        "entry_trend_states": paired["trend_state"].value_counts().to_dict(),
        "median_hold_minutes": float(paired["hold_minutes"].median()),
        "p90_hold_minutes": float(paired["hold_minutes"].quantile(0.9)),
        "night_pnl": float(paired.loc[paired["entry_session"] == "night", "net_profit"].sum()),
        "day_pnl": float(paired.loc[paired["entry_session"] == "day", "net_profit"].sum()),
        "entry_hour_pnl": paired.groupby("entry_hour")["net_profit"].sum().round(2).to_dict(),
    }
    by_month = paired.groupby("entry_month")["net_profit"].sum()
    summary["positive_months"] = int((by_month > 0).sum())
    summary["negative_months"] = int((by_month < 0).sum())
    summary["exit_year_pnl"] = paired.groupby("exit_year")["net_profit"].sum().round(2).to_dict()
    z_bucket = pd.cut(
        paired["entry_z"].abs(),
        bins=[2.0, 2.25, 2.5, 3.0, float("inf")],
        labels=["2.0-2.25", "2.25-2.5", "2.5-3.0", ">3.0"],
        include_lowest=True,
    )
    hold_bucket = pd.cut(
        paired["hold_minutes"],
        bins=[-float("inf"), 15, 30, float("inf")],
        labels=["<=15m", "15-30m", ">30m"],
    )
    summary["z_bucket_pnl"] = paired.groupby(z_bucket, observed=False)["net_profit"].sum().round(2).to_dict()
    summary["z_bucket_count"] = paired.groupby(z_bucket, observed=False).size().to_dict()
    summary["hold_bucket_pnl"] = paired.groupby(hold_bucket, observed=False)["net_profit"].sum().round(2).to_dict()
    return summary


if __name__ == "__main__":
    for symbol in SYMBOLS:
        print(summarize(symbol))
