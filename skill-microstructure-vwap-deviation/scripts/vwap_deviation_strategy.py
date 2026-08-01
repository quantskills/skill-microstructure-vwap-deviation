# -*- coding: utf-8 -*-
"""Minute VWAP deviation strategy for SSQuant v0.4.6 V5."""

from collections import deque
from datetime import datetime, time, timedelta
import os
import sys

import numpy as np
import pandas as pd

from ssquant.api.strategy_api import StrategyAPI
from ssquant.data.indicator_cache import IndicatorCache

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import shared.framework_bridge as _fw
from shared.framework_bridge import (
    PARAM_CONSECUTIVE_LOSS_LIMIT,
    PARAM_COOLDOWN_BARS,
    PARAM_CONTRACT_MULTIPLIER,
    PARAM_MAX_VOLUME,
    _get_last_fill_adj_price,
    _get_last_fill_raw_price,
    _get_raw_price,
    calc_position_size,
    reset_bridge,
)


PARAM_VWAP_MINUTES = 30
PARAM_BAR_MINUTES = 5
PARAM_ENTRY_Z = 2.0
PARAM_EXIT_Z = 0.5
PARAM_TREND_WINDOW = 30
PARAM_TREND_THRESHOLD = 0.003
PARAM_TAIL_LOCK_MINUTES = 30
PARAM_SYMBOL_PREFIX = "IM"
PARAM_MAX_VOLUME_OVERRIDE = 1
PARAM_MARGIN_RATE_FALLBACK = 0.12

g_cache = IndicatorCache()
g_entry_price = 0.0
g_entry_raw_price = 0.0
g_entry_bar = 0
g_session_id = None
g_values = deque()
g_trend_history = deque()
g_sum_volume = 0.0
g_sum_pv = 0.0
g_sum_pv2 = 0.0


def _reset_signal_state():
    global g_session_id, g_values, g_trend_history
    global g_sum_volume, g_sum_pv, g_sum_pv2
    g_session_id = None
    g_values = deque()
    g_trend_history = deque()
    g_sum_volume = 0.0
    g_sum_pv = 0.0
    g_sum_pv2 = 0.0


def initialize(api: StrategyAPI):
    """Reset framework and strategy state only; no data or indicator precomputation."""

    global g_cache, g_entry_price, g_entry_raw_price, g_entry_bar
    reset_bridge()
    g_cache.reset()
    _reset_signal_state()
    g_entry_price = 0.0
    g_entry_raw_price = 0.0
    g_entry_bar = 0
    api.log("[VWAP deviation] initialized")


def _current_datetime(klines):
    if isinstance(klines.index, pd.DatetimeIndex):
        return pd.Timestamp(klines.index[-1]).to_pydatetime()
    for name in ("datetime", "date", "time", "timestamp"):
        if name in klines.columns:
            return pd.Timestamp(klines[name].iloc[-1]).to_pydatetime()
    return None


def _session_key(current_dt):
    if current_dt is None:
        return None
    if current_dt.hour >= 21:
        return current_dt.date()
    if current_dt.hour < 4:
        return (current_dt - timedelta(days=1)).date()
    return current_dt.date()


def _tail_locked(current_dt, prefix, lock_minutes):
    if current_dt is None or lock_minutes <= 0:
        return False
    if prefix == "AU":
        session_ends = (time(15, 0), time(2, 30))
    elif prefix in {"CU", "AG", "NI", "SN", "AL", "ZN", "PB"}:
        session_ends = (time(15, 0), time(1, 0))
    elif prefix in {"RB", "HC", "BU", "FU", "RU", "SP", "SS"}:
        session_ends = (time(15, 0), time(23, 0))
    else:
        session_ends = (time(15, 0),)

    current = current_dt.time()
    for end_time in session_ends:
        end_dt = datetime.combine(current_dt.date(), end_time)
        if end_time.hour < 4 and current_dt.hour >= 12:
            end_dt += timedelta(days=1)
        start_time = (end_dt - timedelta(minutes=lock_minutes)).time()
        if start_time <= current <= end_time:
            return True
    return False


def _update_rolling(price, volume, window_bars):
    global g_sum_volume, g_sum_pv, g_sum_pv2
    volume = max(float(volume), 0.0)
    if len(g_values) >= window_bars:
        old_price, old_volume = g_values.popleft()
        g_sum_volume -= old_volume
        g_sum_pv -= old_price * old_volume
        g_sum_pv2 -= old_price * old_price * old_volume
    g_values.append((price, volume))
    g_sum_volume += volume
    g_sum_pv += price * volume
    g_sum_pv2 += price * price * volume
    if g_sum_volume <= 0:
        return np.nan, np.nan, len(g_values)
    vwap = g_sum_pv / g_sum_volume
    variance = max(g_sum_pv2 / g_sum_volume - vwap * vwap, 0.0)
    return vwap, float(np.sqrt(variance)), len(g_values)


def _trend_state(price, trend_window, trend_threshold):
    g_trend_history.append(float(price))
    while len(g_trend_history) > trend_window:
        g_trend_history.popleft()
    if len(g_trend_history) < trend_window:
        return "unknown"
    first = g_trend_history[0]
    change = g_trend_history[-1] / first - 1.0 if first else 0.0
    if change > trend_threshold:
        return "up"
    if change < -trend_threshold:
        return "down"
    return "range"


def _set_custom_cache(vwap, dispersion, trend_state):
    g_cache.set("rolling_vwap", np.asarray([vwap], dtype=float))
    g_cache.set("rolling_vwap_std", np.asarray([dispersion], dtype=float))
    trend_code = {"unknown": 0.0, "range": 1.0, "up": 2.0, "down": -2.0}[trend_state]
    g_cache.set("vwap_trend_state", np.asarray([trend_code], dtype=float))


def _close_position(api, pos, reason):
    direction = "LONG" if pos > 0 else "SHORT"
    _fw.g_bridge.register_pending_close(api, direction)
    if pos > 0:
        api.sell(volume=pos, order_type="next_bar_open", reason=reason)
    else:
        api.buycover(volume=-pos, order_type="next_bar_open", reason=reason)


def handle_bar(api: StrategyAPI):
    global g_cache, g_entry_price, g_entry_raw_price, g_entry_bar, g_session_id

    vwap_minutes = api.get_param("vwap_minutes", PARAM_VWAP_MINUTES)
    bar_minutes = api.get_param("bar_minutes", PARAM_BAR_MINUTES)
    entry_z = api.get_param("entry_z", PARAM_ENTRY_Z)
    exit_z = api.get_param("exit_z", PARAM_EXIT_Z)
    trend_window = api.get_param("trend_window", PARAM_TREND_WINDOW)
    trend_threshold = api.get_param("trend_threshold", PARAM_TREND_THRESHOLD)
    tail_lock_minutes = api.get_param("tail_lock_minutes", PARAM_TAIL_LOCK_MINUTES)
    symbol_prefix = api.get_param("symbol_prefix", PARAM_SYMBOL_PREFIX)
    consecutive_limit = api.get_param("consecutive_loss_limit", PARAM_CONSECUTIVE_LOSS_LIMIT)
    cooldown_bars = api.get_param("cooldown_bars", PARAM_COOLDOWN_BARS)

    window_bars = max(1, int(round(vwap_minutes / bar_minutes)))
    min_bars = max(window_bars, trend_window)
    bar_count = api.get_idx()
    if bar_count < min_bars:
        return

    klines = api.get_klines()
    if klines is None or len(klines) < min_bars:
        return
    if "volume" not in klines.columns:
        api.log("[VWAP deviation] missing volume column; skip bar")
        return

    current_dt = _current_datetime(klines)
    current_session = _session_key(current_dt)
    session_changed = g_session_id is not None and current_session != g_session_id
    session_reset_ordered = False

    # Step 2: previous next-bar orders are matched by the engine on this bar.
    # Step 3: confirm P&L through AccountBridge before making a new decision.
    _fw.g_bridge.confirm_closed_trades(
        api,
        bar_count,
        consecutive_loss_limit=consecutive_limit,
        cooldown_bars=cooldown_bars,
    )
    pos = api.get_pos()

    if pos != 0 and g_entry_price == 0.0:
        fill_raw = _get_last_fill_raw_price(api)
        fill_adj = _get_last_fill_adj_price(api)
        if fill_raw > 0:
            g_entry_raw_price = fill_raw
            g_entry_price = fill_adj if fill_adj > 0 else float(klines["close"].values[-1])
            g_entry_bar = bar_count
            _fw.g_bridge.snapshot_balance_at_entry(api)

    if session_changed:
        if pos != 0:
            _close_position(api, pos, "session_reset")
            session_reset_ordered = True
            g_entry_price = 0.0
            g_entry_raw_price = 0.0
            g_entry_bar = 0
        _reset_signal_state()
    g_session_id = current_session

    # Step 3.5: rollover compensation is required for a continuous futures series.
    if pos != 0 and g_entry_raw_price > 0 and g_entry_price > 0:
        contract_mult = api.get_param("contract_multiplier", PARAM_CONTRACT_MULTIPLIER)
        margin_rate = api.get_param("margin_rate", PARAM_MARGIN_RATE_FALLBACK)
        rollover = _fw.compensate_rollover_gap(
            api, g_entry_price, g_entry_raw_price, contract_mult, margin_rate
        )
        if rollover["compensated"]:
            g_entry_price = rollover["entry_price"]
            g_entry_raw_price = rollover["entry_raw_price"]

    # Step 4: incremental custom indicators from the current bar only.
    if bar_count <= g_cache.n:
        return
    close = float(klines["close"].values[-1])
    volume = float(klines["volume"].values[-1])
    vwap, dispersion, sample_size = _update_rolling(close, volume, window_bars)
    trend_state = _trend_state(close, trend_window, trend_threshold)
    _set_custom_cache(vwap, dispersion, trend_state)
    g_cache.n = bar_count

    if session_reset_ordered:
        return

    deviation = close - vwap if np.isfinite(vwap) else np.nan
    zscore = deviation / dispersion if np.isfinite(dispersion) and dispersion > 1e-12 else np.nan
    if not np.isfinite(zscore) or sample_size < window_bars:
        return

    # Step 5: exits first; every order is next_bar_open.
    if pos != 0:
        if abs(deviation) <= exit_z * dispersion and np.isfinite(dispersion):
            _close_position(api, pos, "vwap_reversion")
            g_entry_price = 0.0
            g_entry_raw_price = 0.0
            g_entry_bar = 0
        return

    if _tail_locked(current_dt, str(symbol_prefix).upper(), tail_lock_minutes):
        return
    if trend_state != "range":
        return
    if abs(zscore) < entry_z:
        return

    raw_price = _get_raw_price(api)
    contract_mult = api.get_param("contract_multiplier", PARAM_CONTRACT_MULTIPLIER)
    margin_rate = api.get_param("margin_rate", PARAM_MARGIN_RATE_FALLBACK)
    max_volume = api.get_param("max_volume", PARAM_MAX_VOLUME_OVERRIDE)
    if max_volume != PARAM_MAX_VOLUME_OVERRIDE:
        max_volume = min(max_volume, PARAM_MAX_VOLUME_OVERRIDE)
    if _fw.g_bridge.check_cooling_down("LONG", bar_count) or _fw.g_bridge.check_cooling_down("SHORT", bar_count):
        return

    volume_to_trade = calc_position_size(api, contract_mult, raw_price, margin_rate)
    volume_to_trade = max(1, min(int(volume_to_trade), int(max_volume)))
    if deviation > 0:
        api.sellshort(volume=volume_to_trade, order_type="next_bar_open", reason="vwap_overbought")
    else:
        api.buy(volume=volume_to_trade, order_type="next_bar_open", reason="vwap_oversold")
