# -*- coding: utf-8 -*-
"""Cost-aware, confirmed VWAP deviation strategy for SSQuant V5."""

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


PARAM_VWAP_MINUTES = 60
PARAM_LOCAL_VWAP_MINUTES = 30
PARAM_BAR_MINUTES = 5
PARAM_ENTRY_Z = 2.5
PARAM_EXIT_Z = 0.25
PARAM_ATR_WINDOW = 14
PARAM_ATR_STOP_MULTIPLIER = 1.5
PARAM_MAX_HOLD_BARS = 6
PARAM_MAX_HOLD_MINUTES = 30
PARAM_HIGHER_TF_TREND_BARS = 3
PARAM_HIGHER_TF_TREND_THRESHOLD = 0.005
PARAM_HIGHER_TF_FAST_BARS = 3
PARAM_HIGHER_TF_SLOW_BARS = 8
PARAM_HIGHER_TF_SLOPE_BARS = 3
PARAM_HIGHER_TF_EFFICIENCY_THRESHOLD = 0.35
PARAM_HIGHER_PERIOD_MINUTES = 60
PARAM_TAIL_LOCK_MINUTES = 30
PARAM_BREAK_LOCK_MINUTES = 10
PARAM_ENTRY_CONFIRMATION_BARS = 1
PARAM_ENABLE_CORE_MEAN_REVERSION = True
PARAM_ENABLE_TREND_PULLBACK = False
PARAM_ENABLE_TREND_BREAKOUT = False
PARAM_ENABLE_OPENING_FAILURE_SHORT = False
PARAM_AUXILIARY_MAX_CORE_Z = 2.25
PARAM_AUXILIARY_COOLDOWN_BARS = 7
PARAM_OPENING_FAILURE_MINIMUM_BODY = 0.20
PARAM_OPENING_FAILURE_MINIMUM_FLOW = 0.05
PARAM_PULLBACK_Z = 0.8
PARAM_PULLBACK_RECLAIM_DELTA = 0.35
PARAM_PULLBACK_MINIMUM_BODY = 0.15
PARAM_PULLBACK_MINIMUM_FLOW = 0.05
PARAM_BREAKOUT_LOOKBACK_BARS = 6
PARAM_BREAKOUT_VOLUME_LOOKBACK_BARS = 12
PARAM_BREAKOUT_MINIMUM_BODY = 0.35
PARAM_BREAKOUT_MINIMUM_FLOW = 0.08
PARAM_BREAKOUT_VOLUME_MULTIPLIER = 1.15
PARAM_SYMBOL_PREFIX = "IM"
PARAM_MAX_VOLUME_OVERRIDE = 1
PARAM_MARGIN_RATE_FALLBACK = 0.12

g_cache = IndicatorCache()
g_entry_price = 0.0
g_entry_raw_price = 0.0
g_entry_bar = 0
g_entry_datetime = None
g_session_id = None
g_values = deque()
g_trend_history = deque()
g_sum_volume = 0.0
g_sum_pv = 0.0
g_sum_pv2 = 0.0
g_atr_values = deque()
g_previous_close = None
g_pending_z = None
g_pending_direction = 0
g_pending_bar = -1
g_local_values = deque()
g_recent_highs = deque()
g_recent_lows = deque()
g_recent_volumes = deque()
g_previous_local_z = np.nan
g_previous_core_z = np.nan
g_previous_signal_close = np.nan
g_pending_entry_setup = None
g_entry_setup = None
g_pending_entry_reference = np.nan
g_entry_reference = np.nan
g_last_auxiliary_signal_bar = -1
g_opening_high = np.nan
g_opening_low = np.nan
g_previous_session_high = np.nan


def _reset_signal_state():
    global g_session_id, g_values, g_trend_history
    global g_sum_volume, g_sum_pv, g_sum_pv2
    global g_atr_values, g_previous_close
    global g_pending_z, g_pending_direction, g_pending_bar
    global g_local_values, g_recent_highs, g_recent_lows, g_recent_volumes
    global g_previous_local_z, g_previous_core_z, g_previous_signal_close
    global g_pending_entry_setup, g_entry_setup, g_last_auxiliary_signal_bar
    global g_pending_entry_reference, g_entry_reference
    global g_opening_high, g_opening_low, g_previous_session_high
    g_session_id = None
    g_values = deque()
    g_trend_history = deque()
    g_sum_volume = 0.0
    g_sum_pv = 0.0
    g_sum_pv2 = 0.0
    g_atr_values = deque()
    g_previous_close = None
    g_pending_z = None
    g_pending_direction = 0
    g_pending_bar = -1
    g_local_values = deque()
    g_recent_highs = deque()
    g_recent_lows = deque()
    g_recent_volumes = deque()
    g_previous_local_z = np.nan
    g_previous_core_z = np.nan
    g_previous_signal_close = np.nan
    g_pending_entry_setup = None
    g_entry_setup = None
    g_pending_entry_reference = np.nan
    g_entry_reference = np.nan
    g_last_auxiliary_signal_bar = -1
    g_opening_high = np.nan
    g_opening_low = np.nan
    g_previous_session_high = np.nan


def initialize(api: StrategyAPI):
    """Reset framework and strategy state only; no data or indicator precomputation."""

    global g_cache, g_entry_price, g_entry_raw_price, g_entry_bar, g_entry_datetime
    reset_bridge()
    g_cache.reset()
    _reset_signal_state()
    g_entry_price = 0.0
    g_entry_raw_price = 0.0
    g_entry_bar = 0
    g_entry_datetime = None
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


def _break_locked(current_dt, prefix, lock_minutes):
    """Lock index futures before the 11:30-13:00 day-session break."""
    if current_dt is None or lock_minutes <= 0 or str(prefix).upper() not in {"IM", "IF", "IC"}:
        return False
    break_end = datetime.combine(current_dt.date(), time(11, 30))
    break_start = break_end - timedelta(minutes=int(lock_minutes))
    return break_start.time() <= current_dt.time() < break_end.time()


def _auxiliary_holding_window(current_dt, prefix, max_hold_minutes, bar_minutes):
    """Require enough continuous index-futures session time for the planned hold."""
    if current_dt is None or str(prefix).upper() not in {"IM", "IF", "IC"}:
        return False
    planned_exit = current_dt + timedelta(
        minutes=max(0, int(bar_minutes)) + max(0, int(max_hold_minutes))
    )
    if current_dt.time() < time(11, 30):
        session_exit = datetime.combine(current_dt.date(), time(11, 25))
    elif current_dt.time() >= time(13, 0):
        session_exit = datetime.combine(current_dt.date(), time(14, 30))
    else:
        return False
    return planned_exit <= session_exit


def _confirm_reversion(signal_z, current_z):
    """Require the next closed bar to move toward zero without crossing sides."""
    if not np.isfinite(signal_z) or not np.isfinite(current_z):
        return False
    return signal_z * current_z > 0 and abs(current_z) < abs(signal_z)


def _mean_reversion_exit_triggered(direction, zscore, exit_z):
    """Exit inside the VWAP band or immediately after crossing VWAP."""
    if not np.isfinite(zscore) or not np.isfinite(exit_z):
        return False
    return abs(zscore) <= float(exit_z) or float(direction) * float(zscore) >= 0.0


def _exit_order_type(reason):
    """Execute every decision after the signal bar, at the next bar open."""
    return "next_bar_open"


def _entry_confirmation_ready(pending_bar, current_bar, confirmation_bars):
    """Return whether a pending extreme has waited the configured bars."""
    if int(pending_bar) < 0:
        return False
    return int(current_bar) - int(pending_bar) >= max(0, int(confirmation_bars))


def _extreme_entry_action(zscore, entry_z, confirmation_bars):
    """Classify a core extreme without delaying zero-confirmation entries."""
    if not np.isfinite(zscore) or not np.isfinite(entry_z) or abs(zscore) < float(entry_z):
        return "none", 0
    direction = -1 if zscore > 0 else 1
    action = "submit" if int(confirmation_bars) == 0 else "pend"
    return action, direction


def _trend_pullback_direction(
    higher_bias,
    previous_local_z,
    local_z,
    previous_core_z,
    core_z,
    previous_close,
    close,
    body_strength,
    flow_imbalance,
    max_core_z,
    pullback_z,
    reclaim_delta,
    minimum_body,
    minimum_flow,
):
    """Return a higher-trend pullback/reclaim direction, or zero."""
    direction = int(np.sign(higher_bias))
    values = (
        previous_local_z,
        local_z,
        previous_core_z,
        core_z,
        previous_close,
        close,
        body_strength,
        flow_imbalance,
    )
    if direction == 0 or not all(np.isfinite(value) for value in values):
        return 0
    if abs(previous_core_z) >= float(max_core_z) or abs(core_z) >= float(max_core_z):
        return 0
    if direction * previous_local_z > -float(pullback_z):
        return 0
    if direction * (local_z - previous_local_z) < float(reclaim_delta):
        return 0
    if direction * (close - previous_close) <= 0:
        return 0
    if direction * body_strength < float(minimum_body):
        return 0
    if direction * flow_imbalance < float(minimum_flow):
        return 0
    return direction


def _trend_breakout_direction(
    higher_bias,
    previous_core_z,
    core_z,
    close,
    previous_high,
    previous_low,
    body_strength,
    flow_imbalance,
    volume,
    previous_volume_median,
    max_core_z,
    minimum_body,
    minimum_flow,
    volume_multiplier,
):
    """Return a higher-trend, order-flow-confirmed breakout direction, or zero."""
    direction = int(np.sign(higher_bias))
    values = (
        previous_core_z,
        core_z,
        close,
        previous_high,
        previous_low,
        body_strength,
        flow_imbalance,
        volume,
        previous_volume_median,
    )
    if direction == 0 or not all(np.isfinite(value) for value in values):
        return 0
    if abs(previous_core_z) >= float(max_core_z) or abs(core_z) >= float(max_core_z):
        return 0
    price_broke = close > previous_high if direction > 0 else close < previous_low
    if not price_broke:
        return 0
    if direction * body_strength < float(minimum_body):
        return 0
    if direction * flow_imbalance < float(minimum_flow):
        return 0
    if volume < float(volume_multiplier) * previous_volume_median:
        return 0
    return direction


def _update_opening_range(current_dt, high, low, opening_high, opening_low):
    """Update the fixed 09:30-09:55 index-futures opening range."""
    if current_dt is None or not time(9, 30) <= current_dt.time() <= time(9, 55):
        return opening_high, opening_low
    next_high = float(high) if not np.isfinite(opening_high) else max(float(opening_high), high)
    next_low = float(low) if not np.isfinite(opening_low) else min(float(opening_low), low)
    return next_high, next_low


def _opening_failure_window(current_dt, prefix):
    if current_dt is None or str(prefix).upper() not in {"IM", "IF", "IC"}:
        return False
    return time(10, 0) <= current_dt.time() <= time(10, 50)


def _opening_failure_short_direction(
    higher_bias,
    previous_high,
    opening_high,
    close,
    previous_core_z,
    core_z,
    body_strength,
    flow_imbalance,
    max_core_z,
    minimum_body,
    minimum_flow,
):
    """Return short after an opening-high breakout closes back below the range."""
    values = (
        previous_high,
        opening_high,
        close,
        previous_core_z,
        core_z,
        body_strength,
        flow_imbalance,
    )
    if not all(np.isfinite(value) for value in values):
        return 0
    if int(np.sign(higher_bias)) > 0:
        return 0
    if abs(previous_core_z) >= float(max_core_z) or abs(core_z) >= float(max_core_z):
        return 0
    if previous_high <= opening_high or close >= opening_high:
        return 0
    if body_strength > -float(minimum_body):
        return 0
    if flow_imbalance > -float(minimum_flow):
        return 0
    return -1


def _opening_failure_invalidated(direction, close, opening_high):
    return (
        int(direction) < 0
        and np.isfinite(close)
        and np.isfinite(opening_high)
        and float(close) > float(opening_high)
    )


def _warmed_indicator_value(value, sample_size, required_size):
    """Do not expose a partial rolling-window value to the next bar."""
    if int(sample_size) < int(required_size):
        return np.nan
    return value


def _session_allowed(current_dt, prefix):
    """Product-specific entry windows; exits remain allowed all session."""
    if current_dt is None:
        return False
    prefix = str(prefix).upper()
    if current_dt.hour >= 21 or current_dt.hour < 4:
        return False
    if prefix == "AG":
        return 9 <= current_dt.hour < 12
    return (9 <= current_dt.hour < 12) or (13 <= current_dt.hour < 15)


def _ema_series(values, span):
    """Return an EMA series using only the supplied chronological values."""
    if span <= 0 or len(values) == 0:
        return np.asarray([], dtype=float)
    alpha = 2.0 / (float(span) + 1.0)
    result = np.empty(len(values), dtype=float)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def _higher_tf_bias(
    closes,
    trend_bars,
    threshold,
    fast_bars=PARAM_HIGHER_TF_FAST_BARS,
    slow_bars=PARAM_HIGHER_TF_SLOW_BARS,
    slope_bars=PARAM_HIGHER_TF_SLOPE_BARS,
    efficiency_threshold=PARAM_HIGHER_TF_EFFICIENCY_THRESHOLD,
):
    """Return direction from closed higher-timeframe bars only.

    A directional bias requires net movement, EMA alignment, EMA slope, and
    sufficient directional efficiency. Choppy paths with a similar endpoint
    are therefore classified as neutral instead of as a trend.
    """
    values = np.asarray(list(closes), dtype=float)
    values = values[np.isfinite(values)]
    trend_bars = int(trend_bars)
    fast_bars = int(fast_bars)
    slow_bars = int(slow_bars)
    slope_bars = int(slope_bars)
    if (
        trend_bars <= 0
        or fast_bars <= 0
        or slow_bars <= fast_bars
        or slope_bars <= 0
        or not 0.0 <= float(efficiency_threshold) <= 1.0
    ):
        raise ValueError("invalid higher-timeframe trend parameters")
    required = max(trend_bars + 1, slow_bars, slope_bars + 1)
    if len(values) < required:
        return 0

    change = values[-1] / values[-1 - trend_bars] - 1.0
    fast = _ema_series(values, fast_bars)
    slow = _ema_series(values, slow_bars)
    fast_slope = fast[-1] / fast[-1 - slope_bars] - 1.0
    window = values[-slow_bars:]
    path = float(np.abs(np.diff(window)).sum())
    efficiency = abs(float(window[-1] - window[0])) / path if path > 0 else 0.0

    directional = efficiency >= float(efficiency_threshold)
    if (
        directional
        and change > float(threshold)
        and fast[-1] > slow[-1]
        and fast_slope > 0
    ):
        return 1
    if (
        directional
        and change < -float(threshold)
        and fast[-1] < slow[-1]
        and fast_slope < 0
    ):
        return -1
    return 0


def _stop_triggered(entry_price, low, high, atr, direction, multiplier):
    if not np.isfinite(entry_price) or not np.isfinite(atr) or atr <= 0:
        return False
    if direction > 0:
        return low <= entry_price - multiplier * atr
    if direction < 0:
        return high >= entry_price + multiplier * atr
    return False


def _time_stop_triggered(
    entry_datetime,
    current_datetime,
    max_hold_minutes,
    entry_bar,
    current_bar,
    max_hold_bars,
    execution_delay_minutes=0,
):
    """Use wall-clock time across trading breaks, with a bar-count fallback."""
    if entry_datetime is not None and current_datetime is not None:
        if max_hold_minutes <= 0:
            return False
        elapsed = pd.Timestamp(current_datetime) - pd.Timestamp(entry_datetime)
        planned_elapsed = elapsed + pd.Timedelta(minutes=max(0, int(execution_delay_minutes)))
        return planned_elapsed >= pd.Timedelta(minutes=int(max_hold_minutes))
    return (
        max_hold_bars > 0
        and entry_bar >= 0
        and current_bar - entry_bar >= int(max_hold_bars)
    )


def _update_atr(high, low, close, window):
    global g_atr_values, g_previous_close
    if g_previous_close is None or not np.isfinite(g_previous_close):
        true_range = high - low
    else:
        true_range = max(high - low, abs(high - g_previous_close), abs(low - g_previous_close))
    g_atr_values.append(float(true_range))
    while len(g_atr_values) > window:
        g_atr_values.popleft()
    g_previous_close = close
    if len(g_atr_values) < window:
        return np.nan
    return float(np.mean(g_atr_values))


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


def _update_local_rolling(price, volume, window_bars):
    volume = max(float(volume), 0.0)
    g_local_values.append((float(price), volume))
    while len(g_local_values) > int(window_bars):
        g_local_values.popleft()
    if len(g_local_values) < int(window_bars):
        return np.nan, np.nan, len(g_local_values)
    prices = np.asarray([item[0] for item in g_local_values], dtype=float)
    volumes = np.asarray([item[1] for item in g_local_values], dtype=float)
    total_volume = float(volumes.sum())
    if total_volume <= 0:
        return np.nan, np.nan, len(g_local_values)
    vwap = float(np.dot(prices, volumes) / total_volume)
    variance = max(float(np.dot(prices * prices, volumes) / total_volume - vwap * vwap), 0.0)
    return vwap, float(np.sqrt(variance)), len(g_local_values)


def _previous_auxiliary_range(high, low, volume, range_bars, volume_bars):
    previous_high = max(g_recent_highs) if len(g_recent_highs) >= int(range_bars) else np.nan
    previous_low = min(g_recent_lows) if len(g_recent_lows) >= int(range_bars) else np.nan
    previous_volume_median = (
        float(np.median(g_recent_volumes))
        if len(g_recent_volumes) >= int(volume_bars)
        else np.nan
    )
    g_recent_highs.append(float(high))
    g_recent_lows.append(float(low))
    g_recent_volumes.append(float(volume))
    while len(g_recent_highs) > int(range_bars):
        g_recent_highs.popleft()
        g_recent_lows.popleft()
    while len(g_recent_volumes) > int(volume_bars):
        g_recent_volumes.popleft()
    return previous_high, previous_low, previous_volume_median


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


def _close_position(api, pos, reason, order_type="next_bar_open"):
    direction = "LONG" if pos > 0 else "SHORT"
    _fw.g_bridge.register_pending_close(api, direction)
    if pos > 0:
        api.sell(volume=pos, order_type=order_type, reason=reason)
    else:
        api.buycover(volume=-pos, order_type=order_type, reason=reason)


def _closed_higher_tf_closes(api, current_dt):
    source = api.get_data_source(1)
    if source is None or source.data is None or source.data.empty or "close" not in source.data.columns:
        return []
    period_minutes = api.get_param("higher_period_minutes", PARAM_HIGHER_PERIOD_MINUTES)
    close_times = pd.DatetimeIndex(source.data.index) + pd.Timedelta(minutes=int(period_minutes))
    closed = source.data.loc[close_times <= pd.Timestamp(current_dt), "close"]
    return closed.dropna().tolist()


def _direction_allowed(direction, higher_bias):
    if higher_bias > 0 and direction < 0:
        return False
    if higher_bias < 0 and direction > 0:
        return False
    return True


def _submit_entry(api, direction, bar_count, setup, reason, reference=np.nan):
    """Size and submit one next-open entry through the shared account bridge."""
    global g_pending_entry_setup, g_pending_entry_reference
    if _fw.g_bridge.check_cooling_down("LONG", bar_count) or _fw.g_bridge.check_cooling_down(
        "SHORT", bar_count
    ):
        return False
    raw_price = _get_raw_price(api)
    contract_mult = api.get_param("contract_multiplier", PARAM_CONTRACT_MULTIPLIER)
    margin_rate = api.get_param("margin_rate", PARAM_MARGIN_RATE_FALLBACK)
    max_volume = min(
        int(api.get_param("max_volume", PARAM_MAX_VOLUME_OVERRIDE)),
        PARAM_MAX_VOLUME_OVERRIDE,
    )
    volume_to_trade = calc_position_size(api, contract_mult, raw_price, margin_rate)
    volume_to_trade = max(1, min(int(volume_to_trade), max_volume))
    g_pending_entry_setup = str(setup)
    g_pending_entry_reference = float(reference) if np.isfinite(reference) else np.nan
    if direction > 0:
        api.buy(volume=volume_to_trade, order_type="next_bar_open", reason=reason)
    else:
        api.sellshort(volume=volume_to_trade, order_type="next_bar_open", reason=reason)
    return True


def handle_bar(api: StrategyAPI):
    global g_cache, g_entry_price, g_entry_raw_price, g_entry_bar, g_entry_datetime, g_session_id
    global g_pending_z, g_pending_direction, g_pending_bar
    global g_previous_local_z, g_previous_core_z, g_previous_signal_close
    global g_pending_entry_setup, g_entry_setup, g_last_auxiliary_signal_bar
    global g_pending_entry_reference, g_entry_reference
    global g_opening_high, g_opening_low, g_previous_session_high

    vwap_minutes = api.get_param("vwap_minutes", PARAM_VWAP_MINUTES)
    local_vwap_minutes = api.get_param("local_vwap_minutes", PARAM_LOCAL_VWAP_MINUTES)
    bar_minutes = api.get_param("bar_minutes", PARAM_BAR_MINUTES)
    entry_z = api.get_param("entry_z", PARAM_ENTRY_Z)
    exit_z = api.get_param("exit_z", PARAM_EXIT_Z)
    atr_window = api.get_param("atr_window", PARAM_ATR_WINDOW)
    atr_stop_multiplier = api.get_param("atr_stop_multiplier", PARAM_ATR_STOP_MULTIPLIER)
    max_hold_bars = api.get_param("max_hold_bars", PARAM_MAX_HOLD_BARS)
    max_hold_minutes = api.get_param("max_hold_minutes", PARAM_MAX_HOLD_MINUTES)
    entry_confirmation_bars = max(
        0, int(api.get_param("entry_confirmation_bars", PARAM_ENTRY_CONFIRMATION_BARS))
    )
    enable_core_mean_reversion = bool(
        api.get_param("enable_core_mean_reversion", PARAM_ENABLE_CORE_MEAN_REVERSION)
    )
    enable_trend_pullback = bool(
        api.get_param("enable_trend_pullback", PARAM_ENABLE_TREND_PULLBACK)
    )
    enable_trend_breakout = bool(
        api.get_param("enable_trend_breakout", PARAM_ENABLE_TREND_BREAKOUT)
    )
    enable_opening_failure_short = bool(
        api.get_param("enable_opening_failure_short", PARAM_ENABLE_OPENING_FAILURE_SHORT)
    )
    auxiliary_max_core_z = api.get_param("auxiliary_max_core_z", PARAM_AUXILIARY_MAX_CORE_Z)
    auxiliary_cooldown_bars = int(
        api.get_param("auxiliary_cooldown_bars", PARAM_AUXILIARY_COOLDOWN_BARS)
    )
    opening_failure_minimum_body = api.get_param(
        "opening_failure_minimum_body", PARAM_OPENING_FAILURE_MINIMUM_BODY
    )
    opening_failure_minimum_flow = api.get_param(
        "opening_failure_minimum_flow", PARAM_OPENING_FAILURE_MINIMUM_FLOW
    )
    pullback_z = api.get_param("pullback_z", PARAM_PULLBACK_Z)
    pullback_reclaim_delta = api.get_param(
        "pullback_reclaim_delta", PARAM_PULLBACK_RECLAIM_DELTA
    )
    pullback_minimum_body = api.get_param("pullback_minimum_body", PARAM_PULLBACK_MINIMUM_BODY)
    pullback_minimum_flow = api.get_param("pullback_minimum_flow", PARAM_PULLBACK_MINIMUM_FLOW)
    breakout_lookback_bars = int(
        api.get_param("breakout_lookback_bars", PARAM_BREAKOUT_LOOKBACK_BARS)
    )
    breakout_volume_lookback_bars = int(
        api.get_param(
            "breakout_volume_lookback_bars", PARAM_BREAKOUT_VOLUME_LOOKBACK_BARS
        )
    )
    breakout_minimum_body = api.get_param("breakout_minimum_body", PARAM_BREAKOUT_MINIMUM_BODY)
    breakout_minimum_flow = api.get_param("breakout_minimum_flow", PARAM_BREAKOUT_MINIMUM_FLOW)
    breakout_volume_multiplier = api.get_param(
        "breakout_volume_multiplier", PARAM_BREAKOUT_VOLUME_MULTIPLIER
    )
    higher_trend_bars = api.get_param("higher_trend_bars", PARAM_HIGHER_TF_TREND_BARS)
    higher_trend_threshold = api.get_param("higher_trend_threshold", PARAM_HIGHER_TF_TREND_THRESHOLD)
    higher_fast_bars = api.get_param("higher_fast_bars", PARAM_HIGHER_TF_FAST_BARS)
    higher_slow_bars = api.get_param("higher_slow_bars", PARAM_HIGHER_TF_SLOW_BARS)
    higher_slope_bars = api.get_param("higher_slope_bars", PARAM_HIGHER_TF_SLOPE_BARS)
    higher_efficiency_threshold = api.get_param(
        "higher_efficiency_threshold", PARAM_HIGHER_TF_EFFICIENCY_THRESHOLD
    )
    evaluation_start = pd.Timestamp(api.get_param("evaluation_start", "1900-01-01"))
    tail_lock_minutes = api.get_param("tail_lock_minutes", PARAM_TAIL_LOCK_MINUTES)
    break_lock_minutes = api.get_param("break_lock_minutes", PARAM_BREAK_LOCK_MINUTES)
    symbol_prefix = api.get_param("symbol_prefix", PARAM_SYMBOL_PREFIX)
    consecutive_limit = api.get_param("consecutive_loss_limit", PARAM_CONSECUTIVE_LOSS_LIMIT)
    cooldown_bars = api.get_param("cooldown_bars", PARAM_COOLDOWN_BARS)

    window_bars = max(1, int(round(vwap_minutes / bar_minutes)))
    local_window_bars = max(1, int(round(local_vwap_minutes / bar_minutes)))
    min_bars = max(
        window_bars,
        local_window_bars,
        int(atr_window),
        breakout_lookback_bars,
        breakout_volume_lookback_bars,
    )
    bar_count = api.get_idx()
    if bar_count < min_bars:
        return

    klines = api.get_klines(index=0)
    if klines is None or len(klines) < min_bars:
        return
    if not {"high", "low", "close", "volume"}.issubset(klines.columns):
        api.log("[VWAP optimized] missing OHLCV column; skip bar")
        return

    current_dt = _current_datetime(klines)
    current_session = _session_key(current_dt)
    session_changed = g_session_id is not None and current_session != g_session_id
    session_reset_ordered = False

    # Step 2: match prior next-bar orders; Step 3: confirm P&L.
    _fw.g_bridge.confirm_closed_trades(
        api,
        bar_count,
        consecutive_loss_limit=consecutive_limit,
        cooldown_bars=cooldown_bars,
    )
    pos = api.get_pos(index=0)

    if pos != 0 and g_entry_price == 0.0:
        fill_raw = _get_last_fill_raw_price(api)
        fill_adj = _get_last_fill_adj_price(api)
        if fill_raw > 0:
            g_entry_raw_price = fill_raw
            g_entry_price = fill_adj if fill_adj > 0 else float(klines["close"].values[-1])
            g_entry_bar = bar_count
            g_entry_datetime = current_dt
            g_entry_setup = g_pending_entry_setup or "core_mean_reversion"
            g_entry_reference = g_pending_entry_reference
            g_pending_entry_setup = None
            g_pending_entry_reference = np.nan
            _fw.g_bridge.snapshot_balance_at_entry(api)

    if session_changed:
        if pos != 0:
            _close_position(api, pos, "session_reset")
            session_reset_ordered = True
            g_entry_price = 0.0
            g_entry_raw_price = 0.0
            g_entry_bar = 0
            g_entry_datetime = None
        _reset_signal_state()
    g_session_id = current_session

    # Step 3.5: preserve the framework's rollover-gap compensation.
    if pos != 0 and g_entry_raw_price > 0 and g_entry_price > 0:
        contract_mult = api.get_param("contract_multiplier", PARAM_CONTRACT_MULTIPLIER)
        margin_rate = api.get_param("margin_rate", PARAM_MARGIN_RATE_FALLBACK)
        rollover = _fw.compensate_rollover_gap(
            api, g_entry_price, g_entry_raw_price, contract_mult, margin_rate
        )
        if rollover["compensated"]:
            g_entry_price = rollover["entry_price"]
            g_entry_raw_price = rollover["entry_raw_price"]

    if bar_count <= g_cache.n:
        return

    high = float(klines["high"].values[-1])
    low = float(klines["low"].values[-1])
    close = float(klines["close"].values[-1])
    volume = float(klines["volume"].values[-1])
    previous_session_high = g_previous_session_high
    g_opening_high, g_opening_low = _update_opening_range(
        current_dt, high, low, g_opening_high, g_opening_low
    )
    g_previous_session_high = high
    atr = _update_atr(high, low, close, int(atr_window))
    vwap, dispersion, sample_size = _update_rolling(close, volume, window_bars)
    local_vwap, local_dispersion, local_sample_size = _update_local_rolling(
        close, volume, local_window_bars
    )
    previous_high, previous_low, previous_volume_median = _previous_auxiliary_range(
        high,
        low,
        volume,
        breakout_lookback_bars,
        breakout_volume_lookback_bars,
    )
    higher_closes = _closed_higher_tf_closes(api, current_dt)
    higher_bias = _higher_tf_bias(
        higher_closes,
        int(higher_trend_bars),
        higher_trend_threshold,
        fast_bars=int(higher_fast_bars),
        slow_bars=int(higher_slow_bars),
        slope_bars=int(higher_slope_bars),
        efficiency_threshold=float(higher_efficiency_threshold),
    )
    _set_custom_cache(vwap, dispersion, "range")
    g_cache.set("higher_tf_bias", np.asarray([higher_bias], dtype=float))
    g_cache.n = bar_count

    if session_reset_ordered:
        return

    deviation = close - vwap if np.isfinite(vwap) else np.nan
    zscore = deviation / dispersion if np.isfinite(dispersion) and dispersion > 1e-12 else np.nan
    local_deviation = close - local_vwap if np.isfinite(local_vwap) else np.nan
    local_zscore = (
        local_deviation / local_dispersion
        if np.isfinite(local_dispersion) and local_dispersion > 1e-12
        else np.nan
    )
    previous_local_z = g_previous_local_z
    previous_core_z = g_previous_core_z
    previous_signal_close = g_previous_signal_close
    g_previous_local_z = _warmed_indicator_value(
        local_zscore, local_sample_size, local_window_bars
    )
    g_previous_core_z = _warmed_indicator_value(zscore, sample_size, window_bars)
    g_previous_signal_close = close
    if not np.isfinite(zscore) or sample_size < window_bars:
        return
    if pd.Timestamp(current_dt) < evaluation_start:
        return

    # Step 5: exits first; every decision executes at the next bar open.
    if pos != 0:
        direction = 1 if pos > 0 else -1
        trend_auxiliary_position = g_entry_setup in {"trend_pullback", "trend_breakout"}
        opening_failure_position = g_entry_setup == "opening_failure_short"
        auxiliary_position = trend_auxiliary_position or opening_failure_position
        stop_hit = not opening_failure_position and _stop_triggered(
            g_entry_price, low, high, atr, direction, float(atr_stop_multiplier)
        )
        mean_reverted = not auxiliary_position and _mean_reversion_exit_triggered(
            direction, zscore, exit_z
        )
        trend_failed = trend_auxiliary_position and higher_bias == -direction
        opening_failure_invalidated = opening_failure_position and _opening_failure_invalidated(
            direction, close, g_entry_reference
        )
        time_stopped = _time_stop_triggered(
            g_entry_datetime,
            current_dt,
            int(max_hold_minutes),
            g_entry_bar,
            bar_count,
            int(max_hold_bars),
            execution_delay_minutes=int(bar_minutes),
        )
        tail_exit = _tail_locked(current_dt, str(symbol_prefix).upper(), tail_lock_minutes)
        break_exit = _break_locked(current_dt, str(symbol_prefix).upper(), break_lock_minutes)
        if (
            stop_hit
            or mean_reverted
            or trend_failed
            or opening_failure_invalidated
            or time_stopped
            or tail_exit
            or break_exit
        ):
            reason = (
                "session_break_exit"
                if break_exit
                else "session_tail_exit"
                if tail_exit
                else "atr_stop"
                if stop_hit
                else "opening_range_recovery"
                if opening_failure_invalidated
                else "trend_regime_failure"
                if trend_failed
                else "time_stop"
                if time_stopped
                else "vwap_reversion"
            )
            _close_position(api, pos, reason, order_type=_exit_order_type(reason))
            g_entry_price = 0.0
            g_entry_raw_price = 0.0
            g_entry_bar = 0
            g_entry_datetime = None
            g_pending_z = None
            g_pending_direction = 0
            g_pending_bar = -1
            g_pending_entry_setup = None
            g_entry_setup = None
            g_pending_entry_reference = np.nan
            g_entry_reference = np.nan
        return

    # A pending extreme must be confirmed by the following closed bar.
    if g_pending_bar >= 0:
        if _entry_confirmation_ready(g_pending_bar, bar_count, entry_confirmation_bars):
            pending_direction = g_pending_direction
            confirmed = (
                entry_confirmation_bars == 0
                or _confirm_reversion(g_pending_z, zscore)
            )
            g_pending_z = None
            g_pending_direction = 0
            g_pending_bar = -1
            if (
                confirmed
                and abs(zscore) > exit_z
                and _session_allowed(current_dt, symbol_prefix)
                and not _tail_locked(current_dt, str(symbol_prefix).upper(), tail_lock_minutes)
                and not _break_locked(current_dt, str(symbol_prefix).upper(), break_lock_minutes)
                and _direction_allowed(pending_direction, higher_bias)
            ):
                _submit_entry(
                    api,
                    pending_direction,
                    bar_count,
                    "core_mean_reversion",
                    "confirmed_oversold" if pending_direction > 0 else "confirmed_overbought",
                )
            return
        g_pending_z = None
        g_pending_direction = 0
        g_pending_bar = -1

    if not _session_allowed(current_dt, symbol_prefix):
        return
    if _tail_locked(current_dt, str(symbol_prefix).upper(), tail_lock_minutes):
        return
    if _break_locked(current_dt, str(symbol_prefix).upper(), break_lock_minutes):
        return

    action, direction = (
        _extreme_entry_action(zscore, entry_z, entry_confirmation_bars)
        if enable_core_mean_reversion
        else ("none", 0)
    )
    if action != "none":
        if not _direction_allowed(direction, higher_bias):
            return
        if action == "submit":
            _submit_entry(
                api,
                direction,
                bar_count,
                "core_mean_reversion",
                "oversold_extreme" if direction > 0 else "overbought_extreme",
            )
        else:
            g_pending_z = zscore
            g_pending_direction = direction
            g_pending_bar = bar_count
        return

    price_range = high - low
    body_strength = (
        (close - float(klines["open"].values[-1])) / price_range if price_range > 0 else 0.0
    )
    flow_imbalance = np.nan
    if {"B", "S"}.issubset(klines.columns):
        buy_flow = float(klines["B"].values[-1])
        sell_flow = float(klines["S"].values[-1])
        total_flow = buy_flow + sell_flow
        flow_imbalance = (buy_flow - sell_flow) / total_flow if total_flow > 0 else 0.0

    opening_failure_ready = (
        enable_opening_failure_short
        and _opening_failure_window(current_dt, symbol_prefix)
        and _auxiliary_holding_window(current_dt, symbol_prefix, max_hold_minutes, bar_minutes)
        and (
            g_last_auxiliary_signal_bar < 0
            or bar_count - g_last_auxiliary_signal_bar > auxiliary_cooldown_bars
        )
    )
    if opening_failure_ready:
        opening_failure_direction = _opening_failure_short_direction(
            higher_bias,
            previous_session_high,
            g_opening_high,
            close,
            previous_core_z,
            zscore,
            body_strength,
            flow_imbalance,
            auxiliary_max_core_z,
            opening_failure_minimum_body,
            opening_failure_minimum_flow,
        )
        if opening_failure_direction and _submit_entry(
            api,
            opening_failure_direction,
            bar_count,
            "opening_failure_short",
            "opening_high_breakout_failure",
            reference=g_opening_high,
        ):
            g_last_auxiliary_signal_bar = bar_count
            return

    auxiliary_ready = (
        (enable_trend_pullback or enable_trend_breakout)
        and local_sample_size >= local_window_bars
        and _auxiliary_holding_window(
            current_dt,
            symbol_prefix,
            max_hold_minutes,
            bar_minutes,
        )
        and (
            g_last_auxiliary_signal_bar < 0
            or bar_count - g_last_auxiliary_signal_bar > auxiliary_cooldown_bars
        )
    )
    if not auxiliary_ready:
        return

    auxiliary_direction = 0
    auxiliary_setup = None
    auxiliary_reason = None
    if enable_trend_pullback:
        auxiliary_direction = _trend_pullback_direction(
            higher_bias,
            previous_local_z,
            local_zscore,
            previous_core_z,
            zscore,
            previous_signal_close,
            close,
            body_strength,
            flow_imbalance,
            auxiliary_max_core_z,
            pullback_z,
            pullback_reclaim_delta,
            pullback_minimum_body,
            pullback_minimum_flow,
        )
        if auxiliary_direction:
            auxiliary_setup = "trend_pullback"
            auxiliary_reason = "trend_pullback_reclaim"
    if not auxiliary_direction and enable_trend_breakout:
        auxiliary_direction = _trend_breakout_direction(
            higher_bias,
            previous_core_z,
            zscore,
            close,
            previous_high,
            previous_low,
            body_strength,
            flow_imbalance,
            volume,
            previous_volume_median,
            auxiliary_max_core_z,
            breakout_minimum_body,
            breakout_minimum_flow,
            breakout_volume_multiplier,
        )
        if auxiliary_direction:
            auxiliary_setup = "trend_breakout"
            auxiliary_reason = "trend_volume_breakout"
    if auxiliary_direction and _submit_entry(
        api,
        auxiliary_direction,
        bar_count,
        auxiliary_setup,
        auxiliary_reason,
    ):
        g_last_auxiliary_signal_bar = bar_count
