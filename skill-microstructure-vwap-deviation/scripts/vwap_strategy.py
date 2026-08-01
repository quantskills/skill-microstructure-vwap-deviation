"""Incremental minute VWAP deviation signal engine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VWAPConfig:
    """Parameters for one symbol and one bar frequency."""

    vwap_minutes: int = 30
    bar_minutes: int = 1
    entry_z: float = 2.0
    exit_z: float = 0.5
    trend_filter: bool = True
    trend_window: int = 30
    trend_threshold: float = 0.003
    session_end: str | None = "15:00"
    tail_lock_minutes: int = 30
    min_std: float = 1e-12

    @property
    def window_bars(self) -> int:
        if self.vwap_minutes <= 0 or self.bar_minutes <= 0:
            raise ValueError("vwap_minutes and bar_minutes must be positive")
        return max(1, int(round(self.vwap_minutes / self.bar_minutes)))

    def validate(self) -> None:
        if self.entry_z <= 0 or self.exit_z < 0:
            raise ValueError("entry_z must be positive and exit_z cannot be negative")
        if self.trend_window < 2:
            raise ValueError("trend_window must be at least 2")
        if self.tail_lock_minutes < 0:
            raise ValueError("tail_lock_minutes cannot be negative")
        if self.session_end is not None:
            _parse_time(self.session_end)


class RollingVWAP:
    """Volume-weighted rolling mean and standard deviation using current history only."""

    def __init__(self, window: int):
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self._values: deque[tuple[float, float]] = deque(maxlen=window)
        self._sum_volume = 0.0
        self._sum_pv = 0.0
        self._sum_pv2 = 0.0

    def reset(self) -> None:
        self._values.clear()
        self._sum_volume = 0.0
        self._sum_pv = 0.0
        self._sum_pv2 = 0.0

    def update(self, price: float, volume: float) -> tuple[float, float, int]:
        price = float(price)
        volume = max(float(volume), 0.0)
        if len(self._values) == self.window:
            old_price, old_volume = self._values.popleft()
            self._sum_volume -= old_volume
            self._sum_pv -= old_price * old_volume
            self._sum_pv2 -= old_price * old_price * old_volume

        self._values.append((price, volume))
        self._sum_volume += volume
        self._sum_pv += price * volume
        self._sum_pv2 += price * price * volume

        if self._sum_volume <= 0:
            return np.nan, np.nan, len(self._values)
        vwap = self._sum_pv / self._sum_volume
        variance = max(self._sum_pv2 / self._sum_volume - vwap * vwap, 0.0)
        return vwap, float(np.sqrt(variance)), len(self._values)


def _parse_time(value: str) -> time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise ValueError("session_end must use HH:MM") from exc


def _session_key(timestamp: pd.Timestamp, row: pd.Series) -> object:
    if "session" in row and pd.notna(row["session"]):
        return row["session"]
    return timestamp.normalize()


def _tail_locked(timestamp: pd.Timestamp, config: VWAPConfig) -> bool:
    if config.session_end is None or config.tail_lock_minutes == 0:
        return False
    end = _parse_time(config.session_end)
    current = timestamp.time()
    end_dt = datetime.combine(timestamp.date(), end)
    lock_start = (end_dt - timedelta(minutes=config.tail_lock_minutes)).time()
    return current >= lock_start


def _trend_state(history: deque[float], config: VWAPConfig) -> str:
    if len(history) < config.trend_window:
        return "unknown"
    first = history[0]
    if first == 0:
        return "range"
    change = history[-1] / first - 1.0
    if change > config.trend_threshold:
        return "up"
    if change < -config.trend_threshold:
        return "down"
    return "range"


def _empty_output(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vwap": np.nan,
            "std": np.nan,
            "deviation": np.nan,
            "zscore": np.nan,
            "trend_state": "unknown",
            "action": "hold",
            "signal": 0,
            "blocked_reason": "",
            "position_after": 0,
        },
        index=index,
    )


def generate_signals(bars: pd.DataFrame, config: VWAPConfig | None = None) -> pd.DataFrame:
    """Return features and bar-close signal events for one symbol.

    The signal on row t is intended for execution at row t+1 open. The function
    never reads a row after t and resets all rolling state at each session key.
    """

    config = config or VWAPConfig()
    config.validate()
    required = {"close", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if len(bars) == 0:
        return _empty_output(bars.index)

    frame = bars.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()

    rolling = RollingVWAP(config.window_bars)
    trend_history: deque[float] = deque(maxlen=config.trend_window)
    previous_session = object()
    position = 0
    output = []

    for timestamp, row in frame.iterrows():
        session = _session_key(pd.Timestamp(timestamp), row)
        if session != previous_session:
            rolling.reset()
            trend_history.clear()
            position = 0
            previous_session = session

        price = float(row["close"])
        volume = float(row["volume"])
        vwap, dispersion, sample_size = rolling.update(price, volume)
        trend_history.append(price)
        trend = _trend_state(trend_history, config)
        deviation = price - vwap if np.isfinite(vwap) else np.nan
        zscore = deviation / dispersion if np.isfinite(dispersion) and dispersion > config.min_std else np.nan
        action = "hold"
        signal = position
        blocked_reason = ""

        if position and np.isfinite(deviation) and np.isfinite(dispersion):
            if abs(deviation) <= config.exit_z * dispersion:
                action = "exit"
                position = 0
                signal = 0
        elif (
            position == 0
            and sample_size >= config.window_bars
            and np.isfinite(zscore)
            and abs(zscore) >= config.entry_z
        ):
            proposed = -1 if deviation > 0 else 1
            if _tail_locked(pd.Timestamp(timestamp), config):
                blocked_reason = "tail_guard"
            elif config.trend_filter and trend != "range":
                blocked_reason = "trend_guard" if trend in {"up", "down"} else "trend_warmup"
            else:
                action = "enter_short" if proposed == -1 else "enter_long"
                position = proposed
                signal = position

        output.append(
            {
                "vwap": vwap,
                "std": dispersion,
                "deviation": deviation,
                "zscore": zscore,
                "trend_state": trend,
                "action": action,
                "signal": signal,
                "blocked_reason": blocked_reason,
                "position_after": position,
            }
        )

    return pd.DataFrame(output, index=frame.index)


def generate_multi_symbol_signals(
    bars: pd.DataFrame, config: VWAPConfig | None = None
) -> pd.DataFrame:
    """Apply the strategy independently to a `symbol` column."""

    if "symbol" not in bars.columns:
        return generate_signals(bars, config)
    pieces = []
    for symbol, group in bars.groupby("symbol", sort=False):
        signals = generate_signals(group.drop(columns=["symbol"]), config)
        signals.insert(0, "symbol", symbol)
        pieces.append(signals)
    return pd.concat(pieces).sort_index()
