import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vwap_strategy import VWAPConfig, generate_multi_symbol_signals, generate_signals


def bars(prices, volumes=None, start="2026-01-05 09:30"):
    index = pd.date_range(start, periods=len(prices), freq="min")
    volume = volumes or [1.0] * len(prices)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": volume,
        },
        index=index,
    )


def test_rolling_vwap_uses_current_and_prior_bars_only():
    result = generate_signals(
        bars([100, 102, 104], [1, 1, 2]),
        VWAPConfig(vwap_minutes=3, bar_minutes=1, trend_filter=False),
    )

    assert result.loc[result.index[1], "vwap"] == 101
    assert result.loc[result.index[2], "vwap"] == 102.5


def test_extreme_above_vwap_emits_short_entry():
    result = generate_signals(
        bars([100, 100, 100, 100, 100, 120]),
        VWAPConfig(vwap_minutes=5, bar_minutes=1, trend_filter=False),
    )

    assert result.iloc[-1]["action"] == "enter_short"
    assert result.iloc[-1]["signal"] == -1


def test_return_to_vwap_band_emits_exit():
    result = generate_signals(
        bars([100, 100, 100, 100, 100, 120, 101]),
        VWAPConfig(vwap_minutes=5, bar_minutes=1, trend_filter=False),
    )

    assert result.iloc[-2]["action"] == "enter_short"
    assert result.iloc[-1]["action"] == "exit"


def test_trend_guard_blocks_countertrend_entry():
    result = generate_signals(
        bars([100, 100, 100, 100, 100, 120]),
        VWAPConfig(
            vwap_minutes=5,
            bar_minutes=1,
            trend_filter=True,
            trend_window=3,
            trend_threshold=0.01,
        ),
    )

    assert result.iloc[-1]["action"] == "hold"
    assert result.iloc[-1]["trend_state"] == "up"
    assert result.iloc[-1]["blocked_reason"] == "trend_guard"


def test_tail_guard_blocks_new_entry():
    result = generate_signals(
        bars([100, 100, 100, 100, 100, 120], start="2026-01-05 14:35"),
        VWAPConfig(
            vwap_minutes=5,
            bar_minutes=1,
            trend_filter=False,
            session_end="15:00",
            tail_lock_minutes=30,
        ),
    )

    assert result.iloc[-1]["action"] == "hold"
    assert result.iloc[-1]["blocked_reason"] == "tail_guard"


def test_session_reset_does_not_mix_prior_day_volume():
    result = generate_signals(
        pd.concat(
            [
                bars([100, 100, 100], [100, 100, 100]),
                bars([200], [1], start="2026-01-06 09:30"),
            ]
        ),
        VWAPConfig(vwap_minutes=3, bar_minutes=1, trend_filter=False),
    )

    assert result.iloc[-1]["vwap"] == 200


def test_multi_symbol_state_is_isolated():
    first = bars([100, 100, 100, 100, 100, 120])
    first.insert(0, "symbol", "A")
    second = bars([100, 100, 100, 100, 100, 100])
    second.insert(0, "symbol", "B")
    result = generate_multi_symbol_signals(
        pd.concat([first, second]),
        VWAPConfig(vwap_minutes=5, bar_minutes=1, trend_filter=False, session_end=None),
    )

    assert result[result["symbol"] == "A"].iloc[-1]["action"] == "enter_short"
    assert result[result["symbol"] == "B"].iloc[-1]["action"] == "hold"
