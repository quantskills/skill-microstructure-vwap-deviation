# -*- coding: utf-8 -*-

from datetime import datetime

import pandas as pd

from vwap_deviation_optimized_strategy import (
    _confirm_reversion,
    _closed_higher_tf_closes,
    _entry_confirmation_ready,
    _break_locked,
    _auxiliary_holding_window,
    _extreme_entry_action,
    _exit_order_type,
    _higher_tf_bias,
    _mean_reversion_exit_triggered,
    _opening_failure_invalidated,
    _opening_failure_short_direction,
    _opening_failure_window,
    _update_opening_range,
    _warmed_indicator_value,
    _tail_locked,
    _session_allowed,
    _stop_triggered,
    _trend_breakout_direction,
    _trend_pullback_direction,
    _time_stop_triggered,
)


def test_confirmation_requires_same_side_and_smaller_absolute_z():
    assert _confirm_reversion(2.6, 2.2)
    assert _confirm_reversion(-2.6, -2.2)
    assert not _confirm_reversion(2.6, 2.8)
    assert not _confirm_reversion(2.6, -0.2)


def test_vwap_crossing_is_an_exit_even_when_zscore_overshoots_exit_band():
    assert _mean_reversion_exit_triggered(1, 0.8, 0.5)
    assert _mean_reversion_exit_triggered(-1, -0.8, 0.5)
    assert not _mean_reversion_exit_triggered(1, -1.0, 0.5)
    assert not _mean_reversion_exit_triggered(-1, 1.0, 0.5)


def test_entry_confirmation_bars_are_configurable():
    assert _entry_confirmation_ready(10, 10, 0)
    assert _entry_confirmation_ready(10, 11, 0)
    assert not _entry_confirmation_ready(10, 10, 1)
    assert _entry_confirmation_ready(10, 11, 1)


def test_exit_order_type_matches_trigger_semantics():
    assert _exit_order_type("atr_stop") == "next_bar_open"
    assert _exit_order_type("vwap_reversion") == "next_bar_open"
    assert _exit_order_type("time_stop") == "next_bar_open"
    assert _exit_order_type("session_tail_exit") == "next_bar_open"


def test_zero_confirmation_extreme_is_submitted_on_the_signal_bar():
    assert _extreme_entry_action(2.5, 2.25, 0) == ("submit", -1)
    assert _extreme_entry_action(-2.5, 2.25, 0) == ("submit", 1)
    assert _extreme_entry_action(2.5, 2.25, 1) == ("pend", -1)
    assert _extreme_entry_action(2.0, 2.25, 0) == ("none", 0)


def test_trend_pullback_requires_reclaim_body_flow_and_core_separation():
    common = dict(
        higher_bias=1,
        previous_local_z=-1.0,
        local_z=-0.5,
        previous_core_z=-1.2,
        core_z=-0.9,
        previous_close=100.0,
        close=101.0,
        body_strength=0.5,
        flow_imbalance=0.2,
        max_core_z=2.25,
        pullback_z=0.8,
        reclaim_delta=0.35,
        minimum_body=0.15,
        minimum_flow=0.05,
    )
    assert _trend_pullback_direction(**common) == 1
    assert _trend_pullback_direction(**{**common, "flow_imbalance": 0.0}) == 0
    assert _trend_pullback_direction(**{**common, "core_z": -2.3}) == 0


def test_trend_breakout_requires_prior_range_volume_and_aligned_flow():
    common = dict(
        higher_bias=-1,
        previous_core_z=0.5,
        core_z=-0.8,
        close=98.0,
        previous_high=102.0,
        previous_low=99.0,
        body_strength=-0.6,
        flow_imbalance=-0.2,
        volume=130.0,
        previous_volume_median=100.0,
        max_core_z=2.25,
        minimum_body=0.35,
        minimum_flow=0.08,
        volume_multiplier=1.15,
    )
    assert _trend_breakout_direction(**common) == -1
    assert _trend_breakout_direction(**{**common, "close": 99.5}) == 0
    assert _trend_breakout_direction(**{**common, "volume": 110.0}) == 0


def test_opening_failure_short_requires_rejection_body_flow_and_non_bullish_bias():
    common = dict(
        higher_bias=0,
        previous_high=101.2,
        opening_high=101.0,
        close=100.8,
        previous_core_z=1.4,
        core_z=1.0,
        body_strength=-0.4,
        flow_imbalance=-0.2,
        max_core_z=2.25,
        minimum_body=0.20,
        minimum_flow=0.05,
    )
    assert _opening_failure_short_direction(**common) == -1
    assert _opening_failure_short_direction(**{**common, "higher_bias": -1}) == -1
    assert _opening_failure_short_direction(**{**common, "higher_bias": 1}) == 0
    assert _opening_failure_short_direction(**{**common, "previous_high": 100.9}) == 0
    assert _opening_failure_short_direction(**{**common, "close": 101.1}) == 0
    assert _opening_failure_short_direction(**{**common, "body_strength": -0.1}) == 0
    assert _opening_failure_short_direction(**{**common, "flow_imbalance": 0.0}) == 0
    assert _opening_failure_short_direction(**{**common, "core_z": 2.3}) == 0


def test_opening_failure_short_uses_the_post_opening_morning_window_only():
    assert _opening_failure_window(datetime(2024, 1, 2, 10, 0), "IM")
    assert _opening_failure_window(datetime(2024, 1, 2, 10, 50), "IM")
    assert not _opening_failure_window(datetime(2024, 1, 2, 9, 55), "IM")
    assert not _opening_failure_window(datetime(2024, 1, 2, 10, 55), "IM")
    assert not _opening_failure_window(datetime(2024, 1, 2, 10, 30), "AU")


def test_opening_range_stops_updating_after_the_first_six_bars():
    opening_high, opening_low = _update_opening_range(
        datetime(2024, 1, 2, 9, 30), 101.0, 99.0, float("nan"), float("nan")
    )
    opening_high, opening_low = _update_opening_range(
        datetime(2024, 1, 2, 9, 55), 102.0, 98.0, opening_high, opening_low
    )
    assert (opening_high, opening_low) == (102.0, 98.0)
    assert _update_opening_range(
        datetime(2024, 1, 2, 10, 0), 103.0, 97.0, opening_high, opening_low
    ) == (102.0, 98.0)


def test_unwarmed_indicator_is_not_carried_into_the_next_signal_bar():
    assert pd.isna(_warmed_indicator_value(1.5, 11, 12))
    assert _warmed_indicator_value(1.5, 12, 12) == 1.5


def test_opening_failure_short_is_invalidated_only_after_close_recovers_opening_high():
    assert _opening_failure_invalidated(-1, 101.1, 101.0)
    assert not _opening_failure_invalidated(-1, 100.9, 101.0)
    assert not _opening_failure_invalidated(1, 101.1, 101.0)


def test_index_futures_break_lock_covers_the_last_five_minutes_before_lunch_break():
    assert _break_locked(datetime(2024, 1, 2, 11, 25), "IM", 5)
    assert not _break_locked(datetime(2024, 1, 2, 11, 20), "IM", 5)
    assert not _break_locked(datetime(2024, 1, 2, 11, 25), "AU", 5)


def test_auxiliary_entry_requires_a_complete_thirty_minute_holding_window():
    assert _auxiliary_holding_window(datetime(2024, 1, 2, 10, 50), "IM", 30, 5)
    assert not _auxiliary_holding_window(datetime(2024, 1, 2, 10, 55), "IM", 30, 5)
    assert _auxiliary_holding_window(datetime(2024, 1, 2, 13, 55), "IM", 30, 5)
    assert not _auxiliary_holding_window(datetime(2024, 1, 2, 14, 0), "IM", 30, 5)
    assert not _auxiliary_holding_window(datetime(2024, 1, 2, 10, 0), "AU", 30, 5)


def test_index_futures_tail_lock_covers_the_last_30_minutes_of_day_session():
    assert _tail_locked(datetime(2024, 1, 2, 14, 45), "IM", 30)
    assert not _tail_locked(datetime(2024, 1, 2, 14, 25), "IM", 30)


def test_session_filter_is_product_specific():
    assert _session_allowed(datetime(2024, 1, 2, 10, 0), "AU")
    assert not _session_allowed(datetime(2024, 1, 2, 22, 0), "AU")
    assert _session_allowed(datetime(2024, 1, 2, 10, 0), "AG")
    assert not _session_allowed(datetime(2024, 1, 2, 14, 0), "AG")


def test_atr_stop_is_directional():
    assert _stop_triggered(100.0, 95.0, 105.0, 2.0, 1, 1.5)
    assert _stop_triggered(100.0, 95.0, 105.0, 2.0, -1, 1.5)
    assert not _stop_triggered(100.0, 99.0, 101.0, 2.0, 1, 1.5)


def test_time_stop_uses_wall_clock_and_keeps_bar_fallback():
    entry = datetime(2024, 1, 2, 11, 15)
    assert _time_stop_triggered(entry, datetime(2024, 1, 2, 11, 45), 30, 9, 10, 6)
    assert not _time_stop_triggered(entry, datetime(2024, 1, 2, 11, 44), 30, 9, 10, 6)
    assert not _time_stop_triggered(entry, datetime(2024, 1, 2, 11, 44), 60, 9, 15, 6)
    assert _time_stop_triggered(None, datetime(2024, 1, 2, 13, 0), 30, 9, 15, 6)


def test_time_stop_accounts_for_next_open_execution_delay():
    entry = datetime(2024, 1, 2, 10, 0)
    assert _time_stop_triggered(
        entry,
        datetime(2024, 1, 2, 10, 25),
        30,
        10,
        15,
        6,
        execution_delay_minutes=5,
    )


def test_higher_timeframe_bias_uses_closed_bars_only():
    assert _higher_tf_bias(
        [100.0, 100.4, 100.8, 101.2, 101.6, 102.0, 102.4, 102.8],
        3,
        0.005,
    ) == 1
    assert _higher_tf_bias(
        [102.8, 102.4, 102.0, 101.6, 101.2, 100.8, 100.4, 100.0],
        3,
        0.005,
    ) == -1
    assert _higher_tf_bias(
        [100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0],
        3,
        0.005,
    ) == 0


def test_higher_timeframe_source_is_unavailable_until_its_bar_has_closed():
    class Source:
        data = pd.DataFrame(
            {"close": [101.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02 10:00:00")]),
        )
        original_data = data.copy()

    class API:
        @staticmethod
        def get_data_source(index):
            assert index == 1
            return Source()

        @staticmethod
        def get_param(name, default):
            assert name == "higher_period_minutes"
            return 120

    assert _closed_higher_tf_closes(API(), datetime(2024, 1, 2, 11, 59)) == []
    assert _closed_higher_tf_closes(API(), datetime(2024, 1, 2, 12, 0)) == [101.0]


def test_higher_timeframe_bias_requires_efficiency_not_only_net_change():
    closes = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 102.0]

    assert _higher_tf_bias(closes, 3, 0.005) == 0


def test_higher_timeframe_bias_warms_up_before_classifying():
    assert _higher_tf_bias([100.0, 101.0, 102.0], 3, 0.005) == 0
