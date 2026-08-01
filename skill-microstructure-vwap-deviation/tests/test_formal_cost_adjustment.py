from datetime import datetime

import pandas as pd

from research.formal_cost_adjustment import apply_close_today_costs


def test_same_day_close_gets_the_close_today_fee_delta_and_curve_adjustment():
    results = {
        "IM888_5m": {
            "initial_capital": 100000.0,
            "equity_curve": pd.Series(
                [100000.0, 100000.0, 100000.0],
                index=pd.date_range("2024-01-02 10:00", periods=3, freq="5min"),
            ),
            "trades": [
                {
                    "datetime": datetime(2024, 1, 2, 10, 0),
                    "raw_price": 5000.0,
                    "price": 5000.0,
                    "volume": 1,
                    "commission": 11.5,
                },
                {
                    "datetime": datetime(2024, 1, 2, 10, 5),
                    "raw_price": 5010.0,
                    "price": 5010.0,
                    "volume": 1,
                    "commission": 11.523,
                    "net_profit": 1000.0,
                    "profit": 1000.0,
                },
            ],
        }
    }

    adjustment = apply_close_today_costs(results, 200, 0.000023, 0.00023)
    expected_delta = 5010.0 * 200 * (0.00023 - 0.000023)

    assert adjustment["total_extra_fee"] == expected_delta
    assert results["IM888_5m"]["trades"][1]["net_profit"] == 1000.0 - expected_delta
    assert results["IM888_5m"]["equity_curve"].iloc[0] == 100000.0
    assert results["IM888_5m"]["equity_curve"].iloc[1] == 100000.0 - expected_delta


def test_overnight_close_keeps_the_regular_fee_and_is_not_adjusted():
    results = {
        "IM888_5m": {
            "initial_capital": 100000.0,
            "equity_curve": pd.Series(
                [100000.0, 100100.0],
                index=pd.date_range("2024-01-02 14:55", periods=2, freq="5min"),
            ),
            "trades": [
                {"datetime": datetime(2024, 1, 2, 14, 55), "raw_price": 5000.0, "volume": 1},
                {
                    "datetime": datetime(2024, 1, 3, 9, 35),
                    "raw_price": 5010.0,
                    "volume": 1,
                    "net_profit": 1000.0,
                },
            ],
        }
    }

    adjustment = apply_close_today_costs(results, 200, 0.000023, 0.00023)

    assert adjustment["total_extra_fee"] == 0.0
    assert results["IM888_5m"]["trades"][1]["net_profit"] == 1000.0
