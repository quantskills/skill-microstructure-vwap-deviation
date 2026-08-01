import pytest

from research.index_matrix_contract import period_minutes, source_id, validate_period_pair


def test_period_minutes_supports_requested_periods():
    assert [period_minutes(value) for value in ("5m", "15m", "30m", "60m", "90m", "120m")] == [5, 15, 30, 60, 90, 120]


def test_period_pair_requires_higher_timeframe():
    assert validate_period_pair("5m", "30m") is True
    assert validate_period_pair("15m", "60m") is True
    assert validate_period_pair("30m", "30m") is False
    assert validate_period_pair("60m", "30m") is False


def test_invalid_period_is_rejected():
    with pytest.raises(ValueError):
        period_minutes("1h")


def test_source_id_is_stable():
    assert source_id("IF888", "90m", "1") == "IF888_90m_1"
