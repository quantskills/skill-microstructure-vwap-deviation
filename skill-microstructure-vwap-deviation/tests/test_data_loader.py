import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from data_loader import freeze_dataset, load_minute_bars, normalize_minute_bars


def raw_bars():
    return pd.DataFrame(
        {
            "datetime": ["2026-01-02 09:31", "2026-01-02 09:30"],
            "open": [101, 100],
            "high": [102, 101],
            "low": [100, 99],
            "close": [101, 100],
            "volume": [20, 10],
        }
    )


def test_normalize_minute_bars_sorts_and_standardizes():
    result = normalize_minute_bars(raw_bars())

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.is_monotonic_increasing
    assert list(result["close"]) == [100, 101]
    assert {"open", "high", "low", "close", "volume", "amount"}.issubset(result.columns)


def test_load_minute_bars_routes_stock_to_official_stock_api(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        get_stock_min=lambda **kwargs: calls.append(("stock", kwargs)) or raw_bars(),
        get_future_min=lambda **kwargs: calls.append(("future", kwargs)) or raw_bars(),
    )
    monkeypatch.setitem(sys.modules, "panda_data", fake)

    result = load_minute_bars("000001.SZ", "20260101", "20260103", "1m")

    assert len(result) == 2
    assert calls[0][0] == "stock"
    assert calls[0][1]["frequency"] == "1m"
    assert result.attrs["data_source"] == "panda_data.get_stock_min"


def test_load_minute_bars_routes_future_to_official_future_api(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        get_stock_min=lambda **kwargs: calls.append(("stock", kwargs)) or raw_bars(),
        get_future_min=lambda **kwargs: calls.append(("future", kwargs)) or raw_bars(),
    )
    monkeypatch.setitem(sys.modules, "panda_data", fake)

    result = load_minute_bars("RB2405.SHF", "20260101", "20260103", "5m")

    assert len(result) == 2
    assert calls[0][0] == "future"
    assert calls[0][1]["symbol"] == "RB2405.SHF"


def test_freeze_dataset_hash_is_stable(tmp_path):
    result = freeze_dataset(raw_bars(), tmp_path, "000001.SZ", adjustment_mode="raw")
    again = freeze_dataset(raw_bars(), tmp_path / "again", "000001.SZ", adjustment_mode="raw")

    assert result["bars_hash"] == again["bars_hash"]
    assert Path(result["bars_path"]).is_absolute()
    assert Path(result["bars_path"]).exists()
    assert result["adjustment_mode"] == "raw"
