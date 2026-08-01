"""Official minute data adapter and reproducible dataset freezer."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MINUTE_FREQUENCIES = ("1m", "5m", "15m", "60m")
STANDARD_COLUMNS = ("open", "high", "low", "close", "volume", "amount", "open_interest")
_STOCK = re.compile(r"^\d{6}\.(SZ|SH|BJ)$", re.IGNORECASE)
_FUTURE = re.compile(r"^[A-Za-z]{1,3}\d{3,4}\.[A-Za-z]{3,4}$")
_DOMINANT = re.compile(r"^[A-Za-z]{1,3}(_DOMINANT)?(\.[A-Za-z]{3,4})?$", re.IGNORECASE)


def classify_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        return "unknown"
    value = symbol.strip()
    if _STOCK.fullmatch(value):
        return "stock"
    if _FUTURE.fullmatch(value) or _DOMINANT.fullmatch(value):
        return "future"
    return "unknown"


def _datetime_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.DatetimeIndex(frame.index)
    for name in ("datetime", "date", "time", "timestamp"):
        if name in frame.columns:
            return pd.to_datetime(frame[name], errors="coerce")
    raise ValueError("minute data must contain datetime/date/time/timestamp or a DatetimeIndex")


def normalize_minute_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize official API output to sorted, numeric OHLCV bars."""

    if raw is None or len(raw) == 0:
        empty = pd.DataFrame(columns=STANDARD_COLUMNS)
        empty.index = pd.DatetimeIndex([], name="datetime")
        return empty
    frame = raw.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    aliases = {"vol": "volume", "turnover": "amount", "oi": "open_interest"}
    frame = frame.rename(columns=aliases)
    frame.index = _datetime_index(frame)
    frame = frame[~frame.index.isna()].sort_index()
    frame.index.name = "datetime"

    for column in STANDARD_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["volume"] = frame["volume"].fillna(0.0).clip(lower=0.0)
    frame["amount"] = frame["amount"].where(frame["amount"].notna(), frame["close"] * frame["volume"])
    frame = frame.dropna(subset=["close"])
    if (frame["close"] <= 0).any():
        raise ValueError("close must be positive")
    keep = list(STANDARD_COLUMNS)
    for extra in ("symbol", "session", "dominant_id"):
        if extra in frame.columns:
            keep.append(extra)
    return frame[keep]


def _import_panda_data():
    try:
        import panda_data
    except ImportError as exc:
        raise RuntimeError("panda_data is required for official minute data") from exc
    return panda_data


def _maybe_login(panda_data: Any, username: str | None, password: str | None) -> None:
    user = username or os.environ.get("PANDA_DATA_USERNAME")
    pwd = password or os.environ.get("PANDA_DATA_PASSWORD")
    if user and pwd and hasattr(panda_data, "init_token"):
        panda_data.init_token(username=user, password=pwd)


def load_minute_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    frequency: str = "1m",
    username: str | None = None,
    password: str | None = None,
) -> pd.DataFrame:
    """Load one symbol through the official stock/futures minute API."""

    if frequency not in MINUTE_FREQUENCIES:
        raise ValueError(f"frequency must be one of {MINUTE_FREQUENCIES}")
    asset = classify_symbol(symbol)
    if asset == "unknown":
        raise ValueError(f"unsupported symbol format: {symbol!r}")
    panda_data = _import_panda_data()
    _maybe_login(panda_data, username, password)
    if asset == "stock":
        method = panda_data.get_stock_min
        source = "panda_data.get_stock_min"
    else:
        method = panda_data.get_future_min
        source = "panda_data.get_future_min"
    raw = method(symbol=symbol, start_date=start_date, end_date=end_date, frequency=frequency)
    result = normalize_minute_bars(raw)
    result.attrs.update(
        {
            "data_source": source,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "frequency": frequency,
        }
    )
    return result


def _canonical_csv(frame: pd.DataFrame) -> str:
    normalized = normalize_minute_bars(frame)
    output = normalized.reset_index()
    output["datetime"] = pd.to_datetime(output["datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return output.to_csv(index=False, float_format="%.15g", lineterminator="\n")


def freeze_dataset(
    bars: pd.DataFrame,
    output_dir: str | os.PathLike[str],
    symbol: str,
    adjustment_mode: str = "official",
) -> dict[str, Any]:
    """Write canonical frozen bars and metadata, returning a reproducible manifest fragment."""

    if adjustment_mode not in {"official", "raw", "synthetic"}:
        raise ValueError("adjustment_mode must be official, raw, or synthetic")
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    canonical = _canonical_csv(bars)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    bars_path = target / "bars.csv"
    bars_path.write_text(canonical, encoding="utf-8", newline="")
    metadata = {
        "symbol": symbol,
        "rows": len(normalize_minute_bars(bars)),
        "bars_hash": digest,
        "hash_algorithm": "sha256",
        "adjustment_mode": adjustment_mode,
        "bars_path": str(bars_path),
    }
    (target / "dataset_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return metadata
