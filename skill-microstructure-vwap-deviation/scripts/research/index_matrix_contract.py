"""Pure validation helpers shared by the index MTF preparation and runner."""

from __future__ import annotations

import re


_PERIOD_RE = re.compile(r"^(5|15|30|60|90|120)m$")


def period_minutes(period: str) -> int:
    value = str(period).lower()
    if not _PERIOD_RE.fullmatch(value):
        raise ValueError(f"Unsupported period: {period!r}")
    return int(value[:-1])


def validate_period_pair(base_period: str, higher_period: str) -> bool:
    return period_minutes(higher_period) > period_minutes(base_period)


def source_id(symbol: str, period: str, adjust_type: str = "1") -> str:
    period_minutes(period)
    return f"{str(symbol).upper()}_{period}_{adjust_type}"
