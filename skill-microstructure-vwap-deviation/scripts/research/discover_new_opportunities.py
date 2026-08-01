# -*- coding: utf-8 -*-
"""Research independent IM intraday setups on the governed frozen dataset."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PROJECT_DIR.parent
CODEX_SKILL_ROOT = Path.home() / ".codex/skills/ssquant-backtest"
SCRIPT_ROOT = PROJECT_DIR / "scripts"
sys.path.insert(0, str(CODEX_SKILL_ROOT))
sys.path.insert(1, str(SCRIPT_ROOT))

from shared.runtime_paths import find_project_root, resolve_engine_path
from research.project_paths import DATASET_MANIFEST, RUNS_ROOT

RUNS_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(RUNS_ROOT)

ENGINE_PATH = resolve_engine_path(find_project_root(PROJECT_ROOT))
if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

from vwap_deviation_optimized_strategy import _higher_tf_bias


MANIFEST_PATH = DATASET_MANIFEST
OUTPUT_DIR = RUNS_ROOT / "index_mtf" / "research_new_opportunities"
SYMBOL = "IM888"
DEV_END = pd.Timestamp("2025-06-30 23:59:59")
VALIDATION_START = pd.Timestamp("2025-07-01")
HORIZONS = (1, 2, 4, 7)
COOLDOWN_BARS = 7
ENTRY_FEE_RATE = 0.000023
CLOSE_TODAY_FEE_RATE = 0.00023
TICK_SIZE = 0.2
SLIPPAGE_TICKS_PER_SIDE = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(manifest: dict, period: str) -> dict:
    source_id = f"{SYMBOL}_{period}_1"
    return next(item for item in manifest["sources"] if item["source_id"] == source_id)


def _load_frame(source: dict) -> pd.DataFrame:
    path = Path(source["file_path"])
    if not path.is_file():
        path = MANIFEST_PATH.parent / path.name
    actual_hash = _sha256(path)
    if actual_hash != source["bars_hash"]:
        raise RuntimeError(f"bars hash mismatch for {source['source_id']}")
    frame = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime").sort_index()
    if frame.index.has_duplicates:
        raise RuntimeError(f"duplicate timestamps in {source['source_id']}")
    return frame


def _session_rolling_vwap(frame: pd.DataFrame, bars: int) -> tuple[pd.Series, pd.Series]:
    vwap = pd.Series(np.nan, index=frame.index, dtype=float)
    dispersion = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, day in frame.groupby(frame.index.date, sort=False):
        volume = day["volume"].clip(lower=0).astype(float)
        price = day["close"].astype(float)
        sum_volume = volume.rolling(bars, min_periods=bars).sum()
        mean = (price * volume).rolling(bars, min_periods=bars).sum() / sum_volume
        second = (price.pow(2) * volume).rolling(bars, min_periods=bars).sum() / sum_volume
        vwap.loc[day.index] = mean
        dispersion.loc[day.index] = np.sqrt((second - mean.pow(2)).clip(lower=0))
    return vwap, dispersion


def _group_shift(frame: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
    return frame.groupby(frame.index.date, sort=False)[column].shift(periods)


def _group_rolling(frame: pd.DataFrame, column: str, bars: int, method: str) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, day in frame.groupby(frame.index.date, sort=False):
        rolling = day[column].shift(1).rolling(bars, min_periods=bars)
        values = getattr(rolling, method)()
        result.loc[day.index] = values
    return result


def _attach_higher_bias(base: pd.DataFrame, higher: pd.DataFrame) -> pd.DataFrame:
    closes: list[float] = []
    bias: list[int] = []
    for value in higher["close"].astype(float):
        closes.append(value)
        bias.append(
            _higher_tf_bias(
                closes,
                trend_bars=3,
                threshold=0.005,
                fast_bars=3,
                slow_bars=8,
                slope_bars=3,
                efficiency_threshold=0.35,
            )
        )
    higher_bias = pd.DataFrame(
        {
            "datetime": higher.index + pd.Timedelta(minutes=120),
            "higher_bias": bias,
        }
    )
    merged = pd.merge_asof(
        base.reset_index().sort_values("datetime"),
        higher_bias.sort_values("datetime"),
        on="datetime",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.set_index("datetime").sort_index()


def _build_features(base: pd.DataFrame, higher: pd.DataFrame) -> pd.DataFrame:
    frame = _attach_higher_bias(base.copy(), higher)
    for bars, label in ((6, "30"), (12, "60")):
        vwap, dispersion = _session_rolling_vwap(frame, bars)
        frame[f"vwap{label}"] = vwap
        frame[f"z{label}"] = (frame["close"] - vwap) / dispersion.replace(0, np.nan)

    denominator = frame["B"].astype(float) + frame["S"].astype(float)
    frame["flow_imbalance"] = (
        (frame["B"].astype(float) - frame["S"].astype(float)) / denominator.replace(0, np.nan)
    ).fillna(0.0)
    frame["body_strength"] = (
        (frame["close"] - frame["open"]) / (frame["high"] - frame["low"]).replace(0, np.nan)
    ).fillna(0.0)
    for column in ("close", "high", "low", "z30", "z60"):
        frame[f"prev_{column}"] = _group_shift(frame, column)
    frame["previous_6_high"] = _group_rolling(frame, "high", 6, "max")
    frame["previous_6_low"] = _group_rolling(frame, "low", 6, "min")
    frame["previous_12_volume_median"] = _group_rolling(frame, "volume", 12, "median")
    return frame


def _entry_window(index: pd.DatetimeIndex) -> pd.Series:
    minutes = index.hour * 60 + index.minute
    morning = (minutes >= 9 * 60) & (minutes <= 10 * 60 + 50)
    afternoon = (minutes >= 13 * 60) & (minutes <= 13 * 60 + 55)
    return pd.Series(morning | afternoon, index=index)


def _candidate_signals(frame: pd.DataFrame) -> dict[str, pd.Series]:
    bias = frame["higher_bias"].fillna(0).astype(int)
    valid = _entry_window(frame.index) & bias.ne(0)
    distinct = frame["z60"].abs().lt(2.25) & frame["prev_z60"].abs().lt(2.25)

    pullback = (
        valid
        & distinct
        & (bias * frame["prev_z30"] <= -0.8)
        & (bias * (frame["z30"] - frame["prev_z30"]) >= 0.35)
        & (bias * (frame["close"] - frame["prev_close"]) > 0)
        & (bias * frame["body_strength"] >= 0.15)
        & (bias * frame["flow_imbalance"] >= 0.05)
    )

    reversal_direction = -np.sign(frame["prev_z60"]).astype(float)
    failed_auction = (
        _entry_window(frame.index)
        & frame["prev_z60"].abs().between(1.5, 2.25, inclusive="left")
        & (frame["prev_z60"] * frame["z60"] > 0)
        & (frame["prev_z60"].abs() - frame["z60"].abs() >= 0.35)
        & (reversal_direction * frame["body_strength"] >= 0.15)
        & (reversal_direction * frame["flow_imbalance"] >= 0.05)
        & ((bias == 0) | (bias == reversal_direction))
    )

    breakout = (
        valid
        & distinct
        & np.where(
            bias > 0,
            frame["close"] > frame["previous_6_high"],
            frame["close"] < frame["previous_6_low"],
        )
        & (bias * frame["body_strength"] >= 0.35)
        & (bias * frame["flow_imbalance"] >= 0.08)
        & (frame["volume"] >= 1.15 * frame["previous_12_volume_median"])
    )
    return {
        "trend_pullback_reclaim": pd.Series(np.where(pullback, bias, 0), index=frame.index),
        "failed_auction_reversal": pd.Series(
            np.where(failed_auction, reversal_direction, 0), index=frame.index
        ),
        "trend_volume_breakout": pd.Series(np.where(breakout, bias, 0), index=frame.index),
    }


def _deduplicate(signal: pd.Series) -> pd.Series:
    kept = pd.Series(0, index=signal.index, dtype=int)
    last_position_by_day: dict[object, int] = {}
    for position, (timestamp, direction) in enumerate(signal.items()):
        if not direction:
            continue
        day = timestamp.date()
        if position - last_position_by_day.get(day, -COOLDOWN_BARS - 1) <= COOLDOWN_BARS:
            continue
        kept.iloc[position] = int(direction)
        last_position_by_day[day] = position
    return kept


def _event_rows(frame: pd.DataFrame, family: str, signal: pd.Series) -> list[dict]:
    rows: list[dict] = []
    core = frame["z60"].abs() >= 2.25
    for signal_position in np.flatnonzero(signal.to_numpy()):
        direction = int(signal.iloc[signal_position])
        entry_position = signal_position + 1
        if entry_position >= len(frame):
            continue
        signal_time = frame.index[signal_position]
        entry_time = frame.index[entry_position]
        if entry_time.date() != signal_time.date():
            continue
        entry_price = float(frame["open"].iloc[entry_position])
        for horizon in HORIZONS:
            exit_position = signal_position + horizon
            if exit_position >= len(frame):
                continue
            exit_time = frame.index[exit_position]
            if exit_time.date() != signal_time.date():
                continue
            expected_elapsed = pd.Timedelta(minutes=5 * (horizon - 1))
            if pd.Timestamp(exit_time) - pd.Timestamp(entry_time) > expected_elapsed:
                continue
            exit_price = float(frame["close"].iloc[exit_position])
            gross_points = direction * (exit_price - entry_price)
            cost_points = (
                entry_price * ENTRY_FEE_RATE
                + exit_price * CLOSE_TODAY_FEE_RATE
                + 2 * SLIPPAGE_TICKS_PER_SIDE * TICK_SIZE
            )
            rows.append(
                {
                    "family": family,
                    "signal_time": signal_time,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "direction": direction,
                    "horizon_bars": horizon,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_points": gross_points,
                    "estimated_cost_points": cost_points,
                    "net_points": gross_points - cost_points,
                    "overlaps_core_extreme": bool(core.iloc[signal_position]),
                    "higher_bias": int(frame["higher_bias"].iloc[signal_position]),
                    "z30": float(frame["z30"].iloc[signal_position]),
                    "z60": float(frame["z60"].iloc[signal_position]),
                    "flow_imbalance": float(frame["flow_imbalance"].iloc[signal_position]),
                }
            )
    return rows


def _summarize(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["sample"] = np.where(events["signal_time"] <= DEV_END, "development", "validation")
    events = events[events["signal_time"] >= pd.Timestamp("2024-01-02")]
    grouped = events.groupby(["family", "sample", "horizon_bars"], sort=True)
    return grouped.agg(
        events=("net_points", "size"),
        mean_gross_points=("gross_points", "mean"),
        mean_net_points=("net_points", "mean"),
        median_net_points=("net_points", "median"),
        net_win_rate=("net_points", lambda values: float((values > 0).mean())),
        p10_net_points=("net_points", lambda values: float(values.quantile(0.10))),
        core_overlap_rate=("overlaps_core_extreme", "mean"),
    ).reset_index()


def _robustness_summary(events: pd.DataFrame) -> pd.DataFrame:
    selected = events.loc[events["horizon_bars"] == 7].copy()
    selected["year"] = selected["signal_time"].dt.year
    selected["side"] = np.where(selected["direction"] > 0, "long", "short")
    return selected.groupby(["family", "year", "side"], sort=True).agg(
        events=("net_points", "size"),
        mean_net_points=("net_points", "mean"),
        median_net_points=("net_points", "median"),
        net_win_rate=("net_points", lambda values: float((values > 0).mean())),
    ).reset_index()


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    base_source = _source(manifest, "5m")
    higher_source = _source(manifest, "120m")
    frame = _build_features(_load_frame(base_source), _load_frame(higher_source))

    event_rows: list[dict] = []
    signal_columns: dict[str, pd.Series] = {}
    raw_counts: dict[str, int] = {}
    kept_counts: dict[str, int] = {}
    for family, raw_signal in _candidate_signals(frame).items():
        raw_counts[family] = int(raw_signal.ne(0).sum())
        signal = _deduplicate(raw_signal)
        signal_columns[family] = signal
        kept_counts[family] = int(signal.ne(0).sum())
        event_rows.extend(_event_rows(frame, family, signal))

    events = pd.DataFrame(event_rows)
    if events.empty:
        raise RuntimeError("no candidate events were generated")
    summary = _summarize(events)
    robustness = _robustness_summary(events)
    signals = pd.DataFrame(signal_columns)
    active = signals.ne(0)
    overlap = active.astype(int).T.dot(active.astype(int))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUTPUT_DIR / "candidate_events.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "candidate_summary.csv", index=False)
    robustness.to_csv(OUTPUT_DIR / "candidate_robustness.csv", index=False)
    signals.loc[active.any(axis=1)].to_csv(OUTPUT_DIR / "candidate_signals.csv", index_label="datetime")
    overlap.to_csv(OUTPUT_DIR / "candidate_overlap.csv", index_label="family")

    research_manifest = {
        "schema_version": 1,
        "run_type": "research-event-study",
        "formal_backtest": False,
        "symbol": SYMBOL,
        "development_end": str(DEV_END),
        "validation_start": str(VALIDATION_START),
        "source_ids": [base_source["source_id"], higher_source["source_id"]],
        "source_hashes": {
            base_source["source_id"]: base_source["bars_hash"],
            higher_source["source_id"]: higher_source["bars_hash"],
        },
        "adjustment_mode": manifest["adjustment_mode"],
        "signal_timing": "features through signal bar close; entry at next bar open",
        "cost_model": {
            "entry_fee_rate": ENTRY_FEE_RATE,
            "close_today_fee_rate": CLOSE_TODAY_FEE_RATE,
            "slippage_ticks_per_side": SLIPPAGE_TICKS_PER_SIDE,
            "tick_size": TICK_SIZE,
        },
        "raw_signal_counts": raw_counts,
        "deduplicated_signal_counts": kept_counts,
        "script_hash": _sha256(Path(__file__)),
    }
    (OUTPUT_DIR / "research_manifest.json").write_text(
        json.dumps(research_manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(json.dumps({"raw": raw_counts, "deduplicated": kept_counts}, ensure_ascii=True))


if __name__ == "__main__":
    main()
