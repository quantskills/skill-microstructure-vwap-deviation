# -*- coding: utf-8 -*-
"""Research causal IM opportunities from synchronized IM/IF/IC bars."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import discover_new_opportunities as common


OUTPUT_DIR = common.RUNS_ROOT / "index_mtf" / "research_cross_index_opportunities"
SYMBOLS = ("IM888", "IF888", "IC888")
RETURN_WINDOW = 6
BETA_WINDOW = 480
STANDARDIZATION_WINDOW = 240


def _source(manifest: dict, symbol: str, period: str) -> dict:
    source_id = f"{symbol}_{period}_1"
    return next(item for item in manifest["sources"] if item["source_id"] == source_id)


def _session_return(series: pd.Series, periods: int) -> pd.Series:
    return series.groupby(series.index.date, sort=False).pct_change(periods, fill_method=None)


def _past_zscore(series: pd.Series, window: int) -> pd.Series:
    past = series.shift(1).rolling(window, min_periods=window // 2)
    return (series - past.mean()) / past.std(ddof=0).replace(0, np.nan)


def _previous_range(series: pd.Series, bars: int, method: str) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    frame = series.to_frame("value")
    for _, day in frame.groupby(frame.index.date, sort=False):
        rolling = day["value"].shift(1).rolling(bars, min_periods=bars)
        result.loc[day.index] = getattr(rolling, method)()
    return result


def _fixed_session_range(frame: pd.DataFrame, mask: pd.Series) -> tuple[pd.Series, pd.Series]:
    highs = pd.Series(np.nan, index=frame.index, dtype=float)
    lows = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, day in frame.groupby(frame.index.date, sort=False):
        selected = day.loc[mask.loc[day.index]]
        if selected.empty:
            continue
        highs.loc[day.index] = float(selected["high"].max())
        lows.loc[day.index] = float(selected["low"].min())
    return highs, lows


def _build_features(manifest: dict) -> tuple[pd.DataFrame, list[dict]]:
    sources = [_source(manifest, symbol, "5m") for symbol in SYMBOLS]
    higher_source = _source(manifest, "IM888", "120m")
    frames = {source["symbol"]: common._load_frame(source) for source in sources}
    higher = common._load_frame(higher_source)

    im_features = common._build_features(frames["IM888"], higher)
    frame = im_features.copy()
    for symbol in ("IF888", "IC888"):
        peer = frames[symbol][["open", "high", "low", "close", "volume"]].add_prefix(
            f"{symbol[:2]}_"
        )
        frame = frame.join(peer, how="inner")

    for symbol, close_column in (
        ("IM", "close"),
        ("IF", "IF_close"),
        ("IC", "IC_close"),
    ):
        frame[f"{symbol}_ret5"] = _session_return(frame[close_column], 1)
        frame[f"{symbol}_ret30"] = _session_return(frame[close_column], RETURN_WINDOW)

    frame["peer_ret5"] = (frame["IF_ret5"] + frame["IC_ret5"]) / 2.0
    frame["peer_ret30"] = (frame["IF_ret30"] + frame["IC_ret30"]) / 2.0
    lagged_im = frame["IM_ret5"].shift(1)
    lagged_peer = frame["peer_ret5"].shift(1)
    rolling_covariance = lagged_im.rolling(BETA_WINDOW, min_periods=BETA_WINDOW // 2).cov(
        lagged_peer
    )
    rolling_variance = lagged_peer.rolling(
        BETA_WINDOW, min_periods=BETA_WINDOW // 2
    ).var(ddof=0)
    frame["rolling_beta"] = (rolling_covariance / rolling_variance.replace(0, np.nan)).clip(
        0.25, 2.5
    )
    frame["residual_ret30"] = frame["IM_ret30"] - frame["rolling_beta"] * frame["peer_ret30"]
    frame["residual_z"] = _past_zscore(frame["residual_ret30"], STANDARDIZATION_WINDOW)
    frame["previous_residual_z"] = frame.groupby(frame.index.date, sort=False)[
        "residual_z"
    ].shift(1)
    frame["peer_momentum_z"] = _past_zscore(frame["peer_ret30"], STANDARDIZATION_WINDOW)

    for peer in ("IF", "IC"):
        frame[f"{peer}_previous_high"] = _previous_range(frame[f"{peer}_high"], 6, "max")
        frame[f"{peer}_previous_low"] = _previous_range(frame[f"{peer}_low"], 6, "min")
    frame["IM_previous_high"] = _previous_range(frame["high"], 6, "max")
    frame["IM_previous_low"] = _previous_range(frame["low"], 6, "min")
    minutes = frame.index.hour * 60 + frame.index.minute
    opening_mask = pd.Series((minutes >= 9 * 60 + 30) & (minutes <= 9 * 60 + 55), index=frame.index)
    morning_mask = pd.Series((minutes >= 9 * 60 + 30) & (minutes <= 11 * 60 + 20), index=frame.index)
    frame["opening_high"], frame["opening_low"] = _fixed_session_range(frame, opening_mask)
    frame["morning_high"], frame["morning_low"] = _fixed_session_range(frame, morning_mask)
    frame["previous_close"] = frame.groupby(frame.index.date, sort=False)["close"].shift(1)
    frame["previous_high"] = frame.groupby(frame.index.date, sort=False)["high"].shift(1)
    frame["previous_low"] = frame.groupby(frame.index.date, sort=False)["low"].shift(1)
    return frame, sources + [higher_source]


def _candidate_signals(frame: pd.DataFrame) -> dict[str, pd.Series]:
    bias = frame["higher_bias"].fillna(0).astype(int)
    window = common._entry_window(frame.index)
    distinct = frame["z60"].abs().lt(2.25) & frame["prev_z60"].abs().lt(2.25)

    peer_direction = np.sign(frame["peer_ret30"]).astype(float)
    peers_agree = np.sign(frame["IF_ret30"]) == np.sign(frame["IC_ret30"])
    catchup = (
        window
        & distinct
        & bias.ne(0)
        & (peer_direction == bias)
        & peers_agree
        & (bias * frame["peer_momentum_z"] >= 1.0)
        & (bias * frame["residual_z"] <= -1.25)
        & (bias * (frame["residual_z"] - frame["previous_residual_z"]) >= 0.25)
        & (bias * frame["body_strength"] >= 0.15)
        & (bias * frame["flow_imbalance"] >= 0.05)
    )

    fade_direction = -np.sign(frame["previous_residual_z"]).astype(float)
    residual_fade = (
        window
        & distinct
        & frame["previous_residual_z"].abs().ge(2.0)
        & (frame["previous_residual_z"] * frame["residual_z"] > 0)
        & (frame["previous_residual_z"].abs() - frame["residual_z"].abs() >= 0.35)
        & ((bias == 0) | (bias == fade_direction))
        & (fade_direction * frame["body_strength"] >= 0.15)
        & (fade_direction * frame["flow_imbalance"] >= 0.05)
    )

    peer_up_break = (frame["IF_close"] > frame["IF_previous_high"]) & (
        frame["IC_close"] > frame["IC_previous_high"]
    )
    peer_down_break = (frame["IF_close"] < frame["IF_previous_low"]) & (
        frame["IC_close"] < frame["IC_previous_low"]
    )
    breakout_direction = pd.Series(
        np.select([peer_up_break, peer_down_break], [1, -1], default=0),
        index=frame.index,
        dtype=int,
    )
    im_has_not_broken = np.where(
        breakout_direction > 0,
        frame["close"] <= frame["IM_previous_high"],
        frame["close"] >= frame["IM_previous_low"],
    )
    peer_lead = (
        window
        & distinct
        & breakout_direction.ne(0)
        & (bias == breakout_direction)
        & im_has_not_broken
        & (breakout_direction * frame["body_strength"] >= 0.10)
        & (breakout_direction * frame["flow_imbalance"] >= 0.03)
    )

    minutes = frame.index.hour * 60 + frame.index.minute
    morning_trade = (minutes >= 10 * 60) & (minutes <= 10 * 60 + 50)
    opening_break_direction = pd.Series(
        np.select(
            [
                (frame["close"] > frame["opening_high"])
                & (frame["previous_close"] <= frame["opening_high"]),
                (frame["close"] < frame["opening_low"])
                & (frame["previous_close"] >= frame["opening_low"]),
            ],
            [1, -1],
            default=0,
        ),
        index=frame.index,
        dtype=int,
    )
    opening_breakout = (
        window
        & morning_trade
        & distinct
        & opening_break_direction.ne(0)
        & (bias == opening_break_direction)
        & (np.sign(frame["IF_ret30"]) == opening_break_direction)
        & (np.sign(frame["IC_ret30"]) == opening_break_direction)
        & (opening_break_direction * frame["body_strength"] >= 0.20)
        & (opening_break_direction * frame["flow_imbalance"] >= 0.05)
    )

    opening_failure_direction = pd.Series(
        np.select(
            [
                (frame["previous_high"] > frame["opening_high"])
                & (frame["close"] < frame["opening_high"]),
                (frame["previous_low"] < frame["opening_low"])
                & (frame["close"] > frame["opening_low"]),
            ],
            [-1, 1],
            default=0,
        ),
        index=frame.index,
        dtype=int,
    )
    opening_failure = (
        window
        & morning_trade
        & distinct
        & opening_failure_direction.ne(0)
        & ((bias == 0) | (bias == opening_failure_direction))
        & (opening_failure_direction * frame["body_strength"] >= 0.20)
        & (opening_failure_direction * frame["flow_imbalance"] >= 0.05)
    )

    afternoon = (minutes >= 13 * 60) & (minutes <= 13 * 60 + 55)
    afternoon_break_direction = pd.Series(
        np.select(
            [
                (frame["close"] > frame["morning_high"])
                & (frame["previous_close"] <= frame["morning_high"]),
                (frame["close"] < frame["morning_low"])
                & (frame["previous_close"] >= frame["morning_low"]),
            ],
            [1, -1],
            default=0,
        ),
        index=frame.index,
        dtype=int,
    )
    afternoon_breakout = (
        window
        & afternoon
        & distinct
        & afternoon_break_direction.ne(0)
        & (bias == afternoon_break_direction)
        & (np.sign(frame["IF_ret30"]) == afternoon_break_direction)
        & (np.sign(frame["IC_ret30"]) == afternoon_break_direction)
        & (afternoon_break_direction * frame["body_strength"] >= 0.20)
        & (afternoon_break_direction * frame["flow_imbalance"] >= 0.05)
    )
    return {
        "cross_index_catchup": pd.Series(np.where(catchup, bias, 0), index=frame.index),
        "residual_overreaction_fade": pd.Series(
            np.where(residual_fade, fade_direction, 0), index=frame.index
        ),
        "peer_breakout_lead": pd.Series(
            np.where(peer_lead, breakout_direction, 0), index=frame.index
        ),
        "opening_range_breakout": pd.Series(
            np.where(opening_breakout, opening_break_direction, 0), index=frame.index
        ),
        "opening_breakout_failure": pd.Series(
            np.where(opening_failure, opening_failure_direction, 0), index=frame.index
        ),
        "opening_breakout_failure_short": pd.Series(
            np.where(opening_failure & opening_failure_direction.eq(-1), -1, 0),
            index=frame.index,
        ),
        "afternoon_morning_range_breakout": pd.Series(
            np.where(afternoon_breakout, afternoon_break_direction, 0), index=frame.index
        ),
    }


def main() -> None:
    manifest = json.loads(common.MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    frame, sources = _build_features(manifest)
    rows: list[dict] = []
    signal_columns: dict[str, pd.Series] = {}
    raw_counts: dict[str, int] = {}
    kept_counts: dict[str, int] = {}
    for family, raw_signal in _candidate_signals(frame).items():
        raw_counts[family] = int(raw_signal.ne(0).sum())
        signal = common._deduplicate(raw_signal)
        signal_columns[family] = signal
        kept_counts[family] = int(signal.ne(0).sum())
        rows.extend(common._event_rows(frame, family, signal))

    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError("no cross-index candidate events were generated")
    summary = common._summarize(events)
    robustness = common._robustness_summary(events)
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
        "run_type": "research-cross-index-event-study",
        "formal_backtest": False,
        "symbols": list(SYMBOLS),
        "source_hashes": {source["source_id"]: source["bars_hash"] for source in sources},
        "adjustment_mode": manifest["adjustment_mode"],
        "higher_bar_availability": "timestamp + 120 minutes <= signal timestamp",
        "signal_timing": "all instruments through signal bar close; entry at next IM bar open",
        "raw_signal_counts": raw_counts,
        "deduplicated_signal_counts": kept_counts,
        "script_hash": common._sha256(Path(__file__)),
    }
    (OUTPUT_DIR / "research_manifest.json").write_text(
        json.dumps(research_manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(json.dumps({"raw": raw_counts, "deduplicated": kept_counts}, ensure_ascii=True))


if __name__ == "__main__":
    main()
