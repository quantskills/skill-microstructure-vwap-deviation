# Specification: Minute VWAP Deviation

## Mode Profiles

The package exposes two profiles. They share causal execution and the same data/audit boundary,
but their defaults are not interchangeable.

| Profile | Entry | Exit | VWAP state | Use |
|---|---:|---:|---|---|
| Standalone baseline | `2.0 std` | `0.5 std` | rolling 30 or 60 minutes | signal and cost research |
| Formal optimized | configured per run, commonly `2.5 std` | commonly `0.25 std` | 60-minute core plus 30-minute local confirmation | SSQuant futures validation |

Formal optimized parameters must be read from the run manifest and strategy constants. Do not
compare a standalone baseline result with a formal optimized result as if they were the same
strategy.

## Objective

Provide a reusable, auditable minute-bar mean-reversion strategy for stocks and futures.
The strategy measures volume-weighted price deviation from a rolling intraday VWAP and
emits next-bar execution events. It must not use future bars.

## Strategy Contract

| Item | Default | Rule |
|---|---:|---|
| Input | 1m/5m/15m/60m OHLCV | One symbol per calculation; multi-symbol data is grouped and isolated |
| VWAP window | 30 minutes | Configurable to 60 minutes; converted to bars from input frequency |
| Entry | 2.0 std | Above VWAP: short; below VWAP: long |
| Exit | 0.5 std | Close when absolute deviation is at or inside VWAP +/- 0.5 std |
| Trend guard | enabled | When enabled, only enter during a range regime |
| Tail guard | 30 minutes | No new entry inside the configured session-end tail |
| Execution | next bar open | Signal at bar t cannot fill on bar t |
| Session | reset by session id/date | Never carry rolling state across sessions |

## Acceptance Criteria

- Rolling VWAP and dispersion use only the current and prior bars.
- Each symbol has independent rolling state.
- Entry, exit, range filter, and tail guard are deterministic and unit tested.
- Real data uses the official `panda_data.get_stock_min` or `get_future_min` adapter.
- Formal runs freeze input bars, record a reproducible hash, and emit the governed artifacts.
- Every run presents execution-cost defaults before execution and records any overrides in the manifest.
- Synthetic data is permitted for pipeline tests only and is marked debug-only.

## Boundaries

- Always: sort and validate bars, preserve the source metadata, freeze data before formal runs,
  and record configuration plus data hash in the manifest.
- Ask first: changing the official data source, changing session semantics, or changing the
  meaning of the entry/exit thresholds.
- Never: use future bars, mix symbols' VWAP state, silently substitute raw bars for an official
  transformed dataset, or publish synthetic results as research conclusions.

## Open Questions

- Exact session calendars for each futures exchange are not present in the workspace. The runner
  therefore requires an explicit session-end time for tail protection and documents that the
  default `15:00` is appropriate for regular A-share sessions only.
