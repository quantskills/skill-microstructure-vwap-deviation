# Data Contract

## Mode Boundary

Standalone runs may use the official adapter or synthetic data for pipeline tests. Formal
conclusions require the official SSQuant transformed and frozen dataset pipeline. Raw or synthetic
bars must be explicitly marked and cannot be silently substituted into a formal comparison.

## Official routes

| Asset | Symbol example | Required API |
|---|---|---|
| A-share | `000001.SZ` | `panda_data.get_stock_min` |
| Futures contract | `RB2405.SHF` | `panda_data.get_future_min` |
| Futures dominant | `RB_DOMINANT.SHF` | `panda_data.get_future_min` |

The adapter accepts `1m`, `5m`, `15m`, and `60m`. The returned frame has a sorted
`DatetimeIndex` and numeric `open`, `high`, `low`, `close`, `volume`, `amount`, and
`open_interest` columns. Optional `session`, `symbol`, and `dominant_id` columns are preserved.

## Identity

`freeze_dataset()` writes a canonical `bars.csv`, `dataset_manifest.json`, and SHA-256 hash. The
hash is computed from sorted canonical CSV bytes and must be recorded in every run manifest.
Adjustment mode is explicit: `official`, `raw`, or `synthetic`. `raw` and `synthetic` runs are not
formal conclusions when the project has an official transformed pipeline.

## Failure behavior

Missing `panda_data`, unsupported symbols, invalid frequency, malformed timestamps, and invalid
prices fail clearly. There is no silent fallback to another data source.
