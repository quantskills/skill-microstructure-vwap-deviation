---
name: skill-microstructure-vwap-deviation
description: Minute-bar VWAP deviation strategy research and auditable SSQuant futures validation. Use for rolling VWAP deviation signals, confirmed mean reversion, trend guards, next-bar execution, execution-cost modeling, frozen-data backtests, and trade-result review.
license: GPL-3.0-only
metadata:
  organization: QuantSkills
  project_type: skill
  collection: futures-research
  maintainer: local
---

# Microstructure VWAP Deviation

## Purpose

Turn minute OHLCV data into a causal VWAP-deviation strategy signal and a governed research
result. The skill supports two execution layers:

- Standalone research harness for fast signal and cost-model validation.
- SSQuant formal runner for frozen-data futures evaluation with next-bar execution and audited
  artifacts.

This skill does not publish conclusions from synthetic data and does not replace SSQuant's
`AccountBridge` with a hand-written formal PnL engine.

## Boundaries

- No future-bar access. A signal on bar `t` can only fill at bar `t+1` open.
- VWAP state is isolated by symbol and reset by explicit session policy.
- Formal comparisons require identical frozen bars, adjustment mode, and dataset hashes.
- Standalone defaults and formal futures contract costs are separate contracts; never mix them
  silently.
- Generated bars, trades, reports, logs, caches, and plots stay outside this package.

## Input Contract

Read [data_guide.md](references/data_guide.md) before adding or changing a data source. The
standalone loader accepts 1m/5m/15m/60m OHLCV data. Futures formal runs use the official SSQuant
pipeline, an explicit session policy, and a schema-v2 frozen dataset manifest.

## Output Contract

Read [output_contract.md](references/output_contract.md) before running a formal evaluation.
Standalone and formal runs must emit the required artifacts under the external run directory,
not under this skill directory.

## Workflow

1. Read [parameter_policy.md](references/parameter_policy.md) and present fee, slippage, cash,
   quantity, multiplier, margin, and session assumptions before a run.
2. Read [strategy_contract.md](references/strategy_contract.md) and select standalone or formal
   mode explicitly.
3. Run `python scripts/test.py` before changing strategy behavior.
4. For research, run the standalone harness and label synthetic data as debug-only.
5. For formal futures work, freeze and hash the official transformed bars first, then run
   `scripts/run_index_mtf_formal.py` or a governed matrix runner in `scripts/research/`.
6. Run [review_checklist.md](references/review_checklist.md) before comparing or publishing a
   result.

## Scripts

| Script | Responsibility |
|---|---|
| `scripts/run_backtest.py` | Standalone VWAP research harness |
| `scripts/vwap_strategy.py` | Baseline rolling VWAP signal logic |
| `scripts/vwap_deviation_optimized_strategy.py` | SSQuant-compatible optimized strategy logic |
| `scripts/run_vwap_deviation_optimized_backtest.py` | Optimized standalone SSQuant harness |
| `scripts/run_index_mtf_formal.py` | Frozen-data formal MTF runner |
| `scripts/research/prepare_index_mtf_dataset.py` | Official frozen dataset preparation |
| `scripts/research/run_*_matrix.py` | Governed parameter and robustness matrices |
| `scripts/research/discover_*_opportunities.py` | Causal research and opportunity discovery |
| `scripts/test.py` | Package regression test entrypoint |

## Run

```powershell
python scripts/test.py
python scripts/run_backtest.py --synthetic `
  --output-dir ..\skill-microstructure-vwap-deviation-runs\standalone\synthetic
```

Formal runs require an external frozen manifest and output directory:

```powershell
$env:SSQUANT_DATA_MODE = "frozen"
$env:SSQUANT_PROJECT_ROOT = "C:\path\to\ssquant-project"
$env:SSQUANT_ENGINE_PATH = "D:\path\to\ssquant\ssquant"
$env:SSQUANT_DATASET_MANIFEST = "C:\path\to\dataset_manifest.json"
$env:SSQUANT_OUTPUT_DIR = "C:\path\to\skill-microstructure-vwap-deviation-runs\formal\IM888\5m_120m"
python scripts/run_index_mtf_formal.py
```

`SSQUANT_PROJECT_ROOT` and `SSQUANT_ENGINE_PATH` are external runtime dependencies. They are not
part of this skill package and must resolve to a real SSQuant project and engine before a formal
run.

The default external root is the sibling directory
`skill-microstructure-vwap-deviation-runs`. Override it with
`VWAP_RESEARCH_OUTPUT_ROOT`. Set `VWAP_RESEARCH_DATASET_MANIFEST` for research matrix scripts.

## Required Checks

```powershell
python scripts/test.py
```

Before a formal claim, all rows in [review_checklist.md](references/review_checklist.md) must
pass. An unknown exchange session calendar, missing data hash, future-bar reference, or missing
artifact blocks the run.

## References

- [method_guide.md](references/method_guide.md): end-to-end research and formal workflow
- [strategy_contract.md](references/strategy_contract.md): signal and regime rules
- [data_guide.md](references/data_guide.md): source, adjustment, and freeze contract
- [execution_contract.md](references/execution_contract.md): fills, fees, slippage, and margin
- [parameter_policy.md](references/parameter_policy.md): defaults and override policy
- [output_contract.md](references/output_contract.md): required run/compare/stitch artifacts
- [review_checklist.md](references/review_checklist.md): audit and delivery gate
- [package-boundary.md](references/package-boundary.md): source-only package rule
