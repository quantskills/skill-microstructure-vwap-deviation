# Package Boundary

`skill-microstructure-vwap-deviation` is the reusable skill package. The package is intentionally
source-only so it can be installed, audited, and reused without carrying local research state.

## Keep In The Skill

- `SKILL.md` and `agents/openai.yaml`
- reusable strategy and data-loader code under `scripts/`
- reusable research runners under `scripts/research/`
- unit tests under `tests/`
- stable contracts and parameter rules under `references/`

## Keep Outside The Skill

- frozen bars and dataset manifests generated for a run
- `equity_curve.csv`, `equity_curve.png`, `trades.csv`, and reports
- engine HTML/TXT reports
- logs, caches, Python bytecode, and exploratory exports

The default external workspace is the sibling directory
`skill-microstructure-vwap-deviation-runs`. Override it with
`VWAP_RESEARCH_OUTPUT_ROOT`. A formal run must receive a frozen manifest through
`VWAP_RESEARCH_DATASET_MANIFEST` or `SSQUANT_DATASET_MANIFEST`.
