# Parameter Prompt Policy

Before any run, present this table to the user and ask whether to keep the defaults or provide
overrides. Do not hide execution costs inside the strategy description.

| Parameter | Default | Meaning | CLI override |
|---|---:|---|---|
| Initial cash | `300000.0` | Starting cash in the standalone research harness | `--initial-cash 500000` |
| Quantity | `1` | Units opened for each signal | `--quantity 2` |
| Fee rate | `0.0003` | 0.03% of entry and exit notional, each side | `--fee-rate 0.0005` |
| Slippage | `1.0 bps` | 0.01% adverse price adjustment per fill | `--slippage-bps 2.0` |

Use the following prompt contract:

```text
当前回测默认参数：初始资金 300000，数量 1，手续费率 0.0003（双边各收取），滑点 1.0 bps。
是否使用默认值？如需调整，请提供 initial_cash、quantity、fee_rate、slippage_bps。
```

When the user provides overrides, validate them before running:

- `initial_cash > 0`
- `quantity` is a positive integer
- `fee_rate >= 0`
- `slippage_bps >= 0`

Example:

```powershell
python scripts/run_backtest.py `
  --symbol RB2405.SHF `
  --start-date 20260101 `
  --end-date 20260131 `
  --initial-cash 500000 `
  --quantity 2 `
  --fee-rate 0.0005 `
  --slippage-bps 2.0 `
  --output-dir ..\\skill-microstructure-vwap-deviation-runs\\RB2405.SHF
```

These are generic standalone-harness assumptions. For formal futures evaluation, replace them
with the project's official contract multiplier, commission schedule, tick size, margin, and
slippage model. Do not claim that the generic defaults represent exchange costs.
