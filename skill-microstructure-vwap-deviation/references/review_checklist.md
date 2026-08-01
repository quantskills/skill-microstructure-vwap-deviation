# Audit Checklist

Run this checklist after changing the strategy, data adapter, or runner.

| Check | Required result |
|---|---|
| Future data | No `shift(-N)`, future index, or post-signal bar access |
| Rolling state | VWAP and dispersion reset per session and never mix symbols |
| Execution | Signal at bar t fills at bar t+1 open; last-bar entries are not filled |
| Trend guard | Trend/range state is computed from current and historical prices only |
| Tail guard | New entries are blocked inside the configured session tail |
| Dataset | Bars are frozen before calculation and SHA-256 is recorded |
| Adjustment | Raw or synthetic input is explicitly marked and not compared as official data |
| Outputs | All five governed standalone artifacts exist and are non-ambiguous |
| SSQuant | Formal runs use the official runner and AccountBridge, not standalone P&L |

Any failed row blocks a formal backtest or comparison. An unknown session calendar is a warning
until the user supplies the exchange-specific policy; it is not permission to guess.

## Package Delivery

- `SKILL.md` and `agents/` metadata are present.
- Every internal link points to an existing file.
- Strategy, data, execution, and output contracts agree on defaults and timing.
- No generated result, cache, log, bytecode, or frozen dataset is inside the skill package.
- `python scripts/test.py` passes with its pytest cache outside the package.
