# Execution Contract

## Signal Timing

At bar `t`, the strategy may read that bar's OHLCV and all earlier bars. An order created from
that signal is eligible at the open of bar `t+1`. The signal bar cannot fill its own close.

```text
signal_time = t
execution_time = next_bar(t)
execution_price = next_bar.open adjusted for adverse slippage
```

Unfilled orders at the end of the available data are not counted as completed trades.

## Standalone Cost Model

For a fill with price `P`, quantity `Q`, contract multiplier `M`, fee rate `f`, and slippage
`s` in basis points:

```text
notional = P * Q * M
commission = notional * f
slippage_price = P * s / 10,000
```

Slippage is adverse to the trade direction on both entry and exit. Commission is charged on each
side. The generic standalone defaults are documented in [parameter_policy.md](parameter_policy.md)
and must not be described as exchange-specific costs.

## IM Formal Contract

When the user requests the real IM contract, record these values in the run manifest:

| Parameter | Value |
|---|---:|
| Contract multiplier | `200` |
| Price tick | `0.2` |
| Margin rate | `12%` |
| Open / overnight close fee | `0.000023` of notional |
| Close-today fee | `0.00023` of notional |
| Slippage | `1` tick per side unless overridden |

Same-day close fees must be distinguished from overnight close fees. If the engine does not
apply that distinction natively, apply and document the governed close-today adjustment before
publishing results.

## Position Rules

- One symbol's position state must not be mixed with another symbol's VWAP state.
- Formal position sizing must use the framework's account and contract metadata.
- Standalone PnL calculations are for research validation only and cannot replace formal SSQuant
  accounting.
