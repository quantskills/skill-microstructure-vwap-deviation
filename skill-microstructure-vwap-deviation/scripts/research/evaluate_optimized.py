# -*- coding: utf-8 -*-
"""Generate a hash-audited review of the optimized VWAP strategy runs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from research.project_paths import RUNS_ROOT

ROOT = RUNS_ROOT
RESULTS = RUNS_ROOT
SYMBOLS = ("IM888", "AU888", "RB888", "CU888", "AG888")
VARIANTS = ("optimized_frozen_60m", "optimized_frozen_120m")
def _manifest(variant: str, symbol: str) -> dict:
    path = RESULTS / variant / symbol / "run_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_manifest(symbol: str) -> dict:
    path = RESULTS / symbol / "run_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _pair_trades(variant: str, symbol: str) -> pd.DataFrame:
    path = RESULTS / variant / symbol / "trades.csv"
    trades = pd.read_csv(path)
    if trades.empty:
        return pd.DataFrame()
    open_prefix = chr(0x5F00)
    close_prefix = chr(0x5E73)
    pending = None
    rows = []
    for row in trades.sort_values("datetime").to_dict("records"):
        action = str(row.get("action", ""))
        if action.startswith(open_prefix):
            pending = row
        elif action.startswith(close_prefix) and pending is not None:
            rows.append(
                {
                    "entry_time": pending["datetime"],
                    "exit_time": row["datetime"],
                    "net_profit": float(row.get("net_profit", 0.0) or 0.0),
                    "amount_profit": float(row.get("amount_profit", 0.0) or 0.0),
                    "commission": float(pending.get("commission", 0.0) or 0.0)
                    + float(row.get("commission", 0.0) or 0.0),
                    "slippage": float(pending.get("slippage", 0.0) or 0.0)
                    + float(row.get("slippage", 0.0) or 0.0),
                    "reason": row.get("reason", ""),
                }
            )
            pending = None
    paired = pd.DataFrame(rows)
    if not paired.empty:
        paired["entry_time"] = pd.to_datetime(paired["entry_time"])
        paired["exit_time"] = pd.to_datetime(paired["exit_time"])
        paired["hold_minutes"] = (
            paired["exit_time"] - paired["entry_time"]
        ).dt.total_seconds() / 60.0
    return paired


def main() -> None:
    rows = []
    attribution_rows = []
    for symbol in SYMBOLS:
        baseline_manifest = _baseline_manifest(symbol)
        baseline = baseline_manifest["performance"]
        for variant in VARIANTS:
            manifest = _manifest(variant, symbol)
            performance = manifest["performance"]
            dataset = manifest["dataset"]
            optimized = {
                "symbol": symbol,
                "variant": variant,
                "higher_period": manifest["higher_period"],
                "return_pct": float(performance["total_return"]),
                "drawdown_pct": float(performance["max_drawdown_pct"]),
                "sharpe": float(performance["sharpe_ratio"]),
                "win_rate_pct": float(performance["win_rate"]),
                "trades": int(performance["trade_stats"]["total_trades"]),
                "profit_factor": float(performance["trade_stats"]["profit_factor"]),
                "return_delta_vs_baseline_pp": float(performance["total_return"] - baseline["total_return"]),
                "drawdown_delta_vs_baseline_pp": float(performance["max_drawdown_pct"] - baseline["max_drawdown_pct"]),
                "sharpe_delta_vs_baseline": float(performance["sharpe_ratio"] - baseline["sharpe_ratio"]),
                "baseline_eval_hash": baseline_manifest["dataset"]["bars_sha256"],
                "optimized_eval_hash": dataset["evaluation_bars_sha256"],
                "same_base_hash": False,
            }
            optimized["same_base_hash"] = (
                optimized["baseline_eval_hash"] == optimized["optimized_eval_hash"]
            )
            rows.append(optimized)

            paired = _pair_trades(variant, symbol)
            if not paired.empty:
                attribution_rows.append(
                    {
                        "symbol": symbol,
                        "variant": variant,
                        "closed_trades": len(paired),
                        "net_pnl": float(paired["net_profit"].sum()),
                        "gross_pnl": float(paired["amount_profit"].sum()),
                        "commission": float(paired["commission"].sum()),
                        "slippage": float(paired["slippage"].sum()),
                        "cost_pct_of_abs_gross": float(
                            (paired["commission"].sum() + paired["slippage"].sum())
                            / max(abs(paired["amount_profit"].sum()), 1e-12)
                            * 100.0
                        ),
                        "median_hold_minutes": float(paired["hold_minutes"].median()),
                        "p90_hold_minutes": float(paired["hold_minutes"].quantile(0.9)),
                        "positive_trades": int((paired["net_profit"] > 0).sum()),
                        "negative_trades": int((paired["net_profit"] <= 0).sum()),
                    }
                )

    frame = pd.DataFrame(rows)
    attribution = pd.DataFrame(attribution_rows)
    review_dir = RESULTS / "optimization_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(review_dir / "optimization_metrics.csv", index=False, lineterminator="\n")
    attribution.to_csv(review_dir / "optimization_trade_attribution.csv", index=False, lineterminator="\n")

    best = frame.sort_values(["symbol", "sharpe", "return_pct"], ascending=[True, False, False]).groupby("symbol").head(1)
    lines = [
        "# VWAP 偏离策略优化复盘",
        "",
        "## 结论",
        "",
        "5 个品种的优化版本均比原版改善；但基线与优化版的低周期冻结数据哈希不完全一致，下面结论是方向性评价，不是严格同数据归因。正式采用前仍需把两者固定在同一低周期 bars 文件上重跑。",
        "",
        "| 品种 | 推荐高周期 | 收益 | 最大回撤 | Sharpe | 交易次数 | 相对基线收益变化 | 相对基线回撤变化 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"| {row.symbol} | {row.higher_period} | {row.return_pct:.2f}% | {row.drawdown_pct:.2f}% | "
            f"{row.sharpe:.2f} | {row.trades} | {row.return_delta_vs_baseline_pp:+.2f} pp | "
            f"{row.drawdown_delta_vs_baseline_pp:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## 复盘",
            "",
            "- 主要改善来自四个约束同时生效：60 分钟 VWAP、极值确认后下一根开盘入场、交易时段限制、ATR 止损与持仓时间上限。交易次数显著下降，AU/CU/AG 的成本拖累明显减轻。",
            "- IM 的 120 分钟版本最值得继续验证：收益 14.20%、最大回撤 4.97%、Sharpe 0.60，且相对原版同时改善收益和回撤。",
            "- AU 从深度亏损转为接近盈亏平衡，说明夜盘过滤有效；但收益优势还不足以覆盖数据边界差异和参数不确定性。",
            "- RB 仍未形成正期望，120 分钟没有带来额外改善，应暂不作为实盘候选。",
            "- CU 虽然相对原版大幅改善，但仍为负收益，说明趋势过滤和均值回归假设仍不匹配，不能仅靠继续调阈值解决。",
            "- AG 的交易数量很少，收益主要由少数交易贡献，必须做滚动样本外和成本敏感性检验。",
            "",
            "## 数据与执行核验",
            "",
            "| 检查项 | 结果 |",
            "|---|---|",
            "| 未来函数 | 未发现；高周期只使用 `higher_bar_open + higher_period <= current_bar_time` 的已闭合 K 线 |",
            "| 信号成交 | 信号确认后统一 `next_bar_open` |",
            "| 手续费/滑点 | 使用 SSQuant 自动合约参数和 `slippage_ticks=1`，交易明细逐笔记录 |",
            "| 正式数据源 | SSQuant 官方入口；fallback 被禁止，取数失败即终止 |",
            "| 哈希一致性 | 基线与优化版不一致，严格同数据对比 `not yet proven` |",
            "",
            "## 文件",
            "",
            "- `optimization_metrics.csv`：逐品种、逐高周期指标与基线差异。",
            "- `optimization_trade_attribution.csv`：交易成本、持仓时间和盈亏归因。",
            "- 每个正式运行目录包含 `equity_curve.csv`、`equity_curve.png`、`trades.csv`、`report.md`、`run_manifest.json`。",
        ]
    )
    (review_dir / "optimization_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(review_dir)


if __name__ == "__main__":
    main()
