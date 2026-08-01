"""Prepare schema-v2 frozen SSQuant bars for index MTF validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODEX_SKILL_ROOT = Path.home() / ".codex" / "skills" / "ssquant-backtest"
sys.path.insert(0, str(CODEX_SKILL_ROOT))
sys.path.insert(1, str(PROJECT_ROOT / "scripts"))

from research.project_paths import RUNS_ROOT

RUNS_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(RUNS_ROOT)

from shared.data_fallback import inject, verify_inject_active
from shared.frozen_data import freeze_frames
from research.index_matrix_contract import period_minutes, source_id


ADJUST_TYPE = "1"
COVERAGE = 0.999
DEFAULT_SYMBOLS = ("IM888", "IF888", "IC888")
DEFAULT_PERIODS = ("5m", "15m", "30m", "60m", "90m", "120m")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen index MTF data")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--periods", nargs="+", default=list(DEFAULT_PERIODS))
    parser.add_argument("--start-date", default="2024-01-02")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument(
        "--output-dir",
        default=str(RUNS_ROOT / "index_mtf" / "frozen_dataset"),
    )
    return parser.parse_args()


def _fetch_frame(symbol: str, period: str, start_date: str, end_date: str):
    import ssquant.data.api_data_fetcher as api_data_fetcher

    frame = api_data_fetcher.get_futures_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        kline_period=period,
        adjust_type=ADJUST_TYPE,
        use_cache=True,
        save_data=True,
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"SSQuant returned no bars: {symbol} {period}")
    markers = [
        str(frame.attrs.get(key, "")).lower()
        for key in ("source", "data_source", "_source")
    ]
    if any("tqsdk" in marker for marker in markers):
        raise RuntimeError(f"TqSdk fallback detected: {symbol} {period}")
    return frame


def main() -> int:
    args = _parse_args()
    for period in args.periods:
        period_minutes(period)

    inject()
    verify_inject_active()

    frames = []
    for symbol in args.symbols:
        for period in args.periods:
            frame = _fetch_frame(symbol, period, args.start_date, args.end_date)
            frames.append(
                {
                    "frame": frame,
                    "source_id": source_id(symbol, period, ADJUST_TYPE),
                    "symbol": symbol,
                    "kline_period": period,
                    "adjust_type": ADJUST_TYPE,
                    "adjustment_mode": "ssquant_adjust_type_1",
                    "coverage": COVERAGE,
                    "coverage_detail": "official preparation log: 99.9%",
                    "source_mode": "ssquant_official",
                }
            )
            print(
                f"[PREPARED] {symbol} {period} rows={len(frame)} "
                f"range={frame.index.min()}~{frame.index.max()}"
            )

    manifest = freeze_frames(
        frames,
        args.output_dir,
        metadata={
            "formal": True,
            "pipeline_id": "official-ssquant-project-pipeline",
            "pipeline_version": "ssquant-v0.4.6-v5",
            "data_source": "ssquant",
            "transformed": True,
            "adjustment_version": "adjust_type_1",
            "coverage_policy": "strict",
            "symbols": list(args.symbols),
            "periods": list(args.periods),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "data_mode": "official_pipeline_prepare",
        },
    )
    spec_path = Path(args.output_dir).expanduser().resolve() / "matrix_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "symbols": list(args.symbols),
                "periods": list(args.periods),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "dataset_manifest": manifest["manifest_path"],
                "source_count": len(manifest["sources"]),
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[DATASET] {manifest['manifest_path']}")
    print(f"[SPEC] {spec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
