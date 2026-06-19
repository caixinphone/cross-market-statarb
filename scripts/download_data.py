"""CLI: download + cache raw data and build the aligned panel.

Usage:
    python scripts/download_data.py [--config PATH] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.align import build_panel
from src.data.fetch import fetch_funding, fetch_universe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="ignore cache")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Universe '{cfg.raw['active_universe']}' | freq={cfg.frequency} | "
          f"{cfg.start} -> {cfg.end}")

    print("Fetching OHLCV ...")
    ohlcv = fetch_universe(cfg, force=args.force)
    print("Fetching funding ...")
    fetch_funding(cfg, force=args.force)

    print("Building aligned panel ...")
    panel = build_panel(cfg, ohlcv)
    panel.save(cfg)

    r = panel.returns
    print(f"\nPanel: {r.shape[0]} bars x {r.shape[1]} assets "
          f"({r.index[0].date()} -> {r.index[-1].date()})")
    print(f"Factors: {list(panel.factor_returns.columns)}")
    miss = (r.isna().mean() * 100).round(1).sort_values(ascending=False)
    print("Missing % by asset (top 8):")
    print(miss.head(8).to_string())


if __name__ == "__main__":
    main()
