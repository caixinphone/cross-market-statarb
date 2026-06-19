"""End-to-end backtest: data -> factors -> signals -> portfolio -> backtest -> metrics.

Usage:
    python scripts/run_backtest.py [--config PATH]

Assumes ``scripts/download_data.py`` has populated the cache / processed panel.
Reproducible: the global seed from config is set before anything stochastic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.attribution import (drawdown_table, periodic_returns,
                                       per_asset_pnl, pnl_decomposition)
from src.analysis.metrics import compute_metrics, format_metrics
from src.analysis.plots import (plot_equity_and_drawdown,
                                plot_factor_exposure, plot_pnl_decomposition)
from src.config import PROJECT_ROOT, load_config
from src.data.align import Panel
from src.data.fetch import fetch_funding
from src.factors.diagnostics import diagnostics_table
from src.factors.factor_model import fit_rolling_factor_model
from src.portfolio.construct import size_positions
from src.portfolio.execution import apply_no_trade_band
from src.portfolio.risk import apply_risk_constraints, build_risk_report
from src.signals.zscore import generate_signals
from src.backtest.engine import run_backtest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg.seed)

    panel = Panel.load(cfg)
    funding = fetch_funding(cfg)  # cached

    print("Fitting factor model ...")
    fm = fit_rolling_factor_model(cfg, panel.returns, panel.factor_returns)
    fm.save(cfg.processed_dir)

    print("\n=== Factor diagnostics ===")
    print(diagnostics_table(cfg, fm).to_string())

    print("\nGenerating signals + sizing ...")
    sig = generate_signals(cfg, fm.residuals)
    asset_notional = size_positions(cfg, sig, fm)
    port = apply_risk_constraints(cfg, asset_notional, fm)
    # Realistic execution: throttle churn with a no-trade band -> actually-held book.
    actual = apply_no_trade_band(cfg, port.positions)
    report = build_risk_report(cfg, actual, list(port.asset_legs.columns), fm)

    print("Running backtest ...")
    result = run_backtest(cfg, panel.returns, actual, funding, panel.dollar_volume)
    result.save(cfg.processed_dir)
    report.to_parquet(cfg.processed_dir / "risk_report.parquet")

    print("\n=== Constraint check (actual held book) ===")
    rep = report
    print(f"  gross leverage  max {rep.gross_leverage.max():.2f}x  (cap "
          f"{cfg.raw['portfolio']['risk']['max_gross_leverage']}x)")
    print(f"  max asset wt    max {rep.max_asset_weight.max():.3f}   (cap "
          f"{cfg.raw['portfolio']['risk']['max_asset_weight']})")
    print(f"  max |netfactor| max {rep.max_abs_netfactor.max():.3f}   (cap "
          f"{cfg.raw['portfolio']['risk']['max_net_factor_exposure']})")

    print("\n=== PnL attribution ($) ===")
    print((result.pnl.sum().round(0)).to_string())

    print("\n=== Performance metrics ===")
    m = compute_metrics(cfg, result)
    print(format_metrics(m))

    print("\n=== PnL decomposition (idiosyncratic edge vs frictions) ===")
    decomp = pnl_decomposition(cfg, result, fm, fm.residuals)
    print(decomp.to_string())

    print("\n=== Idiosyncratic PnL by asset ($) ===")
    print(per_asset_pnl(result, fm.residuals).round(0).to_string())

    print("\n=== Top drawdowns ===")
    print(drawdown_table(result.equity).to_string())

    print("\n=== Annual net returns ===")
    print((periodic_returns(result, "YE") * 100).round(2).to_string())

    plot_equity_and_drawdown(result, PROJECT_ROOT)
    plot_pnl_decomposition(decomp, PROJECT_ROOT)
    plot_factor_exposure(report, cfg.factor_names(), PROJECT_ROOT)
    print(f"\nCharts written to {PROJECT_ROOT / 'reports'}/")


if __name__ == "__main__":
    main()
