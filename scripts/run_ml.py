"""ML enhancement (bonus / Task 6): meta-learning size overlay vs the baseline.

Pipeline:
    data → factors → signals (layer-1 timing) → walk-forward CNN sizer (layer-2)
         → CNN-modulated notionals → SAME risk caps / hedge / no-trade-band / engine
         → compare CNN-sized book vs the equal-volatility baseline.

The CNN multiplier book is renormalised to the baseline's average gross so the
comparison measures *allocation skill*, not leverage. Prints a metrics table +
interpretability summary and writes reports/ml_equity.png, ml_interpretability.png.

Usage:  pip install -e ".[ml]"  &&  python scripts/run_ml.py [--config PATH]
Reproducible: global + torch seeds from config.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.attribution import pnl_decomposition
from src.analysis.metrics import compute_metrics
from src.config import PROJECT_ROOT, load_config
from src.data.align import Panel
from src.data.fetch import fetch_funding
from src.factors.factor_model import fit_rolling_factor_model
from src.ml.meta_sizing import walk_forward_sizes
from src.portfolio.construct import size_positions
from src.portfolio.execution import apply_no_trade_band
from src.portfolio.risk import apply_risk_constraints
from src.signals.zscore import generate_signals
from src.backtest.engine import run_backtest


def _backtest(cfg, panel, funding, fm, asset_notional):
    port = apply_risk_constraints(cfg, asset_notional, fm)
    actual = apply_no_trade_band(cfg, port.positions)
    res = run_backtest(cfg, panel.returns, actual, funding, panel.dollar_volume)
    m = compute_metrics(cfg, res)
    d = pnl_decomposition(cfg, res, fm, fm.residuals)
    return res, m, d


def _variant(cfg, panel, funding, fm, *, gate: bool, cnn: bool):
    """Run one configuration: optional drift gate (signal layer) + optional CNN
    sizer (size layer). Returns (res, metrics, decomp, n_shorts, meta_or_None)."""
    c = copy.deepcopy(cfg)
    c.raw["signals"]["drift_gate"]["enabled"] = gate
    sig = generate_signals(c, fm.residuals, panel.returns)
    n_short = int((sig.target_sign < 0).sum().sum())
    notional = size_positions(c, sig, fm)
    ms = None
    if cnn:
        ms = walk_forward_sizes(c, fm, sig, panel.returns, panel.factor_returns, funding)
        mln = notional * ms.multiplier.reindex_like(notional).fillna(1.0)
        bg, mg = notional.abs().sum(axis=1), mln.abs().sum(axis=1)
        notional = mln * (bg[bg > 0].mean() / mg[mg > 0].mean())   # match own gross
    res, m, d = _backtest(c, panel, funding, fm, notional)
    return res, m, d, n_short, ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg.seed)

    panel = Panel.load(cfg)
    funding = fetch_funding(cfg)
    fm = fit_rolling_factor_model(cfg, panel.returns, panel.factor_returns)

    print("Running baseline / CNN sizer / drift gate / gate+CNN ...")
    b_res, b_m, b_d, b_ns, _ = _variant(cfg, panel, funding, fm, gate=False, cnn=False)
    c_res, c_m, c_d, _, c_ms = _variant(cfg, panel, funding, fm, gate=False, cnn=True)
    g_res, g_m, g_d, g_ns, _ = _variant(cfg, panel, funding, fm, gate=True, cnn=False)
    gc_res, gc_m, gc_d, _, _ = _variant(cfg, panel, funding, fm, gate=True, cnn=True)

    # legacy "tilt" sizing (gentle, centred on 1) for an honest sizing comparison.
    cfg_tilt = copy.deepcopy(cfg)
    cfg_tilt.raw["ml"].update(sizing="tilt", floor_long=0.25, floor_short=0.25,
                              short_veto_pct=0.0)
    t_res, t_m, t_d, _, _ = _variant(cfg_tilt, panel, funding, fm, gate=False, cnn=True)

    def row(name, m, d):
        return (f"  {name:<28} net ${d['net_total']:>11,.0f} | sharpe {m['sharpe']:>6.2f} "
                f"| gross ${d['gross_total']:>9,.0f} | drift ${d['alpha_drift_and_hedge_error']:>10,.0f} "
                f"| turn {m['ann_turnover_2way']:>4.1f}")
    print("\n=== Optimised ML vs baseline (gross-matched) ===")
    print(row("baseline (equal-vol)", b_m, b_d))
    print(row("+ CNN sizer (conviction)", c_m, c_d))
    print(row("+ CNN sizer (tilt, legacy)", t_m, t_d))
    print(row("+ drift gate (no short bull)", g_m, g_d))
    print(row("+ drift gate + CNN (convict.)", gc_m, gc_d))
    print(f"\n  drift gate blocked {(1 - g_ns / b_ns) * 100:.0f}% of short bars "
          f"({b_ns:,} → {g_ns:,}); alpha-drift {b_d['alpha_drift_and_hedge_error']:,.0f} "
          f"→ {g_d['alpha_drift_and_hedge_error']:,.0f}")

    # ---- CNN interpretability (the sizing layer that helps) ----
    tt = c_ms.trades[c_ms.trades.is_test & c_ms.trades.y_pred.notna()]
    print("\n=== CNN sizer interpretability ===")
    print(f"  OOS calibration corr(pred, realised net margin): {c_ms.oos_corr:+.3f} (n={len(tt)})")
    print("  permutation importance (ΔMSE, OOS):")
    for ch, v in c_ms.importance.items():
        print(f"    {ch:<11} {v:+.5f}")

    # ---- charts ----
    rep = PROJECT_ROOT / "reports"
    fig, ax = plt.subplots(figsize=(9, 4))
    for res, lab in [(b_res, "baseline"), (c_res, "+CNN sizer"),
                     (g_res, "+drift gate"), (gc_res, "+gate+CNN")]:
        ax.plot(res.equity.index, res.equity, label=lab)
    ax.set_title("Equity: baseline vs CNN sizer vs drift gate (gross-matched)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(rep / "ml_equity.png", dpi=110); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    c_ms.importance.iloc[::-1].plot.barh(ax=axes[0])
    axes[0].set_title("CNN channel importance (ΔMSE, OOS)")
    axes[1].scatter(tt.y_pred, tt.y_true, s=8, alpha=0.4)
    axes[1].axhline(0, color="k", lw=0.5); axes[1].axvline(0, color="k", lw=0.5)
    axes[1].set_xlabel("predicted net margin"); axes[1].set_ylabel("realised net margin")
    axes[1].set_title(f"OOS calibration (corr {c_ms.oos_corr:+.2f})")
    fig.tight_layout(); fig.savefig(rep / "ml_interpretability.png", dpi=110); plt.close(fig)
    print(f"\nCharts written to {rep}/ (ml_equity.png, ml_interpretability.png)")


if __name__ == "__main__":
    main()
