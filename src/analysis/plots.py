"""Report charts (Task 5). Saves PNGs under ``reports/``.

Headless backend so it runs in CI / over SSH without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _outdir(root: Path) -> Path:
    d = root / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_equity_and_drawdown(result, root: Path) -> Path:
    eq = result.equity
    dd = eq / eq.cummax() - 1.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(eq.index, eq.values, color="navy", lw=1.3)
    ax1.set_title("Equity curve (net of costs)")
    ax1.set_ylabel("Account equity ($)")
    ax1.grid(alpha=0.3)
    ax2.fill_between(dd.index, dd.values * 100, 0, color="firebrick", alpha=0.5)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("%")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = _outdir(root) / "equity_drawdown.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_pnl_decomposition(decomp: pd.Series, root: Path) -> Path:
    parts = decomp[["idiosyncratic_edge", "alpha_drift_and_hedge_error",
                    "fees", "funding", "borrow", "net_total"]]
    colors = ["seagreen" if v >= 0 else "firebrick" for v in parts.values]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(parts.index, parts.values, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("PnL attribution ($)")
    ax.set_xticklabels(parts.index, rotation=30, ha="right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = _outdir(root) / "pnl_attribution.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_factor_exposure(risk_report: pd.DataFrame, factor_names, root: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    for f in factor_names:
        col = f"netfactor_{f}"
        if col in risk_report:
            ax.plot(risk_report.index, risk_report[col] * 100, lw=0.9, label=f)
    ax.axhline(5, color="grey", ls="--", lw=0.7)
    ax.axhline(-5, color="grey", ls="--", lw=0.7)
    ax.set_title("Net factor exposure (% AUM); dashed = ±5% limit")
    ax.set_ylabel("% AUM")
    ax.legend(ncol=len(factor_names), fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = _outdir(root) / "factor_exposure.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
