"""Task 5: parameter / cost / capacity sensitivity analysis.

Runs the pipeline across grids of one knob at a time (everything else at the
config default) and tabulates net PnL, Sharpe, turnover, max drawdown, and the
gross/idiosyncratic decomposition. Writes ``reports/sensitivity.md``.

Robustness, not optimisation: a strategy that only works at one parameter point
is overfit. We want a broad plateau.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.analysis.attribution import pnl_decomposition
from src.analysis.metrics import compute_metrics
from src.backtest.engine import run_backtest
from src.config import PROJECT_ROOT, load_config
from src.data.align import Panel
from src.data.fetch import fetch_funding
from src.factors.factor_model import fit_rolling_factor_model
from src.portfolio.construct import size_positions
from src.portfolio.execution import apply_no_trade_band
from src.portfolio.risk import apply_risk_constraints
from src.signals.zscore import generate_signals


def evaluate(cfg, panel, funding, fm):
    sig = generate_signals(cfg, fm.residuals)
    an = size_positions(cfg, sig, fm)
    port = apply_risk_constraints(cfg, an, fm)
    actual = apply_no_trade_band(cfg, port.positions)
    res = run_backtest(cfg, panel.returns, actual, funding, panel.dollar_volume)
    m = compute_metrics(cfg, res)
    d = pnl_decomposition(cfg, res, fm, fm.residuals)
    return {
        "net$": d["net_total"], "sharpe": m["sharpe"], "calmar": m["calmar"],
        "maxDD": m["max_drawdown"], "turn": m["ann_turnover_2way"],
        "gross$": d["gross_total"], "idio$": d["idiosyncratic_edge"],
    }


def _row(label, r):
    return (f"| {label:<14} | {r['net$']:>10,.0f} | {r['sharpe']:>6.2f} | "
            f"{r['calmar']:>6.2f} | {r['maxDD']*100:>6.1f}% | {r['turn']:>5.1f} | "
            f"{r['gross$']:>9,.0f} | {r['idio$']:>9,.0f} |")


HEADER = ("| param | net$ | sharpe | calmar | maxDD | turn | gross$ | idio$ |\n"
          "|---|--:|--:|--:|--:|--:|--:|--:|")


def main():
    cfg = load_config()
    panel = Panel.load(cfg)
    funding = fetch_funding(cfg)
    base_raw = copy.deepcopy(cfg.raw)
    fm_cache = {}

    def fm_for(window):
        if window not in fm_cache:
            cfg.raw["factor_model"]["rolling_window"] = window
            fm_cache[window] = fit_rolling_factor_model(
                cfg, panel.returns, panel.factor_returns)
        return fm_cache[window]

    out = ["# Sensitivity & capacity analysis\n",
           f"Universe `{cfg.raw['active_universe']}`, frequency `{cfg.frequency}`, "
           f"{panel.returns.index[0].date()}–{panel.returns.index[-1].date()}.\n"]

    sweeps = {
        "entry_threshold": [1.5, 2.0, 2.5, 3.0],
        "exit_threshold": [0.25, 0.5, 1.0],
        "max_half_life_bars": [None, 30, 20, 12, 8],
        "zscore_window": [40, 60, 90],
        "target_pair_vol": [0.10, 0.15, 0.20],
        "no_trade_band": [0.0, 0.0015, 0.005],
    }
    for knob, values in sweeps.items():
        out.append(f"\n## {knob}\n{HEADER}")
        for v in values:
            cfg.raw = copy.deepcopy(base_raw)
            section = "signals" if knob in {
                "entry_threshold", "exit_threshold", "max_half_life_bars",
                "zscore_window"} else "portfolio"
            cfg.raw[section][knob] = v
            fm = fm_for(cfg.raw["factor_model"]["rolling_window"])
            out.append(_row(str(v), evaluate(cfg, panel, funding, fm)))

    # rolling window (refits the factor model)
    out.append(f"\n## factor rolling_window\n{HEADER}")
    for w in [60, 90, 120]:
        cfg.raw = copy.deepcopy(base_raw)
        cfg.raw["factor_model"]["rolling_window"] = w
        out.append(_row(str(w), evaluate(cfg, panel, funding, fm_for(w))))

    # cost sensitivity
    out.append(f"\n## cost: slippage_impact_bps\n{HEADER}")
    for imp in [6.0, 12.0, 24.0]:
        cfg.raw = copy.deepcopy(base_raw)
        cfg.raw["costs"]["slippage_impact_bps"] = imp
        out.append(_row(str(imp), evaluate(cfg, panel, funding,
                                           fm_for(base_raw["factor_model"]["rolling_window"]))))
    out.append(f"\n## cost: perp_fee_bps\n{HEADER}")
    for fee in [2.0, 4.0, 8.0]:
        cfg.raw = copy.deepcopy(base_raw)
        cfg.raw["costs"]["perp_fee_bps"] = fee
        out.append(_row(str(fee), evaluate(cfg, panel, funding,
                                           fm_for(base_raw["factor_model"]["rolling_window"]))))

    # capacity: scale AUM; liquidity slippage grows with participation -> shows
    # the capacity ceiling.
    out.append(f"\n## capacity: AUM ($)\n{HEADER}")
    for aum in [10e6, 25e6, 50e6, 100e6, 200e6]:
        cfg.raw = copy.deepcopy(base_raw)
        cfg.raw["portfolio"]["aum"] = aum
        out.append(_row(f"{aum/1e6:.0f}M", evaluate(cfg, panel, funding,
                                                    fm_for(base_raw["factor_model"]["rolling_window"]))))

    text = "\n".join(out) + "\n"
    (PROJECT_ROOT / "reports" / "sensitivity.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
