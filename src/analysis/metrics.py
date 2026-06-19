"""Performance metrics (Task 5).

All annualisation uses periods-per-year from the run frequency (252 daily,
252·7 hourly — 7 RTH bars/day). Sharpe/Sortino assume a zero risk-free rate (the book is
cash-neutral / market-neutral, so excess return ≈ return).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..portfolio.construct import periods_per_year


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough = dd.idxmin()
    peak_date = equity.loc[:trough].idxmax()
    return float(dd.min()), peak_date, trough


def _avg_holding_period(held: pd.DataFrame) -> float:
    """Mean number of consecutive bars a position stays open, across instruments."""
    runs = []
    active = held.abs() > 1.0
    for col in active.columns:
        a = active[col].to_numpy()
        length = 0
        for v in a:
            if v:
                length += 1
            elif length:
                runs.append(length)
                length = 0
        if length:
            runs.append(length)
    return float(np.mean(runs)) if runs else 0.0


def compute_metrics(cfg: Config, result) -> pd.Series:
    ppy = periods_per_year(cfg)
    r = result.returns.dropna()
    equity = result.equity.dropna()

    n = len(r)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (ppy / n) - 1 if n else np.nan
    ann_vol = r.std() * np.sqrt(ppy)
    sharpe = (r.mean() / r.std() * np.sqrt(ppy)) if r.std() > 0 else np.nan
    downside = r[r < 0].std()
    sortino = (r.mean() / downside * np.sqrt(ppy)) if downside > 0 else np.nan
    mdd, peak_dt, trough_dt = max_drawdown(equity)
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    net = result.pnl["net"]
    gains = net[net > 0].sum()
    losses = -net[net < 0].sum()
    profit_factor = gains / losses if losses > 0 else np.nan
    win_rate = (net > 0).mean()

    ann_turnover = result.turnover.mean() * ppy
    # Holding period on the *signal* (asset) legs only — hedge legs are almost
    # always on, which would otherwise inflate the figure.
    tradeable = cfg.tradeable_names()
    asset_held = result.held[[c for c in tradeable if c in result.held.columns]]
    avg_hold = _avg_holding_period(asset_held)

    return pd.Series({
        "ann_return_cagr": cagr,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "win_rate_bar": win_rate,
        "profit_factor": profit_factor,
        "avg_holding_bars": avg_hold,
        "ann_turnover_2way": ann_turnover,
        "n_bars": n,
    })


def format_metrics(m: pd.Series) -> str:
    pct = {"ann_return_cagr", "ann_volatility", "max_drawdown", "win_rate_bar"}
    lines = []
    for k, v in m.items():
        if k in pct:
            lines.append(f"  {k:<20} {v*100:8.2f}%")
        elif k in {"n_bars"}:
            lines.append(f"  {k:<20} {int(v):8d}")
        else:
            lines.append(f"  {k:<20} {v:8.2f}")
    return "\n".join(lines)
