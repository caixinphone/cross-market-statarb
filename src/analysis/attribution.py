"""Return & risk attribution (Task 5).

The headline tool is :func:`pnl_decomposition`, which splits realised gross PnL
into the part the strategy *intends* to earn (idiosyncratic reversion) and the
part it does not:

    gross = idiosyncratic (Σ held_asset · ε)            <- the reversion edge
          + non_idiosyncratic (alpha drift + hedge error) <- unhedgeable drift / β-drift
    net   = gross − fees − funding − borrow

This makes visible the central finding that the clean ε-edge is positive but is
eroded by the alpha drift of shorting high-momentum names and by trading costs.
"""

from __future__ import annotations

import pandas as pd

from ..config import Config
from ..factors.factor_model import FactorModelResult


def pnl_decomposition(cfg: Config, result, fm: FactorModelResult,
                      residuals: pd.DataFrame) -> pd.Series:
    tradeable = [c for c in result.held.columns if c in residuals.columns]
    held_asset = result.held[tradeable]
    eps = residuals.reindex(result.held.index)[tradeable].fillna(0.0)

    idio = (held_asset * eps).sum().sum()
    gross = result.pnl["gross"].sum()
    decomp = pd.Series({
        "idiosyncratic_edge": idio,
        "alpha_drift_and_hedge_error": gross - idio,
        "gross_total": gross,
        "fees": result.pnl["fees"].sum(),
        "funding": result.pnl["funding"].sum(),
        "borrow": result.pnl["borrow"].sum(),
        "net_total": result.pnl["net"].sum(),
    })
    return decomp.round(0)


def per_asset_pnl(result, residuals: pd.DataFrame) -> pd.Series:
    """Idiosyncratic PnL contribution per asset (Σ held·ε)."""
    tradeable = [c for c in result.held.columns if c in residuals.columns]
    held_asset = result.held[tradeable]
    eps = residuals.reindex(result.held.index)[tradeable].fillna(0.0)
    return (held_asset * eps).sum().sort_values()


def drawdown_table(equity: pd.Series, top: int = 5) -> pd.DataFrame:
    """Top-N drawdown episodes with peak/trough/recovery dates and depth."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    episodes = []
    in_dd = False
    start = trough = None
    for ts, d in dd.items():
        if not in_dd and d < 0:
            in_dd, start, trough = True, ts, ts
        elif in_dd:
            if d < dd.loc[trough]:
                trough = ts
            if d >= 0:  # recovered
                episodes.append((start, trough, ts, float(dd.loc[trough])))
                in_dd = False
    if in_dd:
        episodes.append((start, trough, None, float(dd.loc[trough])))
    tbl = pd.DataFrame(episodes,
                       columns=["peak", "trough", "recovery", "depth"])
    return tbl.sort_values("depth").head(top).reset_index(drop=True)


def periodic_returns(result, freq: str = "YE") -> pd.Series:
    """Period (year/month) net returns from the equity curve."""
    eq = result.equity
    grouped = eq.resample(freq).last()
    base = pd.Series([eq.iloc[0]], index=[eq.index[0]])
    full = pd.concat([base, grouped])
    return (full.pct_change().dropna()).round(4)
