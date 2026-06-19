"""Position sizing — equal-volatility risk budget on the asset legs.

Each active signal is sized so the hedged pair targets a common volatility:

    N_i,t = (target_pair_vol / σ_resid_i,t) · AUM_per_signal

σ_resid is the trailing (point-in-time) annualised idiosyncratic vol — the right
risk measure because the factor hedge removes the systematic variance, leaving
the residual as the pair's risk. AUM_per_signal = AUM / max_concurrent_signals.

This module only sizes the **asset** legs; :mod:`src.portfolio.risk` derives the
factor hedge legs from the (post-constraint) asset legs so factor-neutrality is
preserved by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..factors.factor_model import FactorModelResult
from ..signals.zscore import SignalResult


def periods_per_year(cfg: Config) -> float:
    # Hourly grid keeps the 7-bar RTH core (14:00-20:00 UTC) per trading day.
    return 252.0 if cfg.frequency == "daily" else 252.0 * 7.0


def rolling_resid_vol(cfg: Config, residuals: pd.DataFrame) -> pd.DataFrame:
    """Trailing annualised residual vol (point-in-time)."""
    window = int(cfg.get("signals", "zscore_window", default=60))
    ann = np.sqrt(periods_per_year(cfg))
    return residuals.rolling(window, min_periods=window).std() * ann


def size_positions(
    cfg: Config,
    signals: SignalResult,
    fm: FactorModelResult,
) -> pd.DataFrame:
    """Return signed asset-leg notionals ($), index=bars, cols=tradeable assets.

    Notional is **fixed at entry** and held constant for the life of each trade
    (size-at-entry), as a desk actually does: you set the size when you put the
    position on, not by re-vol-targeting every bar. Re-sizing each bar would
    generate large, unrealistic turnover from σ_resid simply drifting.
    """
    aum = float(cfg.get("portfolio", "aum", default=10_000_000))
    max_concurrent = int(cfg.get("portfolio", "max_concurrent_signals", default=40))
    target_vol = float(cfg.get("portfolio", "target_pair_vol", default=0.15))
    aum_per_signal = aum / max_concurrent

    resid_vol = rolling_resid_vol(cfg, fm.residuals)
    # Notional per unit signal, clipped so a tiny vol estimate can't explode size.
    floor_vol = target_vol / 10.0  # cap leverage-per-signal at ~10x
    base_notional = (target_vol / resid_vol.clip(lower=floor_vol)) * aum_per_signal

    sign = signals.target_sign.reindex(columns=base_notional.columns).fillna(0.0)
    # Freeze the notional at each holding episode's first bar (entry).
    entry_notional = pd.DataFrame(index=sign.index, columns=sign.columns, dtype=float)
    for col in sign.columns:
        s = sign[col]
        episode = (s != s.shift()).cumsum()
        entry_notional[col] = base_notional[col].groupby(episode).transform("first")

    return (sign * entry_notional).fillna(0.0)
