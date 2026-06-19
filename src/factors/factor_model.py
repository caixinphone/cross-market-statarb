"""Rolling multi-factor model → idiosyncratic residuals.

For each tradeable asset we fit, on a trailing window, the linear model

    r_i,t = α_i + Σ_f β_{i,f} · r_f,t + ε_i,t

The idiosyncratic residual ε is what the strategy trades. Betas are also the
hedge ratios used to neutralise factor exposure in the portfolio.

**Point-in-time contract.** Betas/residuals at bar ``t`` use only data through
``t`` (a trailing window ending at ``t``). No-lookahead is then guaranteed by the
backtest engine executing signals on bar ``t+1``: nothing at ``t`` ever depends on
``t+1…``. ``tests/test_no_lookahead.py`` enforces this by perturbing future bars
and asserting past betas/residuals are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from ..config import Config


@dataclass
class FactorModelResult:
    residuals: pd.DataFrame              # index=bars, cols=tradeable assets
    r2: pd.DataFrame                     # rolling R² per asset
    betas: dict[str, pd.DataFrame]       # asset -> [const, f1, f2, ...] over time
    factor_map: dict[str, list[str]]     # asset -> factor names used

    def save(self, processed_dir) -> None:
        self.residuals.to_parquet(processed_dir / "residuals.parquet")
        self.r2.to_parquet(processed_dir / "r2.parquet")


def fit_rolling_factor_model(
    cfg: Config,
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> FactorModelResult:
    window = int(cfg.get("factor_model", "rolling_window", default=90))
    min_obs = int(cfg.get("factor_model", "min_obs", default=60))
    universe = cfg.universe()

    residuals = {}
    r2 = {}
    betas: dict[str, pd.DataFrame] = {}
    factor_map: dict[str, list[str]] = {}

    for name in cfg.tradeable_names():
        spec = universe[name]
        factors = [f for f in spec.factors if f in factor_returns.columns]
        if not factors or name not in returns.columns:
            continue
        factor_map[name] = factors

        # Joint non-NaN sample so an asset's pre-listing gap doesn't poison windows.
        data = pd.concat([returns[name].rename("y"), factor_returns[factors]],
                         axis=1).dropna()
        if len(data) < min_obs + 1:
            continue

        y = data["y"]
        X = sm.add_constant(data[factors], has_constant="add")
        k = X.shape[1]
        model = RollingOLS(y, X, window=window, min_nobs=max(min_obs, k + 1),
                           expanding=False)
        res = model.fit(params_only=False)

        params = res.params                      # rolling betas (window ending t)
        fitted = (params.values * X.values).sum(axis=1)
        resid = pd.Series(y.values - fitted, index=data.index, name=name)

        residuals[name] = resid.reindex(returns.index)
        r2[name] = res.rsquared.reindex(returns.index)
        betas[name] = params.reindex(returns.index)

    return FactorModelResult(
        residuals=pd.DataFrame(residuals).reindex(returns.index),
        r2=pd.DataFrame(r2).reindex(returns.index),
        betas=betas,
        factor_map=factor_map,
    )
