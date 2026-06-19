"""Factor-model diagnostics (Task 2 reporting requirement).

* Mean rolling R² per asset — how much variance the factors explain.
* ADF stationarity test on each residual series — the strategy only has an edge
  where the idiosyncratic residual is mean-reverting (stationary). Assets whose
  residuals fail the ADF test are flagged so they can be dropped from the
  tradeable set.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import adfuller

from ..config import Config
from .factor_model import FactorModelResult


def diagnostics_table(cfg: Config, fm: FactorModelResult) -> pd.DataFrame:
    adf_max = float(cfg.get("factor_model", "adf_pvalue_max", default=0.10))
    rows = []
    for name in fm.residuals.columns:
        resid = fm.residuals[name].dropna()
        if len(resid) < 30:
            continue
        try:
            adf_stat, adf_p = adfuller(resid, autolag="AIC")[:2]
        except Exception:
            adf_stat, adf_p = float("nan"), float("nan")
        rows.append({
            "asset": name,
            "factors": "+".join(fm.factor_map.get(name, [])),
            "mean_r2": round(float(fm.r2[name].mean()), 3),
            "resid_vol_ann": round(float(resid.std() * (252 ** 0.5)), 3),
            "adf_stat": round(float(adf_stat), 2),
            "adf_pvalue": round(float(adf_p), 4),
            "stationary": bool(adf_p <= adf_max),
            "n_obs": int(len(resid)),
        })
    table = pd.DataFrame(rows).set_index("asset")
    return table.sort_values("mean_r2", ascending=False)
