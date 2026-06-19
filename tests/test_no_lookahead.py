"""The critical correctness test: no lookahead.

If signals at bar ``t`` truly use only data through ``t``, then perturbing *future*
bars (``t+1…``) must leave every signal at/<=``t`` byte-for-byte unchanged. We
build a small synthetic panel, run the factor model + signals, perturb the tail,
re-run, and assert the head is identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.factors.factor_model import fit_rolling_factor_model
from src.signals.zscore import compute_zscore, generate_signals


def _synthetic_panel(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    btc = rng.normal(0, 0.03, n)
    qqq = rng.normal(0, 0.01, n)
    # asset = 1.5*BTC + 0.4*QQQ + mean-reverting residual
    resid = np.zeros(n)
    for t in range(1, n):
        resid[t] = -0.2 * resid[t - 1] + rng.normal(0, 0.02)
    mstr = 1.5 * btc + 0.4 * qqq + resid
    returns = pd.DataFrame({"MSTR": mstr, "BTC": btc, "QQQ": qqq}, index=idx)
    factor_returns = returns[["BTC", "QQQ"]]
    return returns, factor_returns


def _config_for_synth():
    cfg = load_config()
    cfg.raw["active_universe"] = "synthetic"
    cfg.raw["universes"]["synthetic"] = {"crypto_equities": ["MSTR"]}
    cfg.raw["factors"] = ["BTC", "QQQ"]
    cfg.raw["factor_map"]["crypto_equities"] = ["BTC", "QQQ"]
    cfg.raw["factor_overrides"] = {}
    return cfg


def test_factor_model_and_signals_are_lookahead_free():
    cfg = _config_for_synth()
    returns, factor_returns = _synthetic_panel()

    fm = fit_rolling_factor_model(cfg, returns, factor_returns)
    sig = generate_signals(cfg, fm.residuals)

    cut = 300  # perturb everything strictly after this bar
    pert_returns = returns.copy()
    pert_factors = factor_returns.copy()
    rng = np.random.default_rng(123)
    pert_returns.iloc[cut + 1:] += rng.normal(0, 0.05, pert_returns.iloc[cut + 1:].shape)
    pert_factors.iloc[cut + 1:] += rng.normal(0, 0.05, pert_factors.iloc[cut + 1:].shape)
    pert_returns["BTC"] = pert_factors["BTC"]
    pert_returns["QQQ"] = pert_factors["QQQ"]

    fm2 = fit_rolling_factor_model(cfg, pert_returns, pert_factors)
    sig2 = generate_signals(cfg, fm2.residuals)

    # Residuals and signals up to the cut must be unchanged.
    pd.testing.assert_series_equal(
        fm.residuals["MSTR"].iloc[:cut + 1],
        fm2.residuals["MSTR"].iloc[:cut + 1],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        sig.target_sign["MSTR"].iloc[:cut + 1],
        sig2.target_sign["MSTR"].iloc[:cut + 1],
        check_names=False,
    )


def test_zscore_uses_only_trailing_data():
    cfg = _config_for_synth()
    returns, factor_returns = _synthetic_panel()
    fm = fit_rolling_factor_model(cfg, returns, factor_returns)

    z = compute_zscore(cfg, fm.residuals)
    resid2 = fm.residuals.copy()
    resid2.iloc[350:] = np.nan          # destroy the tail
    z2 = compute_zscore(cfg, resid2)

    pd.testing.assert_series_equal(
        z["MSTR"].iloc[:350], z2["MSTR"].iloc[:350], check_names=False)
