"""Risk constraints must hold after construction, and pure-factor hedges must
neutralise systematic exposure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.factors.factor_model import FactorModelResult
from src.portfolio.risk import apply_risk_constraints


def _setup():
    cfg = load_config()
    cfg.raw["active_universe"] = "synth"
    cfg.raw["universes"]["synth"] = {"crypto_equities": ["MSTR", "COIN"]}
    cfg.raw["factors"] = ["BTC"]
    cfg.raw["factor_map"]["crypto_equities"] = ["BTC"]
    cfg.raw["factor_overrides"] = {}
    cfg.raw["portfolio"]["aum"] = 10_000_000

    idx = pd.date_range("2023-01-01", periods=50, freq="B", tz="UTC")
    # Deliberately oversized notionals to force the per-asset cap to bind.
    asset_notional = pd.DataFrame(
        {"MSTR": 2_000_000.0, "COIN": -1_500_000.0}, index=idx)
    betas = {
        "MSTR": pd.DataFrame({"const": 0.0, "BTC": 1.5}, index=idx),
        "COIN": pd.DataFrame({"const": 0.0, "BTC": 1.2}, index=idx),
    }
    fm = FactorModelResult(
        residuals=pd.DataFrame(0.0, index=idx, columns=["MSTR", "COIN"]),
        r2=pd.DataFrame(0.0, index=idx, columns=["MSTR", "COIN"]),
        betas=betas,
        factor_map={"MSTR": ["BTC"], "COIN": ["BTC"]},
    )
    return cfg, asset_notional, fm


def test_caps_hold():
    cfg, asset_notional, fm = _setup()
    pr = apply_risk_constraints(cfg, asset_notional, fm)
    aum = cfg.raw["portfolio"]["aum"]
    rc = cfg.raw["portfolio"]["risk"]

    assert pr.report["max_asset_weight"].max() <= rc["max_asset_weight"] + 1e-9
    assert pr.report["gross_leverage"].max() <= rc["max_gross_leverage"] + 1e-9
    # Per-asset cap actually binds given the oversized inputs.
    assert np.isclose(pr.report["max_asset_weight"].max(),
                      rc["max_asset_weight"], atol=1e-6)


def test_pure_factor_is_neutralised():
    cfg, asset_notional, fm = _setup()
    pr = apply_risk_constraints(cfg, asset_notional, fm)
    # BTC is a pure factor here -> net exposure should be ~0 by construction
    # (before any per-asset clip on the factor leg, which the small book avoids).
    assert pr.report["netfactor_BTC"].abs().max() < 1e-6
