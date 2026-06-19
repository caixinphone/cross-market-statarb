"""Tests for signal sizing-at-entry and the half-life filter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.factors.factor_model import fit_rolling_factor_model
from src.signals.zscore import generate_signals, half_life_bars
from src.portfolio.construct import size_positions


def _synth():
    rng = np.random.default_rng(1)
    n = 500
    idx = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    btc = rng.normal(0, 0.03, n)
    resid = np.zeros(n)
    for t in range(1, n):
        resid[t] = -0.3 * resid[t - 1] + rng.normal(0, 0.02)
    asset = 1.4 * btc + resid
    returns = pd.DataFrame({"MSTR": asset, "BTC": btc}, index=idx)
    return returns, returns[["BTC"]]


def _cfg():
    cfg = load_config()
    cfg.raw["active_universe"] = "synth"
    cfg.raw["universes"]["synth"] = {"crypto_equities": ["MSTR"]}
    cfg.raw["factors"] = ["BTC"]
    cfg.raw["factor_map"]["crypto_equities"] = ["BTC"]
    cfg.raw["factor_overrides"] = {}
    # Pin small windows so the test is independent of the production default
    # (the hourly track uses 420; this 500-bar synthetic panel needs a short one).
    cfg.raw["factor_model"]["rolling_window"] = 60
    cfg.raw["factor_model"]["min_obs"] = 40
    cfg.raw["signals"]["zscore_window"] = 60
    return cfg


def test_size_is_constant_within_a_holding_episode():
    cfg = _cfg()
    returns, fr = _synth()
    fm = fit_rolling_factor_model(cfg, returns, fr)
    sig = generate_signals(cfg, fm.residuals)
    notional = size_positions(cfg, sig, fm)["MSTR"]
    sign = sig.target_sign["MSTR"]
    # within each contiguous non-zero-sign run, |notional| must not change
    episode = (sign != sign.shift()).cumsum()
    for ep, grp in notional[sign != 0].groupby(episode[sign != 0]):
        assert grp.abs().round(2).nunique() == 1


def test_half_life_is_lookahead_free():
    cfg = _cfg()
    returns, fr = _synth()
    fm = fit_rolling_factor_model(cfg, returns, fr)
    hl = half_life_bars(cfg, fm.residuals)

    resid2 = fm.residuals.copy()
    resid2.iloc[400:] = np.nan          # destroy the future
    hl2 = half_life_bars(cfg, resid2)
    pd.testing.assert_series_equal(
        hl["MSTR"].iloc[:400], hl2["MSTR"].iloc[:400], check_names=False)


def test_half_life_filter_reduces_or_equals_trades():
    cfg = _cfg()
    returns, fr = _synth()
    fm = fit_rolling_factor_model(cfg, returns, fr)
    base = generate_signals(cfg, fm.residuals).target_sign.abs().sum().sum()
    cfg.raw["signals"]["max_half_life_bars"] = 10
    filt = generate_signals(cfg, fm.residuals).target_sign.abs().sum().sum()
    assert filt <= base                  # a gate can only remove entries
