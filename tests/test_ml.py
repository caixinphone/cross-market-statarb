"""ML sizing sleeve: point-in-time features, determinism, and output sanity.

Skipped entirely when torch (the optional `ml` extra) is not installed, so the
baseline test suite never depends on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")  # skip the whole module without the ml extra

from src.config import load_config
from src.factors.factor_model import fit_rolling_factor_model
from src.signals.zscore import generate_signals
from src.ml.dataset import build_samples
from src.ml.meta_sizing import walk_forward_sizes


def _synth(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    btc = rng.normal(0, 0.02, n)
    cols = {"BTC": btc}
    for k, a in enumerate(["SOL", "BNB", "LTC"]):
        resid = np.zeros(n)
        for t in range(1, n):
            resid[t] = -0.25 * resid[t - 1] + rng.normal(0, 0.02)
        cols[a] = 1.3 * btc + resid
    returns = pd.DataFrame(cols, index=idx)
    return returns, returns[["BTC"]]


def _cfg():
    cfg = load_config()
    cfg.raw["active_universe"] = "synth"
    cfg.raw["universes"]["synth"] = {"altcoins": ["SOL", "BNB", "LTC"]}
    cfg.raw["factors"] = ["BTC"]
    cfg.raw["factor_map"]["altcoins"] = ["BTC"]
    cfg.raw["factor_overrides"] = {}
    cfg.raw["factor_model"]["rolling_window"] = 60
    cfg.raw["factor_model"]["min_obs"] = 40
    cfg.raw["signals"]["zscore_window"] = 60
    # exercise the full channel set (incl. trend/direction); default conviction sizing
    cfg.raw["ml"].update(
        window=16, hidden=8, n_folds=3, epochs=3, batch_size=32, mom_window=40,
        sizing="conviction", w_min=0.0,
        channels=["resid", "zscore", "spread", "resid_vol", "ret", "factor_ret",
                  "mom", "factor_mom", "resid_drift", "sign"])
    return cfg


def _samples(cfg, returns, fr):
    fm = fit_rolling_factor_model(cfg, returns, fr)
    sig = generate_signals(cfg, fm.residuals)
    s = build_samples(cfg, fm, sig, returns, fr, funding={})
    return fm, sig, s


def test_features_are_point_in_time():
    cfg = _cfg()
    returns, fr = _synth()
    fm, sig, s = _samples(cfg, returns, fr)
    assert len(s.y) > 0
    L = cfg.raw["ml"]["window"]
    mom_w = cfg.raw["ml"]["mom_window"]
    ci_resid = s.channels.index("resid")
    ci_drift = s.channels.index("resid_drift")     # trailing-mean trend channel
    ci_sign = s.channels.index("sign")
    drift = {a: fm.residuals[a].rolling(mom_w).mean() for a in fm.residuals.columns}
    for i in range(min(20, len(s.y))):
        a, t0 = s.asset[i], s.t0[i]
        seg = fm.residuals[a].iloc[t0 - L + 1: t0 + 1].to_numpy()
        assert np.allclose(s.X[i, ci_resid], seg)
        assert s.X[i, ci_resid, -1] == fm.residuals[a].iloc[t0]   # last = t0, not t0+1
        # trend channel is also point-in-time: its last value = trailing mean at t0
        assert np.isclose(s.X[i, ci_drift, -1], drift[a].iloc[t0])
        # sign channel is the constant trade direction
        assert np.all(s.X[i, ci_sign] == s.sign[i])


def test_conviction_sizing_range_and_monotonic():
    cfg = _cfg()
    returns, fr = _synth()
    fm, sig, _ = _samples(cfg, returns, fr)
    ms = walk_forward_sizes(cfg, fm, sig, returns, fr, funding={})
    W = ms.multiplier
    assert W.shape == fm.residuals.shape
    w_min, w_max = cfg.raw["ml"]["w_min"], cfg.raw["ml"]["w_max"]
    assert float(W.min().min()) >= w_min - 1e-9
    assert float(W.max().max()) <= w_max + 1e-9
    # capital ∝ conviction: higher predicted reversion margin -> more capital.
    # (w = clip(pred/σ) is monotonic within a fold; σ varies per fold, so compare
    # top- vs bottom-quartile predictions in aggregate.)
    tt = ms.trades[ms.trades.is_test & ms.trades.y_pred.notna()]
    assert len(tt) > 8
    q = tt.y_pred.quantile([0.25, 0.75])
    hi = tt[tt.y_pred >= q.iloc[1]].w.mean()
    lo = tt[tt.y_pred <= q.iloc[0]].w.mean()
    assert hi > lo                                    # more conviction -> more capital


def test_walk_forward_is_deterministic():
    cfg = _cfg()
    returns, fr = _synth()
    fm, sig, _ = _samples(cfg, returns, fr)
    a = walk_forward_sizes(cfg, fm, sig, returns, fr, funding={})
    b = walk_forward_sizes(cfg, fm, sig, returns, fr, funding={})
    pd.testing.assert_frame_equal(a.multiplier, b.multiplier)
