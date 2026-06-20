"""Per-trade training samples for the CNN sizer (no lookahead in the features).

One sample = one *holding episode* produced by the layer-1 mean-reversion state
machine, for one asset:

* **features** ``X`` — a multi-channel 1D window of length ``L`` ending at the
  entry bar ``t0`` (decision time): only data ``≤ t0`` is used, so the live sizing
  decision is point-in-time safe.
* **label** ``y`` — the trade's realised **net margin per unit notional**::

      gross = sign · Σ_{held bars} ε            # idiosyncratic reversion edge
      cost  = round-trip fee+slippage + Σ funding + Σ borrow   (per $)
      y     = gross − cost

The label uses the trade's future outcome — fine for *training*, but the model is
only ever applied walk-forward (trained on trades that closed strictly before the
test window), so no future information reaches a live decision. See
:mod:`src.ml.meta_sizing`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..backtest.costs import per_bar_funding_frame
from ..portfolio.construct import periods_per_year, rolling_resid_vol
from ..factors.factor_model import FactorModelResult
from ..signals.zscore import SignalResult

CHANNELS = ["resid", "zscore", "spread", "resid_vol", "ret", "factor_ret",
            "mom", "factor_mom", "resid_drift", "sign"]
# Trend/direction channels (mom/factor_mom/resid_drift/sign) let the CNN *see* a
# bull regime and the trade direction, so it can learn to veto shorts into
# persistent up-trends — the −$547k alpha-drift drag. They are point-in-time
# (trailing) and computed only from data ≤ the entry bar.


@dataclass
class TradeSamples:
    X: np.ndarray              # (n_samples, n_channels, L)
    y: np.ndarray              # (n_samples,) net margin per $ notional
    asset: list[str]           # asset per sample
    t0: np.ndarray             # entry bar (iloc) per sample
    t1: np.ndarray             # last signal bar (iloc) per sample
    sign: np.ndarray           # +1 long / -1 short
    channels: list[str]


def _episodes(sign: pd.Series) -> list[tuple[int, int, int]]:
    """Return (start_iloc, end_iloc, sign) for each non-zero contiguous run."""
    s = sign.to_numpy()
    out = []
    i, n = 0, len(s)
    while i < n:
        if s[i] != 0:
            j = i
            while j + 1 < n and s[j + 1] == s[i]:
                j += 1
            out.append((i, j, int(np.sign(s[i]))))
            i = j + 1
        else:
            i += 1
    return out


def build_samples(cfg: Config, fm: FactorModelResult, signals: SignalResult,
                  returns: pd.DataFrame, factor_returns: pd.DataFrame,
                  funding: dict[str, pd.Series]) -> TradeSamples:
    L = int(cfg.get("ml", "window", default=64))
    channels = list(cfg.get("ml", "channels", default=CHANNELS))
    mom_w = int(cfg.get("ml", "mom_window", default=120))
    ppy = periods_per_year(cfg)

    resid = fm.residuals
    spread = resid.cumsum()
    z = signals.zscore
    rvol = rolling_resid_vol(cfg, resid)
    # Trailing trend/regime context (point-in-time):
    asset_mom = returns.rolling(mom_w).sum()                  # asset momentum
    factor_mom = factor_returns.rolling(mom_w).sum()          # market/factor trend
    resid_drift = resid.rolling(mom_w).mean()                 # idiosyncratic drift
    idx = resid.index

    # Round-trip fee+slippage per $ by venue (slippage ≈ floor at small size).
    perp = float(cfg.get("costs", "perp_fee_bps", default=4.0))
    spot = float(cfg.get("costs", "spot_fee_bps", default=10.0))
    slip_base = float(cfg.get("costs", "slippage_base_bps", default=1.0))
    borrow_bar = float(cfg.get("costs", "borrow_annual_pct", default=3.0)) / 100.0 / ppy

    crypto = [a for a in fm.residuals.columns if cfg.asset_class_of(a) == "crypto"]
    frate = per_bar_funding_frame(cfg, funding, idx, crypto) if crypto else None

    X, y, a_list, t0s, t1s, signs = [], [], [], [], [], []
    n = len(idx)
    for a in signals.target_sign.columns:
        if a not in resid.columns:
            continue
        factors = fm.factor_map.get(a, [])
        f0 = factors[0] if factors and factors[0] in factor_returns.columns else None
        chan_series = {
            "resid": resid[a], "zscore": z[a], "spread": spread[a],
            "resid_vol": rvol[a], "ret": returns[a] if a in returns else None,
            "factor_ret": factor_returns[f0] if f0 else None,
            "mom": asset_mom[a] if a in asset_mom else None,
            "factor_mom": factor_mom[f0] if f0 else None,
            "resid_drift": resid_drift[a],
            # "sign" is per-trade (constant over the window) -> filled in the loop.
        }
        is_crypto = cfg.asset_class_of(a) == "crypto"
        # round-trip (entry+exit) fee + floor slippage per $, by venue
        fee_rt = 2.0 * ((perp if is_crypto else spot) + slip_base) / 1e4

        eps = resid[a].to_numpy()
        for t0, t1, sgn in _episodes(signals.target_sign[a]):
            if t0 < L - 1 or t1 + 1 >= n:
                continue
            # ---- features: window ending at t0 (decision bar) ----
            win = np.empty((len(channels), L), dtype=np.float64)
            ok = True
            for ci, ch in enumerate(channels):
                if ch == "sign":                       # constant trade-direction channel
                    win[ci] = float(sgn)
                    continue
                ser = chan_series.get(ch)
                if ser is None:
                    ok = False
                    break
                seg = ser.iloc[t0 - L + 1: t0 + 1].to_numpy()
                if len(seg) != L or not np.isfinite(seg).all():
                    ok = False
                    break
                win[ci] = seg
            if not ok:
                continue
            # ---- label: realised net margin per $ over the held bars ----
            held = slice(t0 + 1, t1 + 2)              # held = sign.shift(1) region
            gross = sgn * np.nansum(eps[held])
            cost = fee_rt
            if is_crypto and frate is not None:
                cost += sgn * float(frate[a].iloc[held].sum())   # long pays + funding
            if (not is_crypto) and sgn < 0:
                cost += borrow_bar * (t1 + 1 - (t0 + 1) + 1)     # short equity borrow
            if not np.isfinite(gross):
                continue
            X.append(win)
            y.append(float(gross - cost))
            a_list.append(a)
            t0s.append(t0)
            t1s.append(t1)
            signs.append(sgn)

    return TradeSamples(
        X=np.asarray(X), y=np.asarray(y), asset=a_list,
        t0=np.asarray(t0s), t1=np.asarray(t1s), sign=np.asarray(signs),
        channels=channels,
    )
