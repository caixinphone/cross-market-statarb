"""Residual z-score and the entry/exit state machine.

Two signal flavours (config ``signals.mode``):

* ``resid_spread`` (default): z-score of the *cumulative* residual
  ``S_t = Σ_{τ≤t} ε_τ`` — i.e. how rich/cheap the asset's price is relative to
  what its factors predict. This matches the strategy's economic narrative and
  the Avellaneda-Lee residual-spread setup. The trailing rolling mean detrends
  the (near-random-walk) spread locally.
* ``resid_return``: z-score of the daily residual return ``ε_t`` — the literal
  formula in the brief; a short-horizon reversal signal.

Mean-reversion convention: a *high* z (rich) → SHORT the asset; a *low* z (cheap)
→ LONG. So the target sign is ``-sign(z)``. Entry at ``|z| > entry_threshold``,
exit at ``|z| < exit_threshold`` (hysteresis), with a sign-flip and a
max-holding forced exit. Signals are formed from data ≤ t; the engine executes
them on ``t+1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config


@dataclass
class SignalResult:
    zscore: pd.DataFrame        # index=bars, cols=assets
    target_sign: pd.DataFrame   # {-1, 0, +1} desired position on the asset


def _signal_series(cfg: Config, residuals: pd.DataFrame) -> pd.DataFrame:
    mode = cfg.get("signals", "mode", default="resid_spread")
    if mode == "resid_spread":
        return residuals.cumsum()      # skipna keeps leading NaNs
    if mode == "resid_return":
        return residuals
    raise ValueError(f"unknown signals.mode '{mode}'")


def compute_zscore(cfg: Config, residuals: pd.DataFrame) -> pd.DataFrame:
    window = int(cfg.get("signals", "zscore_window", default=60))
    signal_series = _signal_series(cfg, residuals)
    mean = signal_series.rolling(window, min_periods=window).mean()
    std = signal_series.rolling(window, min_periods=window).std()
    z = (signal_series - mean) / std.replace(0.0, np.nan)
    return z


def half_life_bars(cfg: Config, residuals: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean-reversion half-life (bars) of the signal series via an AR(1)
    fit S_t = a + b·S_{t-1} (Avellaneda-Lee mean-reversion-speed criterion).

    half-life = ln2 / κ with κ = -ln(b). b≥1 → no reversion (∞); b≤0 → fast (1).
    Point-in-time: the window ends at t. Used to gate out slow/non-reverting
    residuals (where idiosyncratic drift, not reversion, dominates).
    """
    window = int(cfg.get("signals", "zscore_window", default=60))
    s = _signal_series(cfg, residuals)
    x, y = s.shift(1), s
    mx = x.rolling(window, min_periods=window).mean()
    my = y.rolling(window, min_periods=window).mean()
    cov = (x * y).rolling(window, min_periods=window).mean() - mx * my
    var = (x * x).rolling(window, min_periods=window).mean() - mx * mx
    b = cov / var.replace(0.0, np.nan)
    hl = np.log(2.0) / (-np.log(b.clip(lower=1e-6, upper=0.999999)))
    hl = hl.where(b > 0, 1.0)        # b<=0: oscillating -> reverts within a bar
    hl = hl.where(b < 1.0, np.inf)   # b>=1: random walk / diverging -> no reversion
    return hl


def _state_machine(z: np.ndarray, entry: float, exit_: float, max_hold: int,
                   can_enter: np.ndarray | None = None) -> np.ndarray:
    """Per-asset hysteresis state machine → target sign in {-1,0,+1}.

    Enter (fade) when flat and |z|>entry (and ``can_enter`` if given). Exit to
    flat when |z|<exit, the sign flips, or holding exceeds max_hold. Exits always
    use the real z, so the entry gate (e.g. a half-life filter) never traps an
    open position.
    """
    n = len(z)
    sign = np.zeros(n)
    state = 0
    held = 0
    for t in range(n):
        zt = z[t]
        if np.isnan(zt):
            sign[t] = state  # carry; gaps handled by engine via prices
            continue
        gate = True if can_enter is None else bool(can_enter[t])
        if state == 0:
            if gate and zt > entry:
                state, held = -1, 0          # rich -> short
            elif gate and zt < -entry:
                state, held = +1, 0          # cheap -> long
        else:
            held += 1
            reverted = abs(zt) < exit_
            flipped = (state == -1 and zt < -entry) or (state == +1 and zt > entry)
            if reverted or held >= max_hold:
                state = 0
            elif flipped:
                state = -state if gate else 0   # re-enter opposite only if allowed
                held = 0
        sign[t] = state
    return sign


def generate_signals(cfg: Config, residuals: pd.DataFrame) -> SignalResult:
    entry = float(cfg.get("signals", "entry_threshold", default=2.0))
    exit_ = float(cfg.get("signals", "exit_threshold", default=0.5))
    max_hold = int(cfg.get("signals", "max_holding_bars", default=20))
    max_hl = cfg.get("signals", "max_half_life_bars", default=None)

    z = compute_zscore(cfg, residuals)
    gate = None
    if max_hl is not None:
        # Mean-reversion-speed filter: only OPEN positions on fast-reverting names.
        gate = (half_life_bars(cfg, residuals) <= float(max_hl)).fillna(False)

    target = pd.DataFrame(
        {c: _state_machine(z[c].to_numpy(), entry, exit_, max_hold,
                           None if gate is None else gate[c].to_numpy())
         for c in z.columns},
        index=z.index,
    )
    return SignalResult(zscore=z, target_sign=target)
