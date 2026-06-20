"""Walk-forward training of the CNN sizer → per-bar size multiplier.

No lookahead: for each test fold, the CNN is trained only on trades that **closed
strictly before** the fold's first entry (expanding window), so a live sizing
decision at ``t0`` never depends on future trades. The first fold has no prior
data and falls back to baseline sizing (multiplier = 1).

The output multiplier frame ``W`` (bars × assets) is applied to the baseline
equal-volatility notionals: ``N'_i,t = W_i,t · N_i,t`` (constant over a trade, so
size stays frozen at entry as in the baseline). Predicted-loss trades → 0
(vetoed); predicted-gain trades → scaled up to ``w_max``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..factors.factor_model import FactorModelResult
from ..signals.zscore import SignalResult
from .cnn import Standardizer, predict, train_model
from .dataset import TradeSamples, build_samples


@dataclass
class MetaSizingResult:
    multiplier: pd.DataFrame       # bars × assets, the size multiplier W
    trades: pd.DataFrame           # per-trade diagnostics
    importance: pd.Series          # permutation importance per channel (OOS)
    oos_corr: float                # corr(predicted, realised) on test trades


def _fold_bounds(t0: np.ndarray, n_folds: int) -> list[float]:
    qs = np.linspace(0.0, 1.0, n_folds + 1)
    return list(np.quantile(t0, qs))


def _permutation_importance(model, X, y, channels, seed) -> pd.Series:
    """Increase in MSE when each channel is shuffled across samples (OOS)."""
    rng = np.random.default_rng(seed)
    base = float(np.mean((predict(model, X) - y) ** 2))
    imp = {}
    for ci, ch in enumerate(channels):
        Xp = X.copy()
        Xp[:, ci, :] = Xp[rng.permutation(len(X)), ci, :]
        imp[ch] = float(np.mean((predict(model, Xp) - y) ** 2)) - base
    return pd.Series(imp).sort_values(ascending=False)


def walk_forward_sizes(cfg: Config, fm: FactorModelResult, signals: SignalResult,
                       returns: pd.DataFrame, factor_returns: pd.DataFrame,
                       funding: dict) -> MetaSizingResult:
    n_folds = int(cfg.get("ml", "n_folds", default=5))
    w_max = float(cfg.get("ml", "w_max", default=3.0))
    seed = int(cfg.get("ml", "seed", default=42))
    sizing = str(cfg.get("ml", "sizing", default="conviction"))
    w_min = float(cfg.get("ml", "w_min", default=0.0))
    conviction_gain = float(cfg.get("ml", "conviction_gain", default=1.0))
    tilt = float(cfg.get("ml", "tilt", default=0.5))
    floor_long = float(cfg.get("ml", "floor_long", default=0.5))
    floor_short = float(cfg.get("ml", "floor_short", default=0.0))
    short_veto_pct = float(cfg.get("ml", "short_veto_pct", default=0.34))
    hp = dict(hidden=int(cfg.get("ml", "hidden", default=16)),
              epochs=int(cfg.get("ml", "epochs", default=40)),
              lr=float(cfg.get("ml", "lr", default=1e-3)),
              batch_size=int(cfg.get("ml", "batch_size", default=64)),
              dropout=float(cfg.get("ml", "dropout", default=0.1)),
              weight_decay=float(cfg.get("ml", "weight_decay", default=1e-4)),
              val_frac=float(cfg.get("ml", "val_frac", default=0.2)),
              patience=int(cfg.get("ml", "patience", default=5)),
              seed=seed)

    s: TradeSamples = build_samples(cfg, fm, signals, returns, factor_returns, funding)
    if len(s.y) < 50:
        raise RuntimeError(f"too few trades for ML ({len(s.y)}); widen the universe/period")

    bounds = _fold_bounds(s.t0, n_folds)
    w = np.ones(len(s.y))                 # default 1 (warmup / no model)
    is_test = np.zeros(len(s.y), dtype=bool)
    y_pred = np.full(len(s.y), np.nan)
    last = None                            # (model, Xtest_std, ytest) for importance

    for k in range(1, n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        test_mask = (s.t0 >= lo) & (s.t0 < (hi if k < n_folds - 1 else hi + 1))
        train_mask = s.t1 < lo             # closed strictly before the fold opens
        if train_mask.sum() < 30 or test_mask.sum() == 0:
            continue
        std = Standardizer(s.X[train_mask])
        model = train_model(std(s.X[train_mask]), s.y[train_mask], **hp)
        Xte = std(s.X[test_mask])
        pred = predict(model, Xte)
        y_pred[test_mask] = pred
        is_test[test_mask] = True
        mu = float(np.mean(s.y[train_mask]))
        sd = max(float(np.std(s.y[train_mask])), 1e-4)
        if sizing == "conviction":
            # Capital ∝ predicted reversion conviction, *relative to peers*:
            # w = 1 + gain·(pred−μ)/σ, floored at w_min, capped at w_max. Trades with
            # above-average predicted reversion get more capital, below-average less.
            # Centred (not gated at pred>0) so an overall net-negative book — where
            # most predictions are negative — still allocates by relative conviction.
            w_te = np.clip(1.0 + conviction_gain * (pred - mu) / sd, w_min, w_max)
        else:
            # Legacy: gentle tilt centred on 1.0 with an asymmetric short floor /
            # rank-based short veto (kept for comparison; not the default).
            sgn_te = s.sign[test_mask]
            floors = np.where(sgn_te < 0, floor_short, floor_long)
            w_te = np.clip(1.0 + tilt * (pred - mu) / sd, floors, w_max)
            short = sgn_te < 0
            if short.sum() > 3:
                thr = np.quantile(pred[short], short_veto_pct)
                w_te[short & (pred <= thr)] = 0.0
        w[test_mask] = w_te
        last = (model, Xte, s.y[test_mask])

    # ---- build per-bar multiplier frame on the signal grid ----
    W = pd.DataFrame(1.0, index=fm.residuals.index, columns=signals.target_sign.columns)
    for i in range(len(s.y)):
        W.iloc[s.t0[i]: s.t1[i] + 1, W.columns.get_loc(s.asset[i])] = w[i]

    trades = pd.DataFrame({
        "asset": s.asset, "t0": s.t0, "t1": s.t1, "sign": s.sign,
        "y_true": s.y, "y_pred": y_pred, "w": w, "is_test": is_test,
    })
    tt = trades[trades.is_test & trades.y_pred.notna()]
    oos_corr = float(np.corrcoef(tt.y_true, tt.y_pred)[0, 1]) if len(tt) > 2 else float("nan")
    importance = (_permutation_importance(*last, s.channels, seed)
                  if last is not None else pd.Series(dtype=float))

    return MetaSizingResult(multiplier=W, trades=trades,
                            importance=importance, oos_corr=oos_corr)
