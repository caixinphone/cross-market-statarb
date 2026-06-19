"""Realistic execution layer: no-trade band.

Compliant *target* positions still jitter bar-to-bar (factor-hedge legs drift with
the rolling betas, constraint-scaling factors move as the signal set changes). A
desk does not chase every tiny delta — it rebalances only when a position drifts
beyond a tolerance. :func:`apply_no_trade_band` turns targets into the *actually
held* book by holding each instrument until its target moves more than
``no_trade_band`` of AUM, then snapping to the new target.

This both reflects reality and removes the last big chunk of unrealistic turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def apply_no_trade_band(cfg: Config, positions: pd.DataFrame) -> pd.DataFrame:
    aum = float(cfg.get("portfolio", "aum", default=10_000_000))
    band = float(cfg.get("portfolio", "no_trade_band", default=0.0015)) * aum
    if band <= 0:
        return positions

    vals = positions.to_numpy(dtype="float64")
    held = np.zeros(vals.shape[1])
    out = np.empty_like(vals)
    for t in range(vals.shape[0]):
        target = vals[t]
        # rebalance an instrument only if it drifted beyond the band; a target of
        # ~0 (signal exit) always exceeds the band for a real position, so exits
        # and entries still fire.
        move = np.abs(target - held) > band
        held = np.where(move, target, held)
        out[t] = held
    return pd.DataFrame(out, index=positions.index, columns=positions.columns)
