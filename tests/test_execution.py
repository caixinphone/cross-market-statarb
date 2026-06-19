"""Tests for the realistic execution layer and entry sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.portfolio.execution import apply_no_trade_band


def _idx(n):
    return pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")


def test_no_trade_band_holds_small_drifts_but_trades_large():
    cfg = load_config()
    cfg.raw["portfolio"]["aum"] = 10_000_000
    cfg.raw["portfolio"]["no_trade_band"] = 0.001       # $10k band
    idx = _idx(5)
    # target drifts by $5k (<band) then jumps $100k (>band) then to 0 (exit)
    target = pd.DataFrame({"A": [100_000, 105_000, 104_000, 5_000, 0.0]}, index=idx)
    actual = apply_no_trade_band(cfg, target)
    held = actual["A"].tolist()
    assert held[0] == 100_000          # initial entry (>band from 0)
    assert held[1] == 100_000          # +5k drift < band -> hold
    assert held[2] == 100_000          # still within band of last trade
    assert held[3] == 5_000            # -95k from held -> trade
    assert held[4] == 5_000            # -5k < band -> hold (no exit yet)


def test_no_trade_band_off_when_zero():
    cfg = load_config()
    cfg.raw["portfolio"]["no_trade_band"] = 0.0
    idx = _idx(3)
    target = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    pd.testing.assert_frame_equal(apply_no_trade_band(cfg, target), target)
