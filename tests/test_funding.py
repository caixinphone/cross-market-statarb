"""Per-bar funding accrual (hourly track).

The hourly fix: real 8h funding settlements must actually be used (not silently
dropped onto a constant fallback), and overnight settlements must accrue onto the
first RTH bar of the next session — the funding a 24/7-held crypto leg pays across
the gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.backtest.costs import funding_costs, per_bar_funding_frame


def _rth_hourly_index():
    """Two RTH sessions of 7 hourly bars (14:00-20:00 UTC)."""
    day1 = pd.date_range("2023-01-03 14:00", "2023-01-03 20:00", freq="1h", tz="UTC")
    day2 = pd.date_range("2023-01-04 14:00", "2023-01-04 20:00", freq="1h", tz="UTC")
    return day1.union(day2)


def _funding_series():
    # 8h settlements at 00:00/08:00/16:00 UTC, 1bp each (positive).
    settle = pd.to_datetime([
        "2023-01-03 00:00", "2023-01-03 08:00", "2023-01-03 16:00",
        "2023-01-04 00:00", "2023-01-04 08:00", "2023-01-04 16:00",
    ], utc=True)
    return {"ETH": pd.Series([1e-4] * len(settle), index=settle)}


def test_real_series_is_used_and_overnight_accrues():
    cfg = load_config()
    idx = _rth_hourly_index()
    frame = per_bar_funding_frame(cfg, _funding_series(), idx, ["ETH", "BTC"])

    # (a) the 16:00 settlement lands on the 16:00 RTH bar (real rate, not fallback)
    assert np.isclose(frame.loc["2023-01-03 16:00+00:00", "ETH"], 1e-4)
    # (b) overnight 00:00 + 08:00 settlements accrue onto the first RTH bar (14:00)
    assert np.isclose(frame.loc["2023-01-04 14:00+00:00", "ETH"], 2e-4)
    # (c) a bar with no settlement carries zero for a real series
    assert np.isclose(frame.loc["2023-01-03 15:00+00:00", "ETH"], 0.0)
    # settlements before the first bar are not charged (no prior holding)
    assert np.isclose(frame.loc["2023-01-03 14:00+00:00", "ETH"], 0.0)


def test_missing_series_falls_back_per_8h_periods():
    cfg = load_config()
    fallback_8h = float(cfg.get("costs", "funding_8h_fallback_bps")) / 1e4
    idx = _rth_hourly_index()
    frame = per_bar_funding_frame(cfg, _funding_series(), idx, ["ETH", "BTC"])

    # BTC has no funding file -> fallback scaled by the 8h periods each bar covers.
    # A 1h intraday bar covers 1/8 of an 8h period.
    assert np.isclose(frame.loc["2023-01-03 15:00+00:00", "BTC"], fallback_8h * 1 / 8)
    # The overnight gap (20:00 -> next 14:00 = 18h) accrues ~18/8 periods.
    assert np.isclose(frame.loc["2023-01-04 14:00+00:00", "BTC"], fallback_8h * 18 / 8)


def test_short_positive_funding_is_a_rebate():
    cfg = load_config()
    idx = _rth_hourly_index()
    frate = per_bar_funding_frame(cfg, _funding_series(), idx, ["ETH"])
    held = pd.DataFrame(0.0, index=idx, columns=["ETH"])
    held.loc["2023-01-03 16:00+00:00", "ETH"] = -1_000_000.0   # short the perp

    cost = funding_costs(held, frate, ["ETH"])
    # short a positive-funding perp -> negative cost (you receive funding)
    assert cost.loc["2023-01-03 16:00+00:00"] < 0
    assert np.isclose(cost.loc["2023-01-03 16:00+00:00"], -1_000_000.0 * 1e-4)
