"""Cost accounting must match hand-computed values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.backtest.costs import borrow_costs, trading_costs


def _idx(n):
    return pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")


def test_trading_costs_charge_on_change_only():
    cfg = load_config()
    cfg.raw["costs"]["slippage_model"] = "flat"
    cfg.raw["costs"]["perp_fee_bps"] = 4.0
    cfg.raw["costs"]["slippage_bps"] = 5.0
    idx = _idx(3)
    # entry 100, hold, exit to 0
    held = pd.DataFrame({"ETH": [100.0, 100.0, 0.0]}, index=idx)
    cost, traded = trading_costs(cfg, held, dollar_volume=None)
    rate = 9e-4   # crypto perp 4bps + flat slippage 5bps
    assert np.isclose(traded.iloc[0], 100.0)   # initial entry
    assert np.isclose(traded.iloc[1], 0.0)     # no change -> no cost
    assert np.isclose(traded.iloc[2], 100.0)   # exit trades 100
    assert np.isclose(cost.iloc[0], 100.0 * rate)
    assert np.isclose(cost.iloc[2], 100.0 * rate)


def test_fee_tier_splits_crypto_perp_vs_equity_spot():
    cfg = load_config()
    cfg.raw["costs"]["slippage_model"] = "flat"
    cfg.raw["costs"]["perp_fee_bps"] = 4.0
    cfg.raw["costs"]["spot_fee_bps"] = 10.0
    cfg.raw["costs"]["slippage_bps"] = 0.0
    idx = _idx(1)
    held = pd.DataFrame({"ETH": [100.0], "MSTR": [100.0]}, index=idx)
    cost, _ = trading_costs(cfg, held, dollar_volume=None)
    # ETH (crypto -> perp 4bps) + MSTR (equity -> spot 10bps) = 0.04 + 0.10
    assert np.isclose(cost.iloc[0], 100.0 * 4e-4 + 100.0 * 1e-3)


def test_liquidity_slippage_scales_with_participation():
    cfg = load_config()
    cfg.raw["costs"].update({"slippage_model": "liquidity", "perp_fee_bps": 0.0,
                             "spot_fee_bps": 0.0, "slippage_base_bps": 1.0,
                             "slippage_impact_bps": 12.0})
    idx = _idx(8)
    held = pd.DataFrame({"A": [0, 0, 0, 0, 0, 0, 100_000.0, 100_000.0]}, index=idx)
    # thin vs deep ADV -> same trade, different slippage
    thin = pd.DataFrame({"A": [1e6] * 8}, index=idx)     # 10% participation
    deep = pd.DataFrame({"A": [1e9] * 8}, index=idx)     # 0.01% participation
    c_thin, _ = trading_costs(cfg, held, thin)
    c_deep, _ = trading_costs(cfg, held, deep)
    assert c_thin.iloc[6] > c_deep.iloc[6]               # thinner = costlier
    # deep book slippage ~ base (1bp) on 100k = ~$10
    assert 5 < c_deep.iloc[6] < 30


def test_borrow_only_on_short_equity():
    cfg = load_config()
    cfg.raw["costs"]["borrow_annual_pct"] = 3.0
    idx = _idx(2)
    # one short (-100) and one long (+100) equity leg
    held = pd.DataFrame({"MSTR": [-100.0, -100.0], "NVDA": [100.0, 100.0]},
                        index=idx)
    cost = borrow_costs(cfg, held, periods_per_year=252.0,
                        equity_instruments=["MSTR", "NVDA"])
    expected = 100.0 * (0.03 / 252.0)   # only the short pays
    assert np.allclose(cost.values, expected)
