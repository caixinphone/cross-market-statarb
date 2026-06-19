"""Transaction-cost, funding, and borrow models (Task 4).

* Trading cost — taker fee + slippage on traded notional at each rebalance.
* Funding — USDⓈ-M perps settle every 8h; longs pay when the rate is positive.
  Crypto legs use the downloaded funding series; equity perps (and any missing
  crypto months) fall back to a configured constant — an explicit assumption.
* Borrow — short *equity* legs accrue an annualised borrow fee per bar.

All functions return a per-bar **cost** series (positive = drag on PnL); funding
on a short leg is naturally negative (a rebate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config

ADV_WINDOW = 20          # bars used to estimate average traded $-volume
MAX_PARTICIPATION = 1.0  # cap a single trade at 100% of a bar's ADV for impact


def trading_costs(cfg: Config, held: pd.DataFrame,
                  dollar_volume: pd.DataFrame | None
                  ) -> tuple[pd.Series, pd.Series]:
    """Cost of changing the held book bar-to-bar. Returns (cost, traded_notional).

    Fee = taker bps on traded notional. Slippage is either flat (``slippage_bps``)
    or a liquidity model: ``base + impact·sqrt(participation)`` per instrument,
    where ``participation = trade_notional / ADV_bar`` (square-root market-impact
    law). The liquidity model makes thin names (e.g. small miners) cost more to
    trade than mega-caps, as in reality.
    """
    fee_bps = float(cfg.get("costs", "perp_fee_bps", default=4.0))
    traded = held.diff().abs()
    traded.iloc[0] = held.iloc[0].abs()      # initial entry from flat
    traded_notional = traded.sum(axis=1)
    fee = traded_notional * fee_bps / 1e4

    model = cfg.get("costs", "slippage_model", default="liquidity")
    if model == "flat" or dollar_volume is None:
        slip_bps = float(cfg.get("costs", "slippage_bps", default=5.0))
        return fee + traded_notional * slip_bps / 1e4, traded_notional

    base = float(cfg.get("costs", "slippage_base_bps", default=1.0))
    impact = float(cfg.get("costs", "slippage_impact_bps", default=12.0))
    adv = dollar_volume.reindex(held.index).reindex(columns=held.columns)
    adv = adv.rolling(ADV_WINDOW, min_periods=5).mean().replace(0.0, np.nan)
    participation = (traded / adv).clip(upper=MAX_PARTICIPATION)
    slip_bps = base + impact * np.sqrt(participation)
    slip_bps = slip_bps.fillna(base + impact)   # unknown ADV -> conservative
    slip = (traded * slip_bps / 1e4).sum(axis=1)
    return fee + slip, traded_notional


def daily_funding_frame(cfg: Config, funding: dict[str, pd.Series],
                        index: pd.DatetimeIndex,
                        instruments: list[str]) -> pd.DataFrame:
    """Per-bar funding *rate* per instrument, aligned to the panel index.

    Crypto: sum the 8h settlement rates within each bar; equity / missing:
    fall back to the configured constant (×3 settlements for a daily bar).
    """
    fallback_8h = float(cfg.get("costs", "funding_8h_fallback_bps", default=1.0)) / 1e4
    per_bar_fallback = fallback_8h * 3 if cfg.frequency == "daily" else fallback_8h
    out = pd.DataFrame(per_bar_fallback, index=index, columns=instruments)

    for name, s in funding.items():
        if name not in out.columns or s.empty:
            continue
        daily = s.groupby(s.index.normalize()).sum()
        daily.index = daily.index.tz_convert("UTC") if daily.index.tz else daily.index
        out[name] = daily.reindex(index).fillna(per_bar_fallback)
    return out


def funding_costs(held: pd.DataFrame, funding_rate: pd.DataFrame,
                  crypto_instruments: list[str]) -> pd.Series:
    cols = [c for c in crypto_instruments if c in held.columns]
    if not cols:
        return pd.Series(0.0, index=held.index)
    # long (held>0) pays a positive rate -> positive cost; short receives.
    return (held[cols] * funding_rate[cols]).sum(axis=1)


def borrow_costs(cfg: Config, held: pd.DataFrame, periods_per_year: float,
                 equity_instruments: list[str]) -> pd.Series:
    cols = [c for c in equity_instruments if c in held.columns]
    if not cols:
        return pd.Series(0.0, index=held.index)
    per_bar = float(cfg.get("costs", "borrow_annual_pct", default=3.0)) / 100.0 \
        / periods_per_year
    short_notional = held[cols].clip(upper=0.0).abs().sum(axis=1)
    return short_notional * per_bar
