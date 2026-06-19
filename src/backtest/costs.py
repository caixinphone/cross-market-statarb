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


def _fee_bps_by_instrument(cfg: Config, columns) -> pd.Series:
    """Per-leg taker fee (bps), by execution venue.

    Crypto legs execute as USDⓈ-M **perps** (they pay funding) → ``perp_fee_bps``
    (Binance perp taker 0.04%). Equity legs execute as **spot** (a short borrows
    real shares → borrow cost) → ``spot_fee_bps`` (Binance spot taker 0.10%).
    Splitting this way uses both Binance fee tiers from the brief and keeps each
    leg's cost model self-consistent (perp-fee↔funding, spot-fee↔borrow).
    """
    perp = float(cfg.get("costs", "perp_fee_bps", default=4.0))
    spot = float(cfg.get("costs", "spot_fee_bps", default=10.0))
    return pd.Series(
        {c: (perp if cfg.asset_class_of(c) == "crypto" else spot) for c in columns})


def trading_costs(cfg: Config, held: pd.DataFrame,
                  dollar_volume: pd.DataFrame | None
                  ) -> tuple[pd.Series, pd.Series]:
    """Cost of changing the held book bar-to-bar. Returns (cost, traded_notional).

    Fee = per-leg taker bps on traded notional (perp for crypto, spot for equity;
    see :func:`_fee_bps_by_instrument`). Slippage is either flat (``slippage_bps``)
    or a liquidity model: ``base + impact·sqrt(participation)`` per instrument,
    where ``participation = trade_notional / ADV_bar`` (square-root market-impact
    law). The liquidity model makes thin names (e.g. small miners) cost more to
    trade than mega-caps, as in reality.
    """
    fee_bps = _fee_bps_by_instrument(cfg, held.columns)
    traded = held.diff().abs()
    traded.iloc[0] = held.iloc[0].abs()      # initial entry from flat
    traded_notional = traded.sum(axis=1)
    fee = (traded * fee_bps / 1e4).sum(axis=1)

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


def per_bar_funding_frame(cfg: Config, funding: dict[str, pd.Series],
                          index: pd.DatetimeIndex,
                          instruments: list[str]) -> pd.DataFrame:
    """Per-bar funding *rate* per instrument, aligned to the panel index.

    Each bar accrues every 8h settlement that falls in its lookback interval
    ``(previous_bar, this_bar]`` — i.e. the funding paid on the crypto held since
    the last bar. This is frequency-general:

    * Daily bars absorb the day's ~3 settlements (00:00/08:00/16:00 UTC).
    * Hourly RTH bars: the 16:00 settlement lands on the 16:00 bar; the overnight
      00:00/08:00 settlements accrue onto the first RTH bar of the next session
      (14:00), exactly as a 24/7-held position pays funding across the gap.

    Settlements are bucketed by ``searchsorted`` into the bar at/after each one.
    Missing series fall back to a constant scaled by the 8h-periods each bar
    covers (``fallback_8h × hours_covered / 8``) — an explicit, documented
    assumption (used only when a crypto funding file is absent).
    """
    fallback_8h = float(cfg.get("costs", "funding_8h_fallback_bps", default=1.0)) / 1e4

    # Hours each bar covers = gap to the previous bar (first bar: one period).
    if len(index) > 1:
        gap_h = index.to_series().diff().dt.total_seconds().to_numpy() / 3600.0
        gap_h[0] = gap_h[1] if len(gap_h) > 1 else 8.0
    else:
        gap_h = np.array([8.0])
    per_bar_fallback = pd.Series(fallback_8h * gap_h / 8.0, index=index)
    # Missing series get the fallback; real series get 0 between settlements
    # (funding only accrues at discrete 8h boundaries) plus the bucketed rate.
    out = pd.DataFrame({c: per_bar_fallback for c in instruments})

    for name, s in funding.items():
        if name not in out.columns or s.empty:
            continue
        st = s.copy()
        st.index = st.index.tz_convert("UTC") if st.index.tz else st.index.tz_localize("UTC")
        st = st[(st.index > index[0]) & (st.index <= index[-1])]
        col = pd.Series(0.0, index=index)
        if not st.empty:
            # Bucket each settlement into the first panel bar at/after it.
            pos = index.searchsorted(st.index, side="left")
            bucketed = pd.Series(st.to_numpy()).groupby(pos).sum()
            col.iloc[bucketed.index.to_numpy()] = bucketed.to_numpy()
        out[name] = col
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
