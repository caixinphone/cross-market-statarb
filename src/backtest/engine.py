"""Vectorised, point-in-time backtest engine.

The single most important line is ``held = positions.shift(1)``: a target formed
from information at bar ``t`` is executed and starts earning on bar ``t+1``. This
one-bar lag is what makes the whole pipeline lookahead-free, regardless of the
(in-sample) factor windows upstream.

PnL per bar = Σ_j held_j · return_j − trading_cost − funding − borrow.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import Config
from ..portfolio.construct import periods_per_year
from .costs import (borrow_costs, funding_costs, per_bar_funding_frame,
                    trading_costs)


@dataclass
class BacktestResult:
    equity: pd.Series              # account equity ($)
    returns: pd.Series             # per-bar portfolio return
    pnl: pd.DataFrame              # gross / fees / funding / borrow / net ($)
    held: pd.DataFrame             # positions held each bar (executed, lagged)
    turnover: pd.Series            # traded notional / AUM per bar

    def save(self, processed_dir) -> None:
        self.equity.to_frame("equity").to_parquet(processed_dir / "equity.parquet")
        self.pnl.to_parquet(processed_dir / "pnl.parquet")


def run_backtest(
    cfg: Config,
    returns: pd.DataFrame,
    positions: pd.DataFrame,
    funding: dict[str, pd.Series],
    dollar_volume: pd.DataFrame | None = None,
) -> BacktestResult:
    aum = float(cfg.get("portfolio", "aum", default=10_000_000))
    instruments = list(positions.columns)
    rets = returns.reindex(columns=instruments).fillna(0.0)

    # Execute next bar: positions decided at t are held through the t+1 return.
    held = positions.shift(1).fillna(0.0)
    gross_pnl = (held * rets).sum(axis=1)

    # Trading costs (charged when the held book changes), liquidity-aware.
    fee_pnl, traded_notional = trading_costs(cfg, held, dollar_volume)

    crypto = [i for i in instruments if cfg.asset_class_of(i) == "crypto"]
    equity = [i for i in instruments if cfg.asset_class_of(i) == "equity"]

    if cfg.get("costs", "apply_funding", default=True):
        frate = per_bar_funding_frame(cfg, funding, returns.index, instruments)
        fund_pnl = funding_costs(held, frate, crypto)
    else:
        fund_pnl = pd.Series(0.0, index=returns.index)

    if cfg.get("costs", "apply_borrow", default=True):
        borrow_pnl = borrow_costs(cfg, held, periods_per_year(cfg), equity)
    else:
        borrow_pnl = pd.Series(0.0, index=returns.index)

    net_pnl = gross_pnl - fee_pnl - fund_pnl - borrow_pnl
    equity_curve = aum + net_pnl.cumsum()

    pnl = pd.DataFrame({
        "gross": gross_pnl,
        "fees": -fee_pnl,
        "funding": -fund_pnl,
        "borrow": -borrow_pnl,
        "net": net_pnl,
    })
    return BacktestResult(
        equity=equity_curve,
        returns=net_pnl / aum,
        pnl=pnl,
        held=held,
        turnover=traded_notional / aum,
    )
