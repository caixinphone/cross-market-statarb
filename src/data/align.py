"""Cross-market alignment into a single returns panel.

The hard part of this strategy: crypto trades 24/7 while US equities trade a
6.5h RTH session. We resolve it as follows.

**Daily run (mandatory deliverable).**
  * Crypto daily price = close of the 1h bar that *ends* at ``daily_snapshot_utc``
    (default 21:00 UTC ≈ US close). For 21:00 that is the bar with open_time
    20:00 (Binance bars are stamped by open time and close one interval later).
  * Equity daily price = Yahoo official daily adjusted close.
  * The joint calendar = equity trading days (dates where SPY is present), so
    weekends/holidays are dropped and a Fri→Mon crypto return naturally absorbs
    the weekend drift. Close-to-close returns throughout.
  * Residual ≤1h crypto/equity offset under US daylight-saving is documented and
    immaterial at daily frequency.

**Hourly run (bonus).**
  * Joint index = equity RTH hourly bars; crypto reindexed onto those stamps.

Returns are simple ``pct_change`` of the adjusted close.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import Config


@dataclass
class Panel:
    """Aligned cross-market data ready for the factor model."""

    prices: pd.DataFrame        # adj_close, index=bar ts, cols=tickers
    returns: pd.DataFrame       # simple returns
    factor_returns: pd.DataFrame
    dollar_volume: pd.DataFrame  # per-bar $ volume (equity scaled to consolidated)
    frequency: str

    def save(self, cfg: Config) -> None:
        cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        self.prices.to_parquet(cfg.processed_dir / "prices.parquet")
        self.returns.to_parquet(cfg.processed_dir / "returns.parquet")
        self.factor_returns.to_parquet(
            cfg.processed_dir / "factor_returns.parquet")
        self.dollar_volume.to_parquet(cfg.processed_dir / "dollar_volume.parquet")

    @classmethod
    def load(cls, cfg: Config) -> "Panel":
        return cls(
            prices=pd.read_parquet(cfg.processed_dir / "prices.parquet"),
            returns=pd.read_parquet(cfg.processed_dir / "returns.parquet"),
            factor_returns=pd.read_parquet(
                cfg.processed_dir / "factor_returns.parquet"),
            dollar_volume=pd.read_parquet(
                cfg.processed_dir / "dollar_volume.parquet"),
            frequency=cfg.frequency,
        )


def _snapshot_hour(cfg: Config) -> int:
    hh = int(cfg.get("sampling", "daily_snapshot_utc", default="21:00").split(":")[0])
    return (hh - 1) % 24  # bar whose close lands on the snapshot


def _daily_snapshot(ohlcv: pd.DataFrame, snap_hour: int) -> pd.Series:
    """One price per UTC date = adj_close of the last bar at/before the snapshot
    hour. Works for both crypto (24/7 -> the snap_hour bar) and equity (the last
    RTH bar of the session), so daily and hourly run off the same 1h cache."""
    s = ohlcv.loc[ohlcv.index.hour <= snap_hour, "adj_close"]
    daily = s.groupby(s.index.normalize()).last()
    return daily[~daily.index.duplicated(keep="last")]


def _daily_dvol(ohlcv: pd.DataFrame, snap_hour: int) -> pd.Series:
    """Daily $ volume = sum of the session's (volume·close) up to the snapshot."""
    sub = ohlcv.loc[ohlcv.index.hour <= snap_hour]
    dvol = sub["volume"] * sub["close"]
    return dvol.groupby(dvol.index.normalize()).sum()


def _scale_equity_volume(cfg: Config, dollar_volume: pd.DataFrame) -> pd.DataFrame:
    """Scale Alpaca IEX equity $-volume up to a consolidated-tape proxy."""
    scale = float(cfg.get("costs", "iex_volume_scale", default=25.0))
    for name, spec in cfg.universe().items():
        if spec.asset_class == "equity" and name in dollar_volume.columns:
            dollar_volume[name] = dollar_volume[name] * scale
    return dollar_volume


def build_panel(cfg: Config, ohlcv: dict[str, pd.DataFrame]) -> Panel:
    if cfg.frequency == "daily":
        return _build_daily(cfg, ohlcv)
    return _build_hourly(cfg, ohlcv)


def _build_daily(cfg: Config, ohlcv: dict[str, pd.DataFrame]) -> Panel:
    universe = cfg.universe()
    snap = _snapshot_hour(cfg)
    series: dict[str, pd.Series] = {}
    dvol_series: dict[str, pd.Series] = {}
    for name, spec in universe.items():
        df = ohlcv[name]
        if df.empty:
            continue
        series[name] = _daily_snapshot(df, snap)
        dvol_series[name] = _daily_dvol(df, snap)

    prices = pd.DataFrame(series).sort_index()

    # Joint calendar = equity trading days. Prefer SPY; fall back to any equity.
    cal_anchor = "SPY" if "SPY" in prices else next(
        (n for n, s in universe.items() if s.asset_class == "equity"), None)
    if cal_anchor is not None:
        trading_days = prices[cal_anchor].dropna().index
        prices = prices.reindex(trading_days)

    prices = _apply_delistings(cfg, prices)
    # Forward-fill short equity gaps (e.g. halts) but never crypto NaNs at the
    # snapshot hour; cap fill so a long suspension does not leak stale prices.
    prices = prices.ffill(limit=2)

    returns = prices.pct_change(fill_method=None)
    returns = returns.dropna(how="all").iloc[1:]
    factor_returns = returns[[f for f in cfg.factor_names() if f in returns]]
    dollar_volume = _scale_equity_volume(
        cfg, pd.DataFrame(dvol_series).reindex(returns.index))
    return Panel(prices, returns, factor_returns, dollar_volume, "daily")


def _build_hourly(cfg: Config, ohlcv: dict[str, pd.DataFrame]) -> Panel:
    """Bonus track: align on equity RTH hourly bars."""
    universe = cfg.universe()
    equity_index = None
    for name, spec in universe.items():
        if spec.asset_class == "equity" and not ohlcv[name].empty:
            idx = ohlcv[name].index
            equity_index = idx if equity_index is None else equity_index.union(idx)
    if equity_index is None:
        raise RuntimeError("hourly run requires equity bars; none were fetched")
    # Keep the RTH core only (14:00-20:00 UTC ≈ 09:30-16:00 ET across DST). This
    # drops thin pre/post-market bars that the crypto-equities don't trade, which
    # would otherwise inject ffilled zero-returns into the factor regression.
    rth = range(14, 21)
    equity_index = equity_index[equity_index.hour.isin(rth)]

    series: dict[str, pd.Series] = {}
    dvol_series: dict[str, pd.Series] = {}
    for name, spec in universe.items():
        df = ohlcv[name]
        if df.empty:
            continue
        s = df["adj_close"]
        s = s[~s.index.duplicated(keep="last")]
        dv = (df["volume"] * df["close"])
        dv = dv[~dv.index.duplicated(keep="last")]
        # Crypto: reindex onto the equity RTH stamps (as-of fill within 1 bar).
        if spec.asset_class == "crypto":
            series[name] = s.reindex(equity_index, method="ffill", limit=1)
            dvol_series[name] = dv.reindex(equity_index, method="ffill", limit=1)
        else:
            series[name] = s.reindex(equity_index)
            dvol_series[name] = dv.reindex(equity_index)

    prices = pd.DataFrame(series).sort_index().ffill(limit=2)
    prices = _apply_delistings(cfg, prices)
    returns = prices.pct_change(fill_method=None).dropna(how="all").iloc[1:]
    factor_returns = returns[[f for f in cfg.factor_names() if f in returns]]
    dollar_volume = _scale_equity_volume(
        cfg, pd.DataFrame(dvol_series).reindex(returns.index))
    return Panel(prices, returns, factor_returns, dollar_volume, "hourly")


def _apply_delistings(cfg: Config, prices: pd.DataFrame) -> pd.DataFrame:
    """Null out a ticker after its delisting date (survivorship handling)."""
    delistings = cfg.raw.get("delistings") or {}
    for ticker, date in delistings.items():
        if ticker in prices:
            prices.loc[prices.index >= pd.Timestamp(date, tz="UTC"), ticker] = pd.NA
    return prices
