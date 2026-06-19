"""Market-data sources.

Two default, key-free sources that sidestep the geo-blocked Binance live API:

* :class:`CryptoArchiveSource` — Binance's public bulk archive at
  ``data.binance.vision`` (spot klines + USDⓈ-M funding-rate dumps). Full history
  back to 2023 at 1h, no API key, not geo-blocked.
* :class:`YahooSource` — Yahoo Finance chart API via ``requests`` + a browser
  User-Agent (needed to avoid HTTP 429). Daily bars reach 2023; hourly bars are
  capped by Yahoo at ~730 days.

All sources expose the same contract::

    get_ohlcv(symbol, interval, start, end) -> pd.DataFrame

returning a UTC tz-aware DatetimeIndex (bar open time) with columns
``[open, high, low, close, adj_close, volume]`` sorted and de-duplicated.

Keyed sources (Polygon / AlphaVantage) are intentionally left as stubs — wiring
one in only touches this module plus a config flag.
"""

from __future__ import annotations

import datetime as dt
import io
import os
import time
import zipfile
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yaml

ARCHIVE_BASE = "https://data.binance.vision"
YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

OHLCV_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _month_range(start: dt.date, end: dt.date) -> Iterable[tuple[int, int]]:
    """Yield (year, month) tuples spanning start..end inclusive."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def _to_utc_index(epoch_series: pd.Series) -> pd.DatetimeIndex:
    """Convert a Binance timestamp column to a UTC index.

    Binance emitted these in ms historically and switched to µs for some
    datasets in 2025 — and a single concatenated series can therefore *mix*
    units. Normalise every value to milliseconds by magnitude (per element),
    then parse, so a unit change mid-series can't blow up into year 56971.
    """
    arr = pd.to_numeric(epoch_series, errors="coerce").to_numpy(dtype="float64")
    ms = np.select(
        [arr > 1e17, arr > 1e14, arr > 1e11],   # ns, µs, ms
        [arr / 1e6, arr / 1e3, arr],
        default=arr * 1e3,                        # seconds
    )
    return pd.to_datetime(ms, unit="ms", utc=True)


# ---------------------------------------------------------------------------
# Binance public archive
# ---------------------------------------------------------------------------
class CryptoArchiveSource:
    """Fetch klines and funding rates from ``data.binance.vision``."""

    KLINE_COLS = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
        "ignore",
    ]

    def __init__(self, session: requests.Session, timeout: int = 20,
                 max_retries: int = 4):
        self.session = session
        self.timeout = timeout
        self.max_retries = max_retries

    # -- low-level zip fetch ------------------------------------------------
    def _get_zip_csv(self, url: str) -> pd.DataFrame | None:
        """Download a .zip, return its single CSV as a raw (headerless) frame.

        Returns ``None`` on 404 (file simply doesn't exist for that period).
        """
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, timeout=self.timeout)
            except requests.RequestException:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                    name = zf.namelist()[0]
                    with zf.open(name) as fh:
                        return pd.read_csv(fh, header=None)
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"failed to fetch {url} after {self.max_retries} tries")

    @staticmethod
    def _strip_header(df: pd.DataFrame) -> pd.DataFrame:
        """Drop a header row if the archive shipped one (newer files do)."""
        first = str(df.iloc[0, 0]).strip().lower()
        if first in {"open_time", "calc_time"}:
            return df.iloc[1:].reset_index(drop=True)
        return df

    # -- public API ---------------------------------------------------------
    def get_ohlcv(self, symbol: str, interval: str, start: dt.date,
                  end: dt.date) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for year, month in _month_range(start, end):
            url = (f"{ARCHIVE_BASE}/data/spot/monthly/klines/{symbol}/{interval}/"
                   f"{symbol}-{interval}-{year:04d}-{month:02d}.zip")
            raw = self._get_zip_csv(url)
            if raw is None:
                # No monthly file (usually the current, incomplete month) ->
                # stitch from daily files.
                raw = self._daily_klines(symbol, interval, year, month)
            if raw is not None and len(raw):
                frames.append(raw)
        if not frames:
            return pd.DataFrame(columns=OHLCV_COLS)
        return self._format_klines(pd.concat(frames, ignore_index=True))

    def _daily_klines(self, symbol: str, interval: str, year: int,
                      month: int) -> pd.DataFrame | None:
        days: list[pd.DataFrame] = []
        d = dt.date(year, month, 1)
        while d.month == month and d <= dt.date.today():
            url = (f"{ARCHIVE_BASE}/data/spot/daily/klines/{symbol}/{interval}/"
                   f"{symbol}-{interval}-{d.isoformat()}.zip")
            raw = self._get_zip_csv(url)
            if raw is not None and len(raw):
                days.append(raw)
            d += dt.timedelta(days=1)
        return pd.concat(days, ignore_index=True) if days else None

    def _format_klines(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = self._strip_header(raw)
        raw = raw.iloc[:, : len(self.KLINE_COLS)]
        raw.columns = self.KLINE_COLS
        idx = _to_utc_index(raw["open_time"])
        # Use .to_numpy() so values position-align with idx; passing Series with
        # their own (integer) index into DataFrame(index=...) would reindex->NaN.
        out = pd.DataFrame(
            {
                "open": pd.to_numeric(raw["open"], errors="coerce").to_numpy(),
                "high": pd.to_numeric(raw["high"], errors="coerce").to_numpy(),
                "low": pd.to_numeric(raw["low"], errors="coerce").to_numpy(),
                "close": pd.to_numeric(raw["close"], errors="coerce").to_numpy(),
                "volume": pd.to_numeric(raw["volume"], errors="coerce").to_numpy(),
            },
            index=idx,
        )
        # Crypto has no corporate actions; adj_close == close.
        out["adj_close"] = out["close"]
        out = out[OHLCV_COLS]
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    def get_funding(self, symbol: str, start: dt.date,
                    end: dt.date) -> pd.Series:
        """USDⓈ-M perp funding rate series (per 8h settlement), UTC-indexed."""
        frames: list[pd.DataFrame] = []
        for year, month in _month_range(start, end):
            url = (f"{ARCHIVE_BASE}/data/futures/um/monthly/fundingRate/{symbol}/"
                   f"{symbol}-fundingRate-{year:04d}-{month:02d}.zip")
            raw = self._get_zip_csv(url)
            if raw is not None and len(raw):
                frames.append(raw)
        if not frames:
            return pd.Series(dtype=float, name="funding")
        raw = self._strip_header(pd.concat(frames, ignore_index=True))
        calc_time = raw.iloc[:, 0]
        rate = pd.to_numeric(raw.iloc[:, -1], errors="coerce")  # last col = rate
        s = pd.Series(rate.values, index=_to_utc_index(calc_time), name="funding")
        # Drop any unparseable rows (interior header lines in multi-month concat
        # coerce to NaT/NaN); keep only clean, unique, sorted timestamps.
        s = s[s.index.notna() & s.notna()]
        return s[~s.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------------------
# Yahoo Finance chart API
# ---------------------------------------------------------------------------
class YahooSource:
    """Fetch equity OHLCV from the Yahoo chart endpoint."""

    def __init__(self, session: requests.Session, timeout: int = 20,
                 max_retries: int = 4):
        self.session = session
        self.timeout = timeout
        self.max_retries = max_retries

    def get_ohlcv(self, symbol: str, interval: str, start: dt.date,
                  end: dt.date) -> pd.DataFrame:
        period1 = int(dt.datetime(start.year, start.month, start.day,
                                  tzinfo=dt.timezone.utc).timestamp())
        period2 = int((dt.datetime(end.year, end.month, end.day,
                                   tzinfo=dt.timezone.utc)
                       + dt.timedelta(days=1)).timestamp())
        params = {
            "interval": interval,
            "period1": period1,
            "period2": period2,
            "events": "div,split",
            "includeAdjustedClose": "true",
        }
        url = f"{YAHOO_BASE}/{symbol}"
        data = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                data = r.json()
                break
            time.sleep(1.5 * (attempt + 1))
        if data is None:
            raise RuntimeError(f"Yahoo fetch failed for {symbol} ({interval})")

        result = data["chart"]["result"]
        if not result or not result[0].get("timestamp"):
            return pd.DataFrame(columns=OHLCV_COLS)
        res = result[0]
        idx = pd.to_datetime(res["timestamp"], unit="s", utc=True)
        q = res["indicators"]["quote"][0]
        adj = (res["indicators"].get("adjclose", [{}])[0].get("adjclose")
               if res["indicators"].get("adjclose") else q["close"])
        out = pd.DataFrame(
            {
                "open": q["open"], "high": q["high"], "low": q["low"],
                "close": q["close"], "adj_close": adj, "volume": q["volume"],
            },
            index=idx,
        )[OHLCV_COLS]
        out = out.dropna(how="all")
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out


# ---------------------------------------------------------------------------
# Alpaca market data (keyed) — full-history hourly equities, IEX feed on free tier
# ---------------------------------------------------------------------------
class AlpacaSource:
    """Fetch equity bars from Alpaca's market-data API (paginated, adjusted)."""

    BASE = "https://data.alpaca.markets/v2/stocks"
    TF = {"1h": "1Hour", "1d": "1Day", "1min": "1Min",
          "1Hour": "1Hour", "1Day": "1Day"}

    def __init__(self, session: requests.Session, feed: str = "iex",
                 timeout: int = 20, max_retries: int = 4):
        self.session = session
        self.feed = feed          # 'iex' (free) | 'sip' (paid)
        self.timeout = timeout
        self.max_retries = max_retries

    def _get(self, url: str, params: dict) -> dict:
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:          # rate limited -> back off
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code in (401, 403):
                raise RuntimeError(
                    f"Alpaca auth/subscription error {r.status_code}: {r.text[:200]} "
                    f"(feed='{self.feed}'; SIP needs a paid plan)")
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Alpaca fetch failed: {url}")

    def get_ohlcv(self, symbol: str, interval: str, start: dt.date,
                  end: dt.date) -> pd.DataFrame:
        params = {
            "timeframe": self.TF.get(interval, interval),
            "start": f"{start.isoformat()}T00:00:00Z",
            "end": f"{end.isoformat()}T23:59:59Z",
            "limit": 10000,
            "adjustment": "all",     # split + dividend adjusted -> clean returns
            "feed": self.feed,
            "sort": "asc",
        }
        url = f"{self.BASE}/{symbol}/bars"
        rows: list[dict] = []
        page_token = None
        while True:
            p = dict(params)
            if page_token:
                p["page_token"] = page_token
            data = self._get(url, p)
            rows.extend(data.get("bars") or [])
            page_token = data.get("next_page_token")
            if not page_token:
                break
        if not rows:
            return pd.DataFrame(columns=OHLCV_COLS)
        df = pd.DataFrame(rows)
        idx = pd.to_datetime(df["t"], utc=True)
        out = pd.DataFrame(
            {
                "open": df["o"].to_numpy(), "high": df["h"].to_numpy(),
                "low": df["l"].to_numpy(), "close": df["c"].to_numpy(),
                "adj_close": df["c"].to_numpy(),   # already adjustment='all'
                "volume": df["v"].to_numpy(),
            },
            index=idx,
        )[OHLCV_COLS]
        return out[~out.index.duplicated(keep="last")].sort_index()


def _load_alpaca_creds(cfg) -> tuple[str, str]:
    """Resolve Alpaca creds: env vars first, then config/secrets.yaml."""
    kid = os.environ.get("ALPACA_API_KEY_ID")
    sec = os.environ.get("ALPACA_API_SECRET_KEY")
    if kid and sec:
        return kid, sec
    secrets_path = cfg.path.parent / "secrets.yaml"
    if secrets_path.exists():
        with open(secrets_path) as fh:
            s = yaml.safe_load(fh) or {}
        kid = s.get("alpaca_api_key_id")
        sec = s.get("alpaca_api_secret_key")
        if kid and sec:
            return kid, sec
    raise RuntimeError(
        "Alpaca creds not found. Set ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY "
        "env vars or config/secrets.yaml.")


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def build_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s


def make_sources(cfg) -> dict[str, object]:
    """Construct the configured source objects keyed by asset class."""
    session = build_session(cfg.get("sources", "user_agent",
                                    default="Mozilla/5.0"))
    timeout = cfg.get("sources", "request_timeout", default=20)
    retries = cfg.get("sources", "max_retries", default=4)

    equity_source = cfg.get("sources", "equity_source", default="yahoo")
    if equity_source == "yahoo":
        equity = YahooSource(session, timeout, retries)
    elif equity_source == "alpaca":
        kid, sec = _load_alpaca_creds(cfg)
        asess = build_session(cfg.get("sources", "user_agent", default="Mozilla/5.0"))
        asess.headers.update({"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})
        feed = cfg.get("sources", "feed", default="iex")
        equity = AlpacaSource(asess, feed=feed, timeout=timeout, max_retries=retries)
    else:
        raise NotImplementedError(
            f"equity_source='{equity_source}' not wired. Options: yahoo, alpaca.")
    return {
        "crypto": CryptoArchiveSource(session, timeout, retries),
        "equity": equity,
    }
