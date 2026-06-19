"""Download orchestration with an idempotent parquet cache.

Pulls every asset/factor in the active universe from its source and caches the
raw OHLCV (and crypto funding) under ``data/raw/``. Re-runs are cheap: an
existing cache file is reused unless ``force=True``.

Granularity rules:
  * daily run   -> crypto fetched at 1h (snapshotted to the US-close hour in
                   :mod:`align`), equities fetched at 1d.
  * hourly run  -> both fetched at 1h.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import AssetSpec, Config
from .sources import make_sources


def _crypto_interval(freq: str) -> str:
    return "1h"  # always 1h; daily is snapshotted from 1h bars in align.py


def _equity_interval(freq: str) -> str:
    return "1h"  # always 1h (Alpaca); daily is snapshotted from 1h bars too


def _cache_path(cfg: Config, name: str, kind: str) -> Path:
    return cfg.raw_dir / f"{name}_{kind}.parquet"


def _source_symbols(cfg: Config, name: str, spec) -> list[str]:
    """Source symbol(s) for an asset; a list when a ticker rename is stitched."""
    stitch = cfg.raw.get("symbol_stitch") or {}
    return list(stitch[name]) if name in stitch else [spec.source_symbol]


def _concat_dedup(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_universe(cfg: Config, force: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch + cache OHLCV for every asset/factor. Returns {name: ohlcv}."""
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    sources = make_sources(cfg)
    universe = cfg.universe()
    out: dict[str, pd.DataFrame] = {}

    for name, spec in universe.items():
        interval = (_crypto_interval(cfg.frequency)
                    if spec.asset_class == "crypto"
                    else _equity_interval(cfg.frequency))
        kind = f"ohlcv_{interval}"
        path = _cache_path(cfg, name, kind)

        if path.exists() and not force:
            out[name] = pd.read_parquet(path)
            continue

        src = sources[spec.asset_class]
        symbols = _source_symbols(cfg, name, spec)
        df = _concat_dedup([src.get_ohlcv(sym, interval, cfg.start, cfg.end)
                            for sym in symbols])
        if df.empty:
            print(f"  WARNING: no data for {name} ({symbols} {interval})")
        df.to_parquet(path)
        out[name] = df
        tag = "+".join(symbols) if len(symbols) > 1 else ""
        print(f"  fetched {name:<6} [{spec.asset_class}/{interval}] rows={len(df)} {tag}")

    return out


def fetch_funding(cfg: Config, force: bool = False) -> dict[str, pd.Series]:
    """Fetch + cache USDⓈ-M funding for every crypto asset (perp leg cost)."""
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    sources = make_sources(cfg)
    crypto = sources["crypto"]
    out: dict[str, pd.Series] = {}

    for name, spec in cfg.universe().items():
        if spec.asset_class != "crypto":
            continue
        path = _cache_path(cfg, name, "funding")
        if path.exists() and not force:
            out[name] = pd.read_parquet(path)["funding"]
            continue
        symbols = _source_symbols(cfg, name, spec)
        parts = [crypto.get_funding(sym, cfg.start, cfg.end) for sym in symbols]
        frames = [p.to_frame("funding") for p in parts if len(p)]
        df = _concat_dedup(frames)
        s = df["funding"] if "funding" in df.columns else pd.Series(dtype=float, name="funding")
        s.to_frame("funding").to_parquet(path)
        out[name] = s
        print(f"  funding {name:<6} settlements={len(s)}")
    return out
