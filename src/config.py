"""Configuration loading and resolution.

Loads ``config/config.yaml`` and exposes a small typed-ish accessor that resolves
the active universe, per-asset factor maps, and source-symbol mapping so the rest
of the pipeline never re-parses raw YAML.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Which assets are crypto (priced on the Binance archive) vs equities (Yahoo).
# Anything not in CRYPTO_TICKERS is treated as an equity ticker.
CRYPTO_TICKERS = {
    "BTC", "ETH", "SOL", "BNB", "ARB", "OP", "UNI", "AAVE",
    "LINK", "AVAX", "DOT", "LTC", "POL", "MATIC",
}


@dataclass
class AssetSpec:
    """Resolved metadata for a single tradeable asset or factor."""

    name: str                 # canonical ticker, e.g. "MSTR", "ETH"
    category: str             # crypto_equities | altcoins | tech_perp | factor
    asset_class: str          # "crypto" | "equity"
    source_symbol: str        # source-specific symbol, e.g. "ETHUSDT" or "MSTR"
    factors: list[str] = field(default_factory=list)
    is_factor: bool = False


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    # ---- convenience accessors -------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.raw["run"]["seed"])

    @property
    def frequency(self) -> str:
        return self.raw["run"]["frequency"]

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / self.raw["run"]["data_dir"]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def start(self) -> dt.date:
        return dt.date.fromisoformat(self.raw["run"]["start"])

    @property
    def end(self) -> dt.date:
        end = self.raw["run"]["end"]
        if end:
            return dt.date.fromisoformat(end)
        # Default: last day of the previous month (last *complete* month).
        today = dt.date.today()
        first_of_month = today.replace(day=1)
        return first_of_month - dt.timedelta(days=1)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    # ---- universe resolution ---------------------------------------------------
    def asset_class_of(self, ticker: str) -> str:
        return "crypto" if ticker.upper() in CRYPTO_TICKERS else "equity"

    def source_symbol_of(self, ticker: str) -> str:
        overrides = self.raw.get("symbol_overrides") or {}
        ticker = overrides.get(ticker, ticker)
        if self.asset_class_of(ticker) == "crypto":
            return f"{ticker.upper()}USDT"
        return ticker.upper()

    def factors_for(self, ticker: str, category: str) -> list[str]:
        overrides = self.raw.get("factor_overrides") or {}
        if ticker in overrides:
            return list(overrides[ticker])
        return list(self.raw["factor_map"].get(category, []))

    def universe(self) -> dict[str, AssetSpec]:
        """Resolve the active universe + factors into AssetSpec objects.

        Returns a dict keyed by canonical ticker. Factors are included with
        ``is_factor=True`` and category ``"factor"``.
        """
        active = self.raw["active_universe"]
        cats = self.raw["universes"][active]
        specs: dict[str, AssetSpec] = {}

        for category, tickers in cats.items():
            for t in tickers:
                specs[t] = AssetSpec(
                    name=t,
                    category=category,
                    asset_class=self.asset_class_of(t),
                    source_symbol=self.source_symbol_of(t),
                    factors=self.factors_for(t, category),
                )

        for f in self.raw["factors"]:
            if f in specs:
                # Already a tradeable asset; also mark as factor.
                specs[f].is_factor = True
            else:
                specs[f] = AssetSpec(
                    name=f,
                    category="factor",
                    asset_class=self.asset_class_of(f),
                    source_symbol=self.source_symbol_of(f),
                    factors=[],
                    is_factor=True,
                )
        return specs

    def factor_names(self) -> list[str]:
        return list(self.raw["factors"])

    def tradeable_names(self) -> list[str]:
        """Assets that carry signals (everything that is not *only* a factor)."""
        return [s.name for s in self.universe().values() if s.category != "factor"]


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, path=path)
