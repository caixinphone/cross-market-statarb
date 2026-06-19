"""Risk constraints + factor-hedge construction.

Pipeline (per bar, vectorised over the time index):

1. Clip each asset leg to ``max_asset_weight`` of AUM.
2. Scale each sector (category) group down to ``max_sector_weight`` gross.
3. Build factor hedge legs from the *constrained* asset legs:
   ``factor_pos_f = -Σ_i asset_pos_i · β_{i,f}``  → factor-neutral by construction.
4. Assemble combined positions over all instruments (a dual asset/factor such as
   ETH nets its signal leg and its hedge leg).
5. Re-clip per-asset (factor legs can stack); this may leave a small residual
   factor exposure, which is measured and reported.
6. Scale the whole book to ``max_gross_leverage``.

Net-factor-exposure note: pure factor instruments (BTC/SPY/QQQ/SMH — hedge-only)
are neutralised by step 3 and enforced. A dual instrument's own idiosyncratic
leg (e.g. ETH) is an intended bet, so its self-exposure is reported but not
forced to zero. This choice is documented in the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import AssetSpec, Config
from ..factors.factor_model import FactorModelResult


@dataclass
class PortfolioResult:
    positions: pd.DataFrame      # signed $ per instrument, index=bars
    report: pd.DataFrame         # per-bar risk diagnostics
    asset_legs: pd.DataFrame     # constrained asset legs (for attribution)


def _sector_map(universe: dict[str, AssetSpec]) -> dict[str, list[str]]:
    sectors: dict[str, list[str]] = {}
    for name, spec in universe.items():
        if spec.category != "factor":
            sectors.setdefault(spec.category, []).append(name)
    return sectors


def _build_factor_legs(asset_pos: pd.DataFrame, fm: FactorModelResult,
                       factor_names: list[str], idx) -> pd.DataFrame:
    factor_pos = pd.DataFrame(0.0, index=idx, columns=factor_names)
    for asset in asset_pos.columns:
        betas = fm.betas.get(asset)
        if betas is None:
            continue
        for f in fm.factor_map.get(asset, []):
            if f in factor_pos.columns and f in betas.columns:
                b = betas[f].reindex(idx).fillna(0.0)
                factor_pos[f] = factor_pos[f] - asset_pos[asset] * b
    return factor_pos


def apply_risk_constraints(
    cfg: Config,
    asset_notional: pd.DataFrame,
    fm: FactorModelResult,
) -> PortfolioResult:
    universe = cfg.universe()
    aum = float(cfg.get("portfolio", "aum", default=10_000_000))
    rc = cfg.raw["portfolio"]["risk"]
    cap_asset = float(rc["max_asset_weight"]) * aum
    cap_sector = float(rc["max_sector_weight"]) * aum
    cap_gross = float(rc["max_gross_leverage"]) * aum
    cap_factor = float(rc["max_net_factor_exposure"]) * aum
    factor_names = cfg.factor_names()
    idx = asset_notional.index

    # 1. per-asset clip
    asset_pos = asset_notional.clip(-cap_asset, cap_asset)

    # 2. per-sector scaling
    for sector, members in _sector_map(universe).items():
        cols = [c for c in members if c in asset_pos.columns]
        if not cols:
            continue
        gross = asset_pos[cols].abs().sum(axis=1)
        scale = (cap_sector / gross).clip(upper=1.0).replace([np.inf], 1.0).fillna(1.0)
        asset_pos[cols] = asset_pos[cols].mul(scale, axis=0)

    # 3. factor hedge legs from constrained asset legs
    factor_pos = _build_factor_legs(asset_pos, fm, factor_names, idx)

    # 4. assemble combined positions
    instruments = sorted(set(asset_pos.columns) | set(factor_names))
    positions = pd.DataFrame(0.0, index=idx, columns=instruments)
    positions[asset_pos.columns] = positions[asset_pos.columns].add(asset_pos, fill_value=0.0)
    positions[factor_names] = positions[factor_names].add(factor_pos, fill_value=0.0)

    # 5. enforce per-asset cap by scaling the whole book *uniformly* rather than
    #    clipping individual legs. Clipping a factor hedge leg (which must absorb
    #    many signals sharing a factor) would break factor-neutrality; a uniform
    #    down-scale keeps the book neutral while bringing the largest leg to cap.
    maxpos = positions.abs().max(axis=1)
    scale_a = (cap_asset / maxpos).clip(upper=1.0).replace([np.inf], 1.0).fillna(1.0)
    positions = positions.mul(scale_a, axis=0)

    # 6. gross leverage scaling (also uniform -> neutrality preserved)
    gross = positions.abs().sum(axis=1)
    scale = (cap_gross / gross).clip(upper=1.0).replace([np.inf], 1.0).fillna(1.0)
    positions = positions.mul(scale, axis=0)

    # 7. enforce the net-factor-exposure cap. A small residual survives step 3
    #    because a dual factor (ETH) carries its own BTC loading when used to
    #    hedge other altcoins; a uniform down-scale removes it. Target 90% of the
    #    cap so the downstream no-trade band can let exposure drift without
    #    breaching the hard 5% limit on the actually-held book.
    netf = _net_factor_dollars(positions, asset_pos.columns, fm, factor_names)
    max_nf = netf.abs().max(axis=1)
    scale_f = (0.9 * cap_factor / max_nf).clip(upper=1.0).replace([np.inf], 1.0).fillna(1.0)
    positions = positions.mul(scale_f, axis=0)

    report = build_risk_report(cfg, positions, list(asset_pos.columns), fm)
    return PortfolioResult(positions=positions, report=report, asset_legs=asset_pos)


def _net_factor_dollars(positions: pd.DataFrame, signal_cols, fm: FactorModelResult,
                        factor_names: list[str]) -> pd.DataFrame:
    """Net $ exposure to each factor: Σ_j pos_j · loading_{j->f} (self-loading=1)."""
    out = {}
    for f in factor_names:
        exp = positions[f].copy() if f in positions else pd.Series(0.0, positions.index)
        for a in signal_cols:
            betas = fm.betas.get(a)
            if betas is not None and f in fm.factor_map.get(a, []) and f in betas:
                exp = exp + positions[a] * betas[f].reindex(positions.index).fillna(0.0)
        out[f] = exp
    return pd.DataFrame(out)


def build_risk_report(cfg: Config, positions: pd.DataFrame, signal_cols: list[str],
                      fm: FactorModelResult) -> pd.DataFrame:
    """Per-bar risk diagnostics for any book (compliant targets or actual held)."""
    aum = float(cfg.get("portfolio", "aum", default=10_000_000))
    factor_names = cfg.factor_names()
    gross = positions.abs().sum(axis=1)
    net_factor = _net_factor_dollars(positions, signal_cols, fm, factor_names)

    report = pd.DataFrame({
        "gross_leverage": gross / aum,
        "n_positions": (positions.abs() > 1.0).sum(axis=1),
        "max_asset_weight": positions.abs().max(axis=1) / aum,
    })
    for f in factor_names:
        report[f"netfactor_{f}"] = net_factor[f] / aum
    report["max_abs_netfactor"] = net_factor.abs().max(axis=1) / aum
    return report
