# Cross-Market Idiosyncratic Mean-Reversion (Statistical Arbitrage)

A market-neutral statistical-arbitrage backtest across crypto and US-equity
proxies, in the spirit of a Binance unified cross-market account. Each asset is
modelled as a rolling multi-factor regression on systematic factors (BTC, ETH,
SPY, QQQ, SMH); the **idiosyncratic residual** is traded back to its mean while
factor exposure is hedged out, so the book earns only the correction of
short-term mispricing.

```
r_i,t = α_i + Σ_f β_{i,f}·r_f,t + ε_i,t      # rolling factor model
signal = z-score of the residual spread      # rich/cheap vs factors
trade  = fade the deviation, β-hedge factors  # market-neutral pair
```

## Quickstart

```bash
pip install -e .                 # pinned deps from pyproject.toml
# equity data uses Alpaca: put keys in config/secrets.yaml (gitignored) or
# ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY env vars
python scripts/download_data.py  # fetch + cache raw data, build aligned panel
python scripts/run_backtest.py   # factor model -> signals -> portfolio -> metrics
python scripts/sensitivity.py    # parameter / cost / capacity grids -> reports/
pip install -e ".[ml]"           # optional: torch, for the ML bonus (see report §16)
python scripts/run_ml.py         # CNN size overlay + drift gate vs baseline (§16)
pytest -q                        # 16 tests (+3 ML with torch): lookahead, risk, costs, funding, ...
```

Everything is driven by [`config/config.yaml`](config/config.yaml) — universe,
factor map, signal thresholds, risk caps, and costs — so sensitivity/cost/
capacity analyses are config-only, no code edits. The **primary track is hourly**
(`frequency: hourly`, the default); switch `frequency: daily` to reproduce the
daily comparison (both run off the same 1h cache). Bar-count windows are set in the
hourly track's units (the daily calendar convention × 7 RTH bars/day). See
**[reports/strategy_report.md](reports/strategy_report.md)** for the full write-up.

## Architecture

| Module | Responsibility |
|---|---|
| `src/data/sources.py` | Fetchers: Binance bulk archive (crypto klines + funding), Alpaca (equities), Yahoo (fallback) |
| `src/data/fetch.py` | Orchestration + idempotent parquet cache + ticker-rename stitch |
| `src/data/align.py` | Cross-market UTC alignment → returns + dollar-volume panel |
| `src/factors/factor_model.py` | Point-in-time rolling OLS → β, residuals |
| `src/factors/diagnostics.py` | R² + ADF residual-stationarity table |
| `src/signals/zscore.py` | Residual z-score, half-life filter, entry/exit state machine |
| `src/portfolio/construct.py` | Equal-volatility sizing, fixed at entry |
| `src/portfolio/risk.py` | Factor hedging + asset/sector/leverage/factor caps |
| `src/portfolio/execution.py` | No-trade band → actually-held book |
| `src/backtest/{engine,costs}.py` | PIT backtest + fees/liquidity-slippage/funding/borrow |
| `src/analysis/{metrics,attribution,plots}.py` | Metrics, PnL decomposition, charts |
| `scripts/sensitivity.py` | Parameter / cost / capacity grids |

## Data sources & caveats (important)

* **Binance live API is geo-blocked (HTTP 451)** from the build location. Crypto
  data therefore comes from the **official public archive `data.binance.vision`**
  (spot 1h klines + USDⓈ-M funding) — full history to 2023, no key, and the
  better source for backtesting anyway. The archive switched its timestamp unit
  from ms to µs in 2025; `sources.py` normalises per-element.
* **Equities** come from the **Alpaca market-data API** (full 2023→now hourly,
  IEX feed on the free tier; SIP needs a paid plan). Keys live in a gitignored
  `config/secrets.yaml` or env vars. Yahoo remains a daily fallback (`yfinance`
  not required); Yahoo's hourly history is capped at ~730 days, which is why
  Alpaca is used for the full-2023 hourly series.
* **Backtest framing.** Binance's TradFi perps did not exist back to 2023, so the
  equity legs use real US-equity prices and the crypto legs use Binance;
  "Binance cross-market" is the execution thesis, not the historical data source.
* **Alignment.** The hourly (primary) and daily (comparison) panels derive from
  the **same 1h cache**. Hourly joins on the equity RTH core (14:00–20:00 UTC, 7
  bars/day) with crypto reindexed onto those stamps; daily price = the last 1h bar
  at/before the US-close snapshot (21:00 UTC) on the equity trading calendar. A ≤1h
  DST offset is immaterial.
* **Survivorship / ticker changes** handled via config (`delistings`,
  `symbol_overrides`); a pre-listing gap is absent, not back-filled (e.g. ARB/OP
  before their 2023 launch). **POL** is the MATIC→POL rebrand: Binance's
  `POLUSDT` only starts at the ~Sep-2024 rename, so POL carries ~50% history
  until a MATIC→POL stitch is added.
* **Equity feed.** Alpaca free tier = IEX feed (a volume subset); OHLC for liquid
  names is representative, volumes are IEX-only. SIP (full tape) needs a paid plan
  (`sources.feed: sip`).

## Methodology highlights

* **Calendar-equivalent windows (hourly track).** Bar-count windows use the daily
  Avellaneda-Lee convention × 7 RTH bars/day: factor/z-score window 420 (≈60
  trading days), max-hold 140 (≈20 days). Using raw 60 hourly bars (≈8.5 days)
  would be far too short and fabricate turnover — the central methodological point.
* **No lookahead.** Betas/residuals at `t` use a trailing window ending at `t`;
  the engine executes signals on `t+1` (`positions.shift(1)`). Enforced by
  `tests/test_no_lookahead.py` (perturbing future bars cannot change past
  signals).
* **Stationarity gate.** Every residual is ADF-tested; the strategy only has an
  edge where the residual mean-reverts (all 25 stationary, ADF p≈0).
* **Equal-vol sizing.** `N_i = (target_vol / σ_resid_i) · AUM/n_signals`.
* **Risk caps** (all verified to bind/hold): per-asset ≤3%, sector ≤15%, gross
  leverage ≤3×, net factor exposure ≤5% of AUM.
* **Costs.** Per-leg taker fee (equity = spot 0.10%, crypto = perp 0.04%) +
  liquidity slippage on traded notional, USDⓈ-M funding (real 8h series, accrued
  per-bar with overnight settlements rolled onto the next session's first RTH bar),
  and short-equity borrow — each leg's fee, funding/borrow kept self-consistent.

## Reproducibility

Pinned deps in `pyproject.toml`; global seed in config; downloads cached and
idempotent (delete `data/raw` / `data/processed` to force a clean rebuild). The
pipeline is deterministic — re-running yields identical results.

## Key findings (hourly, full 25-asset universe, 2023→2026)

Full write-up: **[reports/strategy_report.md](reports/strategy_report.md)**.
`run_backtest.py` prints a PnL decomposition that separates *where the edge is*
from *what eats it*. All risk caps hold (gross 0.27×, asset 3.0%, net-factor 4.8%);
net beta to every factor ≈ 0. Panel: 5944 hourly bars × 29 columns.

| Component | PnL ($) |
|---|--:|
| Idiosyncratic edge `Σ held·ε` | **+599,187** |
| Alpha-drift + hedge-error | **−547,306** |
| Gross (tradeable) | +51,881 |
| Fees+slippage / funding / borrow | −234,654 |
| **Net** | **−182,772** |

Headline: Sharpe −0.44, CAGR −0.55%, vol 1.24%, max DD −3.10%, turnover 6.2×/yr,
avg hold ≈109 bars (~15.6 trading days). Fees split by venue: equity legs at
Binance spot taker 0.10%, crypto legs at perp taker 0.04%.

* The **idiosyncratic reversion edge is real and large** (+$599k gross), but a
  **market-neutral residual-reversion book is implicitly short idiosyncratic
  momentum** — shorting high-drift names in a 2023-25 bull market bleeds the drift
  the in-sample residual removes but a real β-hedge cannot (−$547k). This drag,
  larger than costs, is the central finding.
* **Methodologically-correct hourly ≈ daily.** With calendar-equivalent 420-bar
  windows the holding period is ~15 days and turnover is only 6.2×/yr — *not* the
  43× of a naive 60-hourly-bar run. The edge/drift/net structure matches the daily
  track (+$596k / −$561k / −$147k). More bars give steadier statistics, **not a
  different strategy.**
* **Capacity is not the limit:** Sharpe is flat from $10M→$200M (huge ADVs;
  impact-light). The ceiling is *edge*, not liquidity.
* **Honest verdict:** thin, parameter-sensitive, net-marginal — a viable
  *component* of a diversified neutral book (esp. with a drift/momentum overlay),
  not a standalone strategy. This is the **methodologically-corrected** hourly run;
  turnover-control / profit tuning was deliberately *not* applied. See the report
  for the full sensitivity/capacity analysis and roadmap.

## Roadmap (next steps)

* **Drift/momentum overlay** (implemented, see report §16.6): a signal-layer gate
  that blocks shorting strongly up-trending names cuts the alpha-drift −$547k→−$329k
  and flips the book net-positive (Sharpe −0.44→+0.37) — but the gain is partly a
  2023-25 bull-market mirror effect (off by default; needs out-of-regime validation).
* **Turnover control + profit tuning** (deliberately not applied here): higher
  thresholds, wider no-trade band, maker fills, the net-positive 280-bar window.
* OU s-score with κ-filter as primary signal; orthogonalised/PCA factors.
* Dynamic factor selection (LASSO/stepwise); perp-vs-spot basis module.
