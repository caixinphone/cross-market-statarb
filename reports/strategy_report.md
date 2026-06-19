# Cross-Market Idiosyncratic Mean-Reversion — Strategy Report

**A market-neutral statistical-arbitrage strategy across crypto and US-equity
proxies, backtested 2023-01 → 2026-05.**

---

## 1. Executive summary

We build and rigorously backtest an *idiosyncratic mean-reversion* statistical-
arbitrage strategy spanning crypto (spot/perp) and US equities (crypto-equities,
tech). Each asset is regressed on systematic factors (BTC, ETH, SPY, QQQ, SMH);
the residual is traded back toward its mean while factor exposure is hedged out,
so the book earns only the correction of short-term idiosyncratic mispricing.

**Headline (daily, full 25-asset universe, brief's ±2.0/0.5 thresholds):**

| Metric | Value |
|---|--:|
| Annualised return (CAGR) | −0.25% |
| Annualised volatility | 1.22% |
| Sharpe | −0.20 |
| Max drawdown | −2.4% |
| Annual turnover (2-way) | 4.8× |
| Net market beta (BTC/SPY/QQQ) | ≈ 0.00 |

The honest conclusion up front: **the idiosyncratic reversion edge is real and
large in gross terms (+$596k over the sample), but it is almost entirely offset
by an alpha-drift / hedge-error drag (−$561k), leaving a thin positive gross
(+$35k) that transaction costs (−$118k) turn slightly negative.** The strategy is
*marginal and parameter-sensitive*, not a robust standalone money-maker at daily
frequency on this universe. Its value — and the focus of this report — is the
**diagnosis of where the alpha is and what destroys it**, supported by a
realistic, auditable backtest engine.

The three findings that matter most:

1. **The edge is structurally real but fights two frictions.** A 2σ residual
   deviation reverts ~+2.3% over 10 days (event study). But shorting names *after*
   they spike means shorting high-idiosyncratic-drift names in a 2023-25 bull
   market — you bleed the drift the in-sample residual removes but a real hedge
   cannot. This *alpha drift* is the dominant cost, larger than fees.
2. **The edge is stronger intraday.** At hourly frequency the gross idiosyncratic
   edge is ~3-4× larger; gross PnL is clearly positive. But turnover (~40×/yr)
   makes fees the binding constraint there.
3. **Capacity is not the binding constraint at daily frequency.** Sharpe is flat
   from $10M to $200M AUM — the universe's daily dollar-volume (e.g. AAPL ~$5B,
   MSTR ~$760M) dwarfs the trade sizes, so market impact is negligible. The
   ceiling is *edge*, not liquidity.

---

## 2. Strategy thesis & economic rationale

**One sentence:** find assets that have moved too far from what their systematic
factors justify, trade against the deviation, hedge the factors, and collect the
mean-reversion.

**Why mispricings exist here.** Binance-style cross-market venues let
retail-driven flow price crypto-equities (MSTR, COIN, miners) and tokenised/perp
US names in the same 24/7 book as crypto. Retail over- and under-reaction to
idiosyncratic news (a Saylor tweet, an earnings rumour, a miner's hash-rate
report) pushes a name away from its factor-implied value faster than it pushes
the factor itself. The decomposition

```
asset return = systematic (factor·β)  +  idiosyncratic (ε)
```

isolates ε, the "asset's own story". When ε accumulates into an extreme
deviation, the bet is that it reverts. Hedging the factors removes market
direction, so the P&L depends only on the correction — *we do not predict
BTC or the Nasdaq*.

**Why it is plausibly an alpha and not a risk premium.** The residuals are
short-horizon, idiosyncratic, and market-neutral; the return is uncorrelated with
the factors by construction (verified: net beta ≈ 0). The economic source is
liquidity provision to retail over-reaction — a capacity-limited but genuine edge.

---

## 3. Data & universe

| Layer | Source | Coverage |
|---|---|---|
| Crypto spot/perp (1h) | `data.binance.vision` public archive | full 2023-01 → 2026-05 |
| Perp funding (8h) | `data.binance.vision` futures/um | full |
| US equities (1h) | Alpaca market-data API (IEX feed) | full 2023-01 → 2026-05 |

**Universe (25 tradeable + 5 factors).** Crypto-equities: MSTR, COIN, MARA, RIOT,
CLSK, HOOD. Altcoins: ETH, SOL, BNB, ARB, OP, UNI, AAVE, LINK, AVAX, DOT, LTC,
POL. Tech: NVDA, TSLA, AMD, META, AAPL, MSFT, AMZN. Factors: BTC, ETH, SPY, QQQ,
SMH.

**Key data decisions and caveats (handled, not hidden):**

- **Binance live API is geo-blocked (HTTP 451)** from the build location; we use
  the official bulk archive, which is the better backtest source anyway (complete,
  no rate limits). The archive switched timestamp units ms→µs in 2025 — normalised
  per-element.
- **Backtest framing.** Binance's TradFi perps did not exist back to 2023, so the
  equity legs use real US-equity prices and crypto legs use Binance. "Binance
  cross-market" is the execution thesis, not the historical data source.
- **Equity feed.** Alpaca's free tier is the IEX feed (a volume subset); OHLC for
  liquid names is representative, volumes are IEX-only and scaled ×25 to a
  consolidated-tape proxy for the capacity/slippage model. SIP (full tape) needs a
  paid plan.
- **Survivorship / ticker changes.** Pre-listing gaps are absent, not back-filled
  (ARB/OP launched in 2023). **POL** is the MATIC→POL rename: we stitch MATICUSDT
  (pre-Sep-2024) + POLUSDT into one continuous series via a general `symbol_stitch`
  mechanism; the ~3-day migration halt remains as honest missing data.
- **Cross-market alignment.** Crypto trades 24/7, equities 6.5h. Daily bars are
  snapshotted at the US close (last 1h bar ≤ 21:00 UTC) on the equity trading
  calendar (weekend crypto drift folds into the Fri→Mon return). Hourly bars join
  on the RTH core (14:00–20:00 UTC), where crypto and equity grids both align on
  the clock hour. A ≤1h DST offset is immaterial at daily frequency. Both daily and
  hourly panels derive from the **same 1h cache**, so they are mutually consistent.

---

## 4. Factor model (Task 2)

For each asset we fit, on a trailing window, the linear model
`r_i,t = α_i + Σ_f β_{i,f}·r_f,t + ε_i,t`. Factor sets are assigned by category:
crypto-equities → BTC+QQQ, miners → BTC+SMH, L1 (ETH) → BTC, L2/DeFi → ETH+BTC,
tech → QQQ+SMH.

- **Window: 60 bars (Avellaneda-Lee standard).** Sensitivity (§9) shows 60 > 90 >
  120: a shorter window adapts β faster and yields cleaner residuals; longer
  windows stale the hedge. 60 is both the literature default and the empirical
  best, so it is a defensible a-priori choice.
- **Estimation is point-in-time.** β/ε at bar `t` use only the window ending at
  `t`; the engine executes on `t+1`. A dedicated test perturbs future bars and
  asserts past β/ε are byte-identical.
- **Diagnostics.** Mean rolling R² ranges 0.38 (TSLA, idiosyncratic) to 0.69
  (ETH/ARB/MSTR). **Every one of the 25 residual series is stationary** (ADF
  p ≈ 0.00) — the mean-reversion premise holds universe-wide, which is the
  precondition for the strategy to have any edge.

---

## 5. Signal construction (Task 3)

- **Signal.** z-score of the *cumulative* residual `S_t = Σ ε` over a 60-bar
  trailing window — i.e. how rich/cheap the price is versus its factor-implied
  path (the Avellaneda-Lee residual-spread). This matches the economic narrative
  better than z-scoring daily residual returns (which we verify is near-martingale:
  lag-1 autocorr ≈ +0.01, no daily reversal edge; the *cumulative* spread reverts).
- **Entry/exit (brief's suggestion).** Enter (fade) at |z|>2.0, exit at |z|<0.5,
  with a sign-flip and a max-holding guard — a hysteresis state machine.
- **Mean-reversion-speed filter (enhancement, Avellaneda-Lee).** We also implement
  an AR(1) half-life gate: only open positions whose residual reverts within a set
  number of bars (κ filter). It screens out slow/non-reverting names — exactly the
  high-drift names that bleed alpha (§8). It improves gross PnL to positive but its
  parameter interaction is unstable (§9), so it is **off in the baseline** and
  reported as an analysed enhancement.

---

## 6. Portfolio construction & risk (Task 3)

- **Sizing: equal-volatility, fixed at entry.** `N_i = (target_vol / σ_resid_i) ·
  AUM/n_signals`, σ_resid being the trailing idiosyncratic vol (the right risk
  measure once factors are hedged). The notional is **frozen at entry** for the
  life of the trade — a desk sets size when it puts a position on, not by
  re-vol-targeting every bar (which would generate large fictitious turnover).
- **Hedging built from constrained legs → factor-neutral by construction.** Factor
  legs `= −Σ_i pos_i·β_{i,f}`.
- **Risk caps (all verified to hold on the *actually-held* book):** single asset
  ≤3% AUM, sector ≤15%, gross leverage ≤3×, **net factor exposure ≤5%**. The
  per-asset cap is enforced by *uniform* down-scaling (not clipping), because
  clipping a shared hedge leg would break neutrality. Realised net beta to every
  factor is ≈ 0.00.

---

## 7. Backtest engine & realistic execution (Task 4)

The engine is vectorised and strictly point-in-time: `held = positions.shift(1)` —
a target formed at `t` is executed and earns on `t+1`. This one-bar lag is the
whole no-lookahead guarantee and is enforced by tests.

**Realistic frictions modelled:**

- **Fees.** Binance perp taker 4 bps on traded notional.
- **Slippage — liquidity model (not flat).** `base + impact·√participation`, where
  `participation = trade$ / rolling ADV$` (square-root market-impact law). Thin
  names (CLSK) cost more than mega-caps (AAPL). ADV uses real archive volume
  (crypto) and IEX×25 (equity).
- **Funding.** USDⓈ-M perp funding from the downloaded 8h series — *signed*, so a
  short of a positive-funding perp earns a rebate (verified: funding is a credit on
  333 bars).
- **Borrow.** Short-equity borrow at 3%/yr annualised per bar.
- **No-trade band.** Rebalance an instrument only when its target drifts >0.15% AUM
  — a desk does not chase every tiny delta. This produces the *actually-held* book;
  the net-factor cap is enforced at 90% to leave the band drift headroom.

**Backtest-pitfall checklist (all addressed):** no lookahead (execution lag +
test); no survivorship bias (pre-listing absent, delistings configurable); time
alignment (single 1h cache, documented snapshot); costs fully modelled; results
reconcile exactly (`equity = AUM + Σ net pnl`, max diff 0).

---

## 8. Results & the central finding

**Performance (daily baseline):** Sharpe −0.20, CAGR −0.25%, vol 1.22%, max DD
−2.4%, Calmar −0.10, win-rate 45%, profit factor 0.96, avg holding ~17 days,
turnover 4.8×/yr. Annual net returns: 2023 −1.5%, 2024 +1.5%, 2025 −1.5%,
2026 (H1) +0.6% — small and regime-dependent.

**PnL decomposition — the heart of the analysis:**

| Component | PnL ($) | Reading |
|---|--:|---|
| Idiosyncratic edge `Σ held·ε` | **+595,924** | the reversion edge is real and large |
| Alpha-drift + hedge-error | **−560,680** | shorting drifting names + β-drift eat it |
| **Gross (tradeable)** | **+35,244** | thin positive |
| Fees | −83,793 | |
| Funding | −5,931 | |
| Borrow | −28,461 | |
| **Net** | **−82,941** | costs tip it negative |

**Why the alpha-drift drag exists.** The in-sample residual ε is mean-zero by
construction — it removes the asset's intercept/drift (α). A real hedge can remove
the *factor* exposure but **cannot remove the asset's own idiosyncratic drift**.
The strategy systematically shorts names *after* they have risen (z>2). In a bull
market those are precisely the names with persistent positive idiosyncratic drift,
so the shorts pay that drift. Per-asset, the reversion edge is positive on most
names (HOOD, AMZN, NVDA, AMD, AAVE…) but strongly negative on the hardest trenders
(LINK, AVAX, TSLA, ETH, MSTR) — the signature of reversion fighting momentum.

This is the deepest insight of the project: **a market-neutral residual-reversion
book is implicitly short idiosyncratic momentum/drift, and that is its main risk,
larger than transaction cost.** It motivates the mean-reversion-speed filter (only
trade fast-reverting residuals) and a shorter horizon (less time exposed to
drift).

---

## 9. Sensitivity & capacity analysis (Task 5)

Full grids are in `reports/sensitivity.md`. We sweep one knob at a time
(robustness, not optimisation). Highlights:

- **Entry threshold is the most important knob, and points to a cost hurdle.**
  Net rises monotonically with threshold: 2.0 (the brief's start, worst) → 3.0
  (Sharpe +0.28, net +$63k). Higher thresholds trade fewer, higher-conviction
  signals that clear costs. This is an economically sensible direction, not a lucky
  point.
- **Longer signal window helps** (zscore_window 60→90 moves net toward breakeven).
- **Factor window 60 ≫ 90 ≫ 120** (idio edge +$418k at 60) — confirms the
  AL-standard choice.
- **The half-life filter helps in isolation** (best ~HL 12) but its *interaction*
  with other knobs is unstable and non-monotonic — a warning against multi-knob
  optimisation. We therefore keep it off in the baseline.
- **Cost sensitivity: slippage barely matters, fees matter linearly.** Doubling
  the impact coefficient (12→24 bps) changes net <2%; the book is not impact-
  constrained at $10M. Halving the fee (4→2 bps) recovers ~$25k.
- **Capacity: Sharpe is flat from $10M to $200M.** Net scales ~linearly with AUM
  because participation stays tiny against the universe's ADV. The capacity ceiling
  for the *daily* strategy is well above $200M; the constraint is edge, not
  liquidity. (This directly answers the brief's $10–50M scalability requirement:
  liquidity is not the limiter.)

**Over-fitting guardrails.** We report the brief's suggested parameters as the
baseline, justify windows a-priori from the literature, sweep single knobs rather
than jointly optimising, and explicitly flag the unstable multi-knob region. We do
*not* present the cherry-picked Sharpe-0.44 combination as the result.

---

## 10. The hourly track (bonus)

Re-running at 1h (crypto-1h + Alpaca equity-1h, same pipeline, `frequency: hourly`):

| | Daily | Hourly |
|---|--:|--:|
| Idiosyncratic edge | +$596k | +$572k–$817k |
| Gross (tradeable) | +$35k | **+$130k (positive)** |
| Turnover (2-way/yr) | 4.8× | ~40× |
| Net | −$83k | −$0.8M (fees-dominated) |

The intraday gross edge is clearly positive and larger — consistent with alpha
drift being smaller over hours and microstructure reversion strongest at short
horizons. But hourly turnover makes **fees the binding constraint**, the mirror
image of the daily picture (where alpha-drift is the binding constraint). The
promising direction is therefore *intraday signal + aggressive turnover control*
(high threshold, throttled rebalancing, maker fills).

---

## 11. Where is the alpha, and is it sustainable? (business insight)

- **Alpha source:** liquidity provision to retail over-reaction in idiosyncratic,
  crypto-correlated equities and alts. It is real (positive gross, stationary
  residuals) but thin and contested.
- **Structural tailwind:** a unified cross-market venue removes the execution
  friction (one account, one margin, 24/7 hedging) that historically made this the
  preserve of multi-prime hedge funds — genuinely lowering the barrier.
- **Long-run ceiling:** (i) the edge competes with the asset's idiosyncratic
  momentum, which dominates in trending regimes; (ii) it is cost-sensitive at high
  frequency; (iii) as more participants run it, the residual reversion compresses.
  Capacity in *dollar* terms is high (>$200M daily, impact-light), but capacity in
  *Sharpe* terms is the real limit and is modest.
- **Honest verdict:** a viable *component* of a diversified market-neutral book —
  especially intraday with tight cost control and a drift/momentum overlay — but
  not a standalone strategy at daily frequency net of realistic costs over this
  sample.

---

## 12. Limitations & honest assessment

- **Net-negative baseline.** With the brief's parameters the daily strategy loses
  ~0.25%/yr after costs. We do not dress this up; the value is the diagnosis.
- **Single regime.** 2023-2025 was a strong crypto/tech bull — the worst regime
  for a book that is implicitly short idiosyncratic momentum. A bear/range sample
  would likely be kinder; we cannot test it with available history.
- **Equity data is IEX, not SIP.** Prices are representative for liquid names but
  volumes are a subset (scaled for the capacity model); thin names (CLSK/HOOD) are
  the least reliable.
- **Linear, static factor sets.** Betas are OLS and factor membership is fixed;
  no orthogonalisation of ETH against BTC (a small residual-exposure source).
- **POL** carries a 3-day migration gap; **borrow at a flat 3%** is an assumption.
- **Backtest, not live:** no queue position, partial fills, or borrow availability
  modelled beyond a flat rate.

---

## 13. Improvements & roadmap

1. **Drift/momentum overlay** (highest expected value): suppress or invert shorts
   on names with strong idiosyncratic momentum — directly attacks the −$561k drag.
2. **Intraday with turnover control:** the hourly gross edge is positive; pair it
   with maker fills, a wider no-trade band, and rebalance throttling.
3. **OU s-score with κ-filter** as the primary signal (partly built) for a cleaner,
   self-normalising reversion measure that gates on reversion speed.
4. **Orthogonalised / PCA factors** to remove the dual-factor (ETH/BTC) residual
   exposure and stabilise hedges.
5. **MATIC→POL full stitch tuning; SIP equity data; richer borrow/financing model.**
6. **Cross-market basis module** (Binance equity-perp vs spot) as an uncorrelated
   sleeve.

---

## 14. Reproducibility

Pinned dependencies (`pyproject.toml`); single global seed; idempotent, cached
downloads; one config file drives every run. `python scripts/download_data.py`
then `python scripts/run_backtest.py` reproduce the headline numbers exactly;
`python scripts/sensitivity.py` regenerates the grids; `pytest` (12 tests) covers
no-lookahead, risk constraints, costs, execution band, and the half-life filter.
Charts are in `reports/` (equity/drawdown, PnL attribution, factor exposure).
