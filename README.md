# All-Weather Momentum — NSE cash equities

A long-only, unlevered momentum-rotation strategy on Indian (NSE) cash
equities, built on free Yahoo Finance data (`yfinance`), with a layered risk
stack designed so that the *whole parameter neighborhood* works — a plateau,
not a tuned point.

**Mandate:** CAGR > 30% and max drawdown < 25% over 2000–2021, then forward
tested on 2022 → today (2026-08-28).

## Headline results (base config, costs 0.30%/side)

| Window | CAGR | Max DD | Vol | Sharpe | Calmar | Sensex CAGR | Sensex MaxDD |
|---|---|---|---|---|---|---|---|
| **2000–2021 (in-sample)** | **31.1%** ✅ | **−24.7%** ✅ | 15.8% | 1.81 | 1.26 | 11.4% | −60.9% |
| **2022–today (forward)** | **13.2%** ❌ | **−18.0%** ✅ | 15.5% | 0.89 | 0.73 | 5.9% | −16.2% |
| 2000–today (full) | 27.9% | −24.7% | 15.8% | 1.65 | 1.11 | 10.5% | −60.9% |

₹1 in Jan 2000 → ≈ ₹700 by Aug 2026 (Sensex: ≈ ₹14). Only 6 negative
calendar years out of 27; worst year −11.4% (2008, when the Sensex fell 52%).

**The honest verdict on the forward test:** the *drawdown* half of the mandate
holds out-of-sample with room to spare. The *CAGR* half does not — and the
sweep shows no configuration of this design family passes in-sample and also
delivers 30% in 2022–2026 (the few configs above 30% forward all carry
in-sample drawdowns of −28% to −46%, i.e. they'd have been rejected in 2021).
2022–2026 simply did not pay 30%/yr to any honest unlevered long-only Indian
equity book: the Sensex compounded at 5.9%, and 2024 H2 → 2026 was a momentum
winter. The strategy's +7.3pp/yr edge over the index, with a Sharpe near 0.9
and max DD of −18%, is what survival of the approach actually looks like —
not a repeat of the in-sample number.

## Strategy

Universe: every currently listed NSE main-board (EQ-series) stock —
~2,060 usable symbols, no index membership required.

**Selection (monthly, last close of month; all trades execute at the next
day's close):**
1. Eligible = trading today, ≥1 year of history, top-300 by 63-day median
   rupee turnover (point-in-time), price > ₹10, above its own 200-day SMA.
2. Score = mean(3, 6, 12-month returns, each skipping the latest month) ÷
   6-month realized vol. Must be positive.
3. Hold top-15 equal-weighted, with a rank buffer: a holding stays while it
   remains in the top 30 by score (halves turnover).

**Risk stack (exposure = product of four scalars, evaluated daily):**
1. *Index trend gate (0/1):* Sensex above its 200-day SMA (1% hysteresis
   band) AND not >10% below its 63-day high (fast-crash breaker — catches
   May-2004 / May-2006 / Jan-2008 style crashes that happen far above the
   200-day line; self-healing as the rolling high decays).
2. *Breadth scalar (0→1):* share of the liquid pool above their own 200-day
   SMA, mapped linearly 25% → 0, 55% → 1 (5-day smoothed). Sizes the book by
   market internals; catches mid/small bears the large-cap index never shows
   (2013, 2018–19, 2025).
3. *Vol targeting:* min(1, 25% ÷ EWMA vol of the book, λ=0.94). A crash
   brake that cuts within days when volatility explodes.
4. *Equity-curve brake:* while strategy equity is below its own 200-day SMA,
   exposure × 0.5 — catches momentum-factor winters that market-level
   signals cannot see (2019).

**Stock-level protection:** 25% trailing stop vs the post-entry peak close;
positions that stop trading for >10 sessions are force-exited at last price
−10%. Freed weight stays in cash until the next monthly rotation.

**Frictions & cash:** 0.30% per side on all traded notional (≈ 4× one-way
turnover/yr); idle cash earns 6%/yr (Indian liquid-fund / T-bill proxy — at
0% cash yield the in-sample CAGR is 27.2%).

No leverage, no derivatives, no shorting, ever. Exposure is only ever ≤ 100%.

## Why these layers exist (each maps to a real failure)

A naive version of this strategy (same selection, Sensex-200DMA gate only)
does 28% CAGR with a **−45% drawdown**. The stack was built by diagnosing
each historical failure of the naive design — not by tuning numbers:

| Failure mode | Episode it caused | Layer that fixes it |
|---|---|---|
| Index gate reacts too slowly to fast crashes from highs | 2004, 2006, 2008: −25→−45% | fast-crash breaker + vol targeting |
| Large-cap index blind to broad mid/small bears | 2018–19: −37% while Sensex made highs | breadth scalar |
| Binary regime flips re-enter at full size into bears | whipsaw bleed 2000-02, 2011, 2019 | continuous breadth scalar + hysteresis |
| Momentum-factor winter while market looks healthy | 2019 | equity-curve brake |
| Single-stock blowups (frauds, defaults) | Vakrangee/DHFL-class events | trailing stop |
| Missing the recovery after a crash | Apr–May 2009 | graded (not binary) re-entry via scalars |

Ablations (everything else at base): no breadth → DD −33%; no equity brake →
−31%; no vol target → −29%; no fast-crash breaker → −27%; no trailing stop →
−28%; **no risk stack at all → CAGR 34% but DD −77%**. The stack costs ~3pp
of CAGR and removes ~52pp of drawdown.

## Plateau evidence (`results/sweep_oat.csv`, `results/sweep_random.csv`)

- **One-at-a-time:** 65 configs varying every parameter around base across
  wide ranges (N 10–30; lookbacks 3–12m single & ensembles; skip 0–42d; pool
  150–800; SMAs 100–300; vol target 18–30% & off; stops 20–35% & off; costs
  0.15–0.5%/side; etc.). 55/65 land in CAGR > 27% & DD < 28%; 27/65 pass the
  strict mandate; **all 65 have Sharpe ≥ 1.37**. The failures are informative,
  not random: pool ≤ 200 (too narrow for the breadth signal + 15 slots) and
  disabled risk layers.
- **Joint random draws:** 200 configs sampled uniformly over the full
  hypercube — deliberately including crippled corners (risk layers off).
  Every single one of the 200 has in-sample Sharpe ≥ 1.06 (median 1.67); the
  worst-drawdown tail is dominated by configs with layers disabled, exactly
  as designed. Among configs with all risk layers armed, 91% keep the
  forward-window drawdown under 25%.
- **No date luck:** rebalancing 10 sessions after month-end instead: 31.4% /
  −22.8%.
- **The deep-pool upside case:** widening the liquidity pool from 300 to 800
  names (one parameter) gives 33.7% / −22.4% in-sample and **27.8% / −18.4%
  forward** — the only region of the design space that approached the 30%
  bar in 2022–2026 is the small/micro-cap end, which is also where
  survivorship bias, slippage, and capacity constraints bite hardest. Among
  all 200 random configs exactly one passed the full mandate in both windows
  (36.8% / −24.5% in-sample, 34.1% / −18.5% forward) — a pool-800,
  inverse-vol, 6&12-month variant. One draw out of 200 is a lottery ticket,
  not a design property; it is reported here for completeness, not chosen.
- The base config's numbers sit inside the plateau, not on a spike: the two
  headline metrics move smoothly along every axis.

## What to be honest about

1. **Survivorship bias (the big one).** Yahoo only serves currently listed
   symbols; companies delisted 2000–2026 are invisible. Mitigations: the
   universe is *all* ~2,060 listed stocks (today's losers included — not an
   index's winners); eligibility is computed point-in-time; a trailing stop
   limits blowup paths; and the forward window (2022+) is nearly bias-free —
   it is the credible part of the evidence. But the in-sample CAGR is
   certainly flattered by this, and the momentum premium here concentrates in
   mid/small caps, where delistings are most common. A large-cap-only variant
   (top-100 turnover) does 19.2% / −31% — the 31% headline does not survive
   restriction to the most bias-resistant pool. Treat in-sample CAGR as an
   upper estimate. A same-bias internal benchmark (passive equal-weight of
   the same pool: 17.0% / −72%) shows the *rules* add ~14pp/yr and remove
   ~47pp of drawdown over the identical biased universe.
2. **Data quality.** Yahoo NSE history contains unadjusted splits/bonuses and
   bad prints (hundreds of fake ±50–2000% days, concentrated 2003–2007). A
   conservative repair layer (`src/panels.py`) fixes spike-reversals and
   exact-fraction cliffs, and caps residual one-day *gains* at +40% while
   never capping losses; genuine crashes (Yes Bank −56%) are preserved.
   640 repairs across 26 years, log in `data/panels/cleaning_log.csv`.
3. **Thin early universe.** Yahoo's NSE coverage before 2002 is ~60–90
   symbols (large caps), so 2000–2002 is effectively a large-cap backtest.
   Coverage is ~480 by 2003 and 900+ by 2010.
4. **Cash yield.** 6%/yr on idle cash (≈ the 2000–2021 T-bill average) is
   realistic for liquid funds but adds ~4pp CAGR vs assuming 0%.
5. **Capacity.** Top-300-turnover mid/small caps at monthly rotation is fine
   for personal capital, not for a large fund. Pool 500–800 performs even
   better in-sample and forward but drifts further down the liquidity curve.
6. **2026 data** comes from Yahoo as-is and covers a live, unfinished year.

## Reproduce

```bash
pip install -r requirements.txt
python3 -m src.download_data   # ~15 min: full NSE universe from Yahoo
python3 -m src.panels          # clean + consolidate into panels
python3 -m src.run_base        # headline backtest + results/*.csv
python3 -m src.run_base --sanity  # show what it would have picked, 2003-2026
python3 -m src.variants        # ablations & bias checks
python3 -m src.sweep oat       # 65-config sensitivity sweep
python3 -m src.sweep random 200 42  # 200-config joint plateau sample
python3 -m src.plots           # charts
```

`src/yf_compat.py` routes yfinance through a Safari TLS fingerprint —
needed behind TLS-re-terminating proxies that reset Chrome impersonation.

## Files

```
src/download_data.py  NSE universe + full-history downloader (checkpointed)
src/panels.py         data-repair layer + wide panels (the cleaning rules)
src/engine.py         the strategy: Params, Market (signals), run() (simulator)
src/metrics.py        CAGR / drawdown / Sharpe / yearly returns
src/run_base.py       headline runs + probe mode
src/variants.py       ablations, survivorship stress tests, date-shift test
src/sweep.py          OAT + random-joint parameter sweeps (parallel)
src/plots.py          report charts
results/              all outputs (CSV/JSON/PNG) committed for inspection
```

*This is research, not investment advice. Past performance, especially
backtested performance on survivorship-biased data, does not predict future
returns.*
