# Pre-registered protocol: v2 candidates to improve forward CAGR

Committed BEFORE any candidate is evaluated (see git history). Purpose: raise
the strategy's realized CAGR — including the 2022+ forward window — without
using the forward window to make any design decision.

## Contamination disclosure

The v1 sweeps already revealed forward-window results for every existing
parameter axis (pool size, lookbacks, SMAs, vol targets, stops, …). Any
repositioning along those axes is therefore contaminated — most notably
`top_turnover=800`, which is known to have forward CAGR ≈ 28%. Consequently:

- **No swept axis may be repositioned** on performance grounds. Pool depth is
  treated purely as a *capacity scenario* (reported with tiered costs, never
  adopted as "the" config).
- Candidates below are **new mechanisms** that have never been evaluated on
  the forward window. Their in-sample results are also unknown at commit time.

## Candidates (with a-priori rationale, fixed before evaluation)

**C1 — Defensive sleeve during factor winter.** Currently, when the
equity-curve brake halves exposure (strategy below its own 200d SMA) the freed
capital sits in cash even when the *market* gates are healthy. Rationale:
momentum winters within healthy markets historically coincide with
low-volatility/defensive leadership (factor-rotation literature; India 2019 is
the canonical case). Mechanism: the exposure removed by the equity brake —
`gate × vol_scalar × (1 − eq_scalar)` — is invested in the top-M lowest-vol
liquid-pool names that are above their own 200d SMA, equal weight, refreshed
monthly, same trailing stops. Cash remains the destination whenever the market
gates (trend/breadth/vol) cut exposure. Mini-sweep: M ∈ {5, 10, 15}.

**C2 — Rank-tapered weights.** Equal weight ignores that expected momentum
return is monotone in rank (documented in-sample and in the literature; the
existing `invvol` axis does not express this). Mechanism: weight ∝
(1.5·N − rank), capped at 10%, renormalized (top ≈ 2.9× bottom).
Mini-sweep: taper K ∈ {1.25N, 1.5N, 2N}.

**C3 — Friction realism (measurement correction, not a tuning knob).** Flat
0.30%/side and flat 6% cash are period-averaged approximations. Replace with
piecewise schedules reflecting documented history — costs/side: 0.50% before
2010, 0.30% 2010–2019, 0.20% from 2020 (discount-broker era; STT 0.1% + stamp
+ impact on top-300 names); cash: 7.5% 2000–02, 6.0% 2003–08, 4.5% 2009,
7.5% 2010–14, 6.5% 2015–19, 3.5% 2020–21, 6.5% 2022+ (approximate 91-day
T-bill averages). Adopted on realism grounds regardless of which direction it
moves results; both directions are reported.

## Adoption rule (fixed now)

Evaluate C1 and C2 on 2000–2021 ONLY (under C3 frictions), plus in-sample
sub-periods 2000–2007, 2008–2014, 2015–2021. Adopt a candidate iff:

1. In-sample CAGR improves by ≥ +0.5pp, or MaxDD improves by ≥ +1pp with
   CAGR within −0.5pp;
2. In-sample MaxDD stays < 25%;
3. Sub-period stability: in-sample CAGR does not fall more than 1pp below the
   comparable baseline in more than one of the three sub-periods;
4. Its mini-sweep is a plateau: every swept variant keeps in-sample
   CAGR within ±2pp of the candidate's and MaxDD < 27%.

The v2 config = baseline + all adopted candidates (mid-grid values, never the
best cell). Only THEN is the forward window run, once, for v2 and for each
candidate individually — results reported verbatim, adopted or not, better or
worse. No candidate is un-adopted or re-parameterized after seeing forward
results; if v2 fails forward, that is the reported finding.

## Also reported (analysis, not adoption)

- **Capacity tiers:** pool 300/500/800 under C3 plus an impact surcharge for
  deeper pools (+0.05pp/side at 500, +0.15pp/side at 800), both windows —
  labeled as scenario analysis inheriting the known-forward contamination.
- **Plateau portfolio:** equal-weight ensemble of ALL v1 OAT configs that
  passed the strict in-sample mandate (excluding friction-axis variants) —
  a zero-discretion rule fixed here; its forward number is computed after.

## Regression guard

Engine changes must reproduce v1 base results exactly when new features are
disabled (base_report.json CAGR/MaxDD to 6 decimals) before any evaluation.
