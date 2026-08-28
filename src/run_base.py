"""Run the a-priori base configuration and report in-sample + forward metrics.

Usage:
    python3 -m src.run_base            # full run + metrics + CSV outputs
    python3 -m src.run_base --sanity   # print selections at probe dates
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import metrics
from .engine import Market, Params, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

IS_START, IS_END = "2000-01-01", "2021-12-31"
FWD_START = "2022-01-01"


def window(equity: pd.Series, start=None, end=None) -> pd.Series:
    e = equity
    if start:
        e = e[e.index >= start]
    if end:
        e = e[e.index <= end]
    return e / e.iloc[0]


def report(mkt: Market, res, label: str) -> dict:
    eq = res.equity
    is_eq = window(eq, IS_START, IS_END)
    fw_eq = window(eq, FWD_START, None)
    full = window(eq)
    rows = {
        "in_sample_2000_2021": metrics.summary(is_eq, label),
        "forward_2022_today": metrics.summary(fw_eq, label),
        "full_2000_today": metrics.summary(full, label),
    }
    years = float((eq.index[-1] - eq.index[0]).days) / 365.25
    rows["diagnostics"] = {
        "avg_exposure": float(res.exposure.mean()),
        "avg_gate": float(res.gate.mean()),
        "avg_vol_scalar": float(res.vol_scalar.mean()),
        "avg_names_when_invested": float(res.n_held[res.n_held > 0].mean()),
        "one_way_turnover_per_year": float(res.turnover_total / 2.0 / years),
        "total_cost_drag": float(res.cost_total),
        "n_rebalances": res.n_rebalances,
        "n_crash_stops": res.n_crash_stops,
        "n_forced_exits": res.n_forced_exits,
    }
    return rows


def benchmarks(mkt: Market) -> dict:
    out = {}
    for name, df in mkt.indices.items():
        s = df["Close"]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        for wlabel, a, b in [
            ("in_sample_2000_2021", IS_START, IS_END),
            ("forward_2022_today", FWD_START, None),
            ("full_2000_today", IS_START, None),
        ]:
            e = s[s.index >= a]
            if b:
                e = e[e.index <= b]
            if len(e) < 100:
                continue
            e = e / e.iloc[0]
            out[f"{name}:{wlabel}"] = metrics.summary(e, name)
    return out


def sanity(mkt: Market):
    p = Params()
    probes = ["2003-12-31", "2007-12-31", "2013-12-31", "2017-12-29",
              "2020-12-31", "2023-12-29", "2025-12-31", "2026-08-14"]
    sig = mkt.sigma(p.sigma_window)
    moms = [mkt.mom(k, p.mom_skip) for k in p.mom_lookbacks]
    sma_stock = mkt.sma(p.stock_sma)
    for probe in probes:
        d = int(mkt.dates.searchsorted(pd.Timestamp(probe), side="right")) - 1
        with np.errstate(invalid="ignore", divide="ignore"):
            score_num = np.mean(np.stack([m[d] for m in moms]), axis=0)
            sg = np.maximum(sig[d], p.sigma_floor)
            score = score_num / sg
            elig = (
                mkt.valid[d]
                & (mkt.obs_total[d] >= p.min_history)
                & (mkt.obs252[d] >= p.min_obs_252)
                & (mkt.C[d] > p.price_floor)
                & np.isfinite(score)
                & (score_num > 0)
                & (mkt.AC[d] > sma_stock[d])
            )
            turn = np.where(np.isfinite(mkt.med_turn[d]), mkt.med_turn[d], -1.0)
            elig &= turn > 0
            if elig.sum() > p.top_turnover:
                thresh = np.partition(turn[elig], -p.top_turnover)[-p.top_turnover]
                elig &= turn >= thresh
        idx_e = np.where(elig)[0]
        order = idx_e[np.argsort(-score[idx_e])][:20]
        names = [mkt.syms[s] for s in order]
        print(f"\n{mkt.dates[d].date()}  eligible={len(idx_e)}  top20:")
        print("  " + ", ".join(names))


def main():
    mkt = Market()
    print(f"panel: {len(mkt.dates)} days x {mkt.n_sym} syms, "
          f"{mkt.dates[0].date()} -> {mkt.dates[-1].date()}")
    if "--sanity" in sys.argv:
        sanity(mkt)
        return

    os.makedirs(RESULTS, exist_ok=True)
    p = Params()
    res = run(mkt, p, IS_START, str(mkt.dates[-1].date()))
    rep = report(mkt, res, "base")
    rep["benchmarks"] = benchmarks(mkt)
    rep["params"] = {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in vars(p).items()}

    print(json.dumps(rep, indent=2, default=str))
    with open(os.path.join(RESULTS, "base_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)

    out = pd.DataFrame({
        "equity": res.equity,
        "exposure": res.exposure,
        "n_held": res.n_held,
        "gate": res.gate,
        "vol_scalar": res.vol_scalar,
    })
    out.to_csv(os.path.join(RESULTS, "base_equity_daily.csv"))

    yr = metrics.yearly_returns(res.equity)
    sx = mkt.indices["SENSEX"]["Close"]
    sx = sx[~sx.index.duplicated(keep="last")].sort_index()
    sx = sx[(sx.index >= res.equity.index[0]) & (sx.index <= res.equity.index[-1])]
    yr_b = metrics.yearly_returns(sx)
    pd.DataFrame({"strategy": yr, "sensex": yr_b}).to_csv(
        os.path.join(RESULTS, "base_yearly_returns.csv"))
    print("\nyearly returns:")
    print(pd.DataFrame({"strategy": yr.round(3), "sensex": yr_b.round(3)}))


if __name__ == "__main__":
    main()
