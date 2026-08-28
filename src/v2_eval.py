"""v2 candidate evaluation per PROTOCOL.md.

Stage 1 (adoption decisions):  python3 -m src.v2_eval insample
    Runs every candidate with the simulation ENDING at 2021-12-31 — forward
    data is never simulated, so decisions cannot peek. Prints full-window
    in-sample metrics + the three sub-periods.

Stage 2 (single-shot validation): python3 -m src.v2_eval forward
    Runs the locked configs over the full span and reveals forward metrics.

Stage 3 (scenario analyses):     python3 -m src.v2_eval scenarios
    Capacity tiers and the plateau portfolio (rules fixed in PROTOCOL.md).
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
SUBPERIODS = [("2000-01-01", "2007-12-31"), ("2008-01-01", "2014-12-31"),
              ("2015-01-01", "2021-12-31")]

# C3 friction realism (fixed in PROTOCOL.md before evaluation)
SCHED_COST = ((0, 0.005), (2010, 0.003), (2020, 0.002))
SCHED_CASH = ((0, 0.075), (2003, 0.060), (2009, 0.045), (2010, 0.075),
              (2015, 0.065), (2020, 0.035), (2022, 0.065))

C3 = {"cost_schedule": SCHED_COST, "cash_schedule": SCHED_CASH}

INSAMPLE_CONFIGS = {
    "v1_flat_frictions": {},
    "baseline_C3": {**C3},
    "C1_sleeve5": {**C3, "sleeve_m": 5},
    "C1_sleeve10": {**C3, "sleeve_m": 10},
    "C1_sleeve15": {**C3, "sleeve_m": 15},
    "C2_taper1.25": {**C3, "weighting": "ranklin", "taper_k": 1.25},
    "C2_taper1.5": {**C3, "weighting": "ranklin", "taper_k": 1.5},
    "C2_taper2.0": {**C3, "weighting": "ranklin", "taper_k": 2.0},
    "C1+C2_mid": {**C3, "sleeve_m": 10, "weighting": "ranklin", "taper_k": 1.5},
}


def slice_metrics(eq, a, b):
    e = eq[(eq.index >= a) & (eq.index <= b)]
    if len(e) < 50:
        return {}
    e = e / e.iloc[0]
    return {"cagr": metrics.cagr(e), "maxdd": metrics.max_drawdown(e),
            "sharpe": metrics.sharpe(e)}


def stage_insample():
    mkt = Market()
    rows = []
    for label, ov in INSAMPLE_CONFIGS.items():
        p = Params(**ov)
        res = run(mkt, p, IS_START, IS_END)      # simulation stops at 2021
        eq = res.equity
        m = slice_metrics(eq, IS_START, IS_END)
        row = {"label": label,
               "is_cagr": m["cagr"], "is_maxdd": m["maxdd"], "is_sharpe": m["sharpe"],
               "sleeve_avg": float(res.sleeve_exposure.mean()),
               "turnover_py": float(res.turnover_total / 2.0 / 22.0)}
        for k, (a, b) in enumerate(SUBPERIODS, 1):
            sm = slice_metrics(eq, a, b)
            row[f"sub{k}_cagr"] = sm.get("cagr")
            row[f"sub{k}_maxdd"] = sm.get("maxdd")
        rows.append(row)
        mkt.mom_cache.clear()
        print(label, "done", flush=True)
    df = pd.DataFrame(rows).set_index("label")
    df.to_csv(os.path.join(RESULTS, "v2_insample.csv"))
    print(df.round(3).to_string())


def stage_forward(final_configs: dict):
    mkt = Market()
    rows = []
    for label, ov in final_configs.items():
        p = Params(**ov)
        res = run(mkt, p, IS_START, str(mkt.dates[-1].date()))
        eq = res.equity
        mi = slice_metrics(eq, IS_START, IS_END)
        mf = slice_metrics(eq, FWD_START, "2099-01-01")
        mfull = slice_metrics(eq, IS_START, "2099-01-01")
        rows.append({
            "label": label,
            "is_cagr": mi["cagr"], "is_maxdd": mi["maxdd"], "is_sharpe": mi["sharpe"],
            "fw_cagr": mf["cagr"], "fw_maxdd": mf["maxdd"], "fw_sharpe": mf["sharpe"],
            "full_cagr": mfull["cagr"], "full_maxdd": mfull["maxdd"],
            "sleeve_avg": float(res.sleeve_exposure.mean()),
        })
        mkt.mom_cache.clear()
        print(label, "done", flush=True)
    df = pd.DataFrame(rows).set_index("label")
    df.to_csv(os.path.join(RESULTS, "v2_forward.csv"))
    print(df.round(3).to_string())


def stage_scenarios(v2: dict):
    mkt = Market()
    rows = []
    tiers = [
        ("pool300", {**v2}),
        ("pool500_+5bp", {**v2, "top_turnover": 500,
                          "cost_schedule": ((0, 0.0055), (2010, 0.0035), (2020, 0.0025))}),
        ("pool800_+15bp", {**v2, "top_turnover": 800,
                           "cost_schedule": ((0, 0.0065), (2010, 0.0045), (2020, 0.0035))}),
    ]
    for label, ov in tiers:
        p = Params(**ov)
        res = run(mkt, p, IS_START, str(mkt.dates[-1].date()))
        eq = res.equity
        mi = slice_metrics(eq, IS_START, IS_END)
        mf = slice_metrics(eq, FWD_START, "2099-01-01")
        rows.append({"label": label, "is_cagr": mi["cagr"], "is_maxdd": mi["maxdd"],
                     "fw_cagr": mf["cagr"], "fw_maxdd": mf["maxdd"]})
        mkt.mom_cache.clear()
        print(label, "done", flush=True)
    df = pd.DataFrame(rows).set_index("label")
    df.to_csv(os.path.join(RESULTS, "v2_capacity_tiers.csv"))
    print(df.round(3).to_string())

    # plateau portfolio: every OAT strict passer excluding friction axes,
    # equal-weight average of daily returns, v1 flat frictions (as swept)
    oat = pd.read_csv(os.path.join(RESULTS, "sweep_oat.csv"))
    strict = oat[(oat["is_cagr"] > 0.30) & (oat["is_maxdd"] > -0.25)]
    strict = strict[~strict["label"].str.startswith(("cost_per_side", "cash_rate"))]
    print(f"\nplateau portfolio members: {len(strict)}")
    rets = []
    for _, r in strict.iterrows():
        ov = json.loads(r["overrides"])
        if "mom_lookbacks" in ov:
            ov["mom_lookbacks"] = tuple(
                int(x) for x in str(ov["mom_lookbacks"]).strip("()[]").split(",")
                if str(x).strip())
        p = Params(**ov)
        res = run(mkt, p, IS_START, str(mkt.dates[-1].date()))
        rets.append(res.equity.pct_change().fillna(0.0))
        mkt.mom_cache.clear()
        mkt.sigma_cache.clear()
        print("  member", r["label"], "done", flush=True)
    avg = pd.concat(rets, axis=1).mean(axis=1)
    eq = (1 + avg).cumprod()
    mi = slice_metrics(eq, IS_START, IS_END)
    mf = slice_metrics(eq, FWD_START, "2099-01-01")
    out = {"members": len(strict),
           "is_cagr": mi["cagr"], "is_maxdd": mi["maxdd"], "is_sharpe": mi["sharpe"],
           "fw_cagr": mf["cagr"], "fw_maxdd": mf["maxdd"], "fw_sharpe": mf["sharpe"]}
    with open(os.path.join(RESULTS, "v2_plateau_portfolio.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "insample"
    if mode == "insample":
        stage_insample()
    elif mode == "forward":
        with open(os.path.join(RESULTS, "v2_locked.json")) as f:
            final = json.load(f)
        final = {k: _untuple(v) for k, v in final.items()}
        stage_forward(final)
    elif mode == "scenarios":
        with open(os.path.join(RESULTS, "v2_locked.json")) as f:
            final = json.load(f)
        stage_scenarios(_untuple(final["v2"]))


def _untuple(ov: dict) -> dict:
    out = dict(ov)
    for k in ("cost_schedule", "cash_schedule"):
        if k in out:
            out[k] = tuple(tuple(x) for x in out[k])
    return out


if __name__ == "__main__":
    main()
