"""Parameter-plateau analysis.

Two experiments, both run over the FULL 2000->today window and reported
separately on the in-sample (2000-2021) and forward (2022->today) slices:

  1. OAT ("one-at-a-time"): every parameter varied across a wide range around
     the base config while all others stay at base. Shows per-parameter
     sensitivity.
  2. RANDOM: N configs drawn uniformly from the full joint neighborhood.
     Shows the joint surface is a plateau, not a spike.

Usage:
    python3 -m src.sweep oat
    python3 -m src.sweep random [n] [seed]
    python3 -m src.sweep worker <infile> <outfile>   (internal)
"""
import itertools
import json
import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd

from . import metrics
from .engine import Market, Params, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
IS_START, IS_END = "2000-01-01", "2021-12-31"
FWD_START = "2022-01-01"

# every axis of the design and the range explored
AXES = {
    "n_stocks": [10, 12, 15, 20, 25, 30],
    "buffer_mult": [1.0, 1.5, 2.0, 3.0],
    "mom_lookbacks": [(63, 126, 252), (126, 252), (252,), (63, 126), (126,), (189,)],
    "mom_skip": [0, 10, 21, 42],
    "sigma_window": [63, 126, 252],
    "top_turnover": [150, 200, 300, 500, 800],
    "stock_sma": [100, 150, 200, 250],
    "weighting": ["equal", "invvol"],
    "regime_ma": [150, 200, 250, 300],
    "idx_dd_exit": [0.0, 0.08, 0.10, 0.12, 0.15],
    "idx_dd_window": [42, 63, 126],
    "breadth_lo": [0.15, 0.20, 0.25, 0.30],
    "breadth_hi": [0.45, 0.50, 0.55, 0.60, 0.65],
    "vol_target": [0.0, 0.18, 0.22, 0.25, 0.30],
    "vol_lambda": [0.90, 0.92, 0.94, 0.97],
    "eq_brake_ma": [0, 100, 150, 200, 300],
    "eq_brake_scalar": [0.3, 0.5, 0.7],
    "crash_stop": [0.0, 0.20, 0.25, 0.30, 0.35],
    "cost_per_side": [0.0015, 0.003, 0.005],
    "cash_rate": [0.0, 0.04, 0.06],
    "use_breadth": [True, False],
}


def evaluate(mkt: Market, p: Params) -> dict:
    res = run(mkt, p, IS_START, str(mkt.dates[-1].date()))
    eq = res.equity
    is_eq = eq[eq.index <= IS_END]
    is_eq = is_eq / is_eq.iloc[0]
    fw_eq = eq[eq.index >= FWD_START]
    fw_eq = fw_eq / fw_eq.iloc[0]
    return {
        "is_cagr": metrics.cagr(is_eq),
        "is_maxdd": metrics.max_drawdown(is_eq),
        "is_sharpe": metrics.sharpe(is_eq),
        "fw_cagr": metrics.cagr(fw_eq),
        "fw_maxdd": metrics.max_drawdown(fw_eq),
        "fw_sharpe": metrics.sharpe(fw_eq),
        "full_cagr": metrics.cagr(eq / eq.iloc[0]),
        "full_maxdd": metrics.max_drawdown(eq / eq.iloc[0]),
        "avg_exposure": float(res.exposure.mean()),
        "turnover_py": float(res.turnover_total / 2.0
                             / ((eq.index[-1] - eq.index[0]).days / 365.25)),
    }


def config_rows_oat():
    rows = [("base", {})]
    for axis, values in AXES.items():
        base_val = getattr(Params(), axis)
        for v in values:
            if v == base_val:
                continue
            rows.append((f"{axis}={v}", {axis: v}))
    return rows


def config_rows_random(n: int, seed: int):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        overrides = {}
        for axis, values in AXES.items():
            if axis in ("cost_per_side", "cash_rate"):
                continue  # frictions are scenario inputs, not design choices
            overrides[axis] = rng.choice(values)
        rows.append((f"rand{i:04d}", overrides))
    return rows


def run_configs(rows, out_csv):
    mkt = Market()
    out = []
    for i, (label, overrides) in enumerate(rows):
        p = Params(**overrides)
        try:
            m = evaluate(mkt, p)
        except Exception as e:
            m = {"error": str(e)[:120]}
        # momentum matrices are ~130MB per (lookback, skip); purge between
        # configs or 4 parallel workers OOM the container
        if len(mkt.mom_cache) > 3:
            mkt.mom_cache.clear()
        if len(mkt.sigma_cache) > 3:
            mkt.sigma_cache.clear()
        m["label"] = label
        m["overrides"] = json.dumps(overrides, default=str)
        out.append(m)
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            pd.DataFrame(out).to_csv(out_csv, index=False)
            print(f"{i+1}/{len(rows)} done", flush=True)
    pd.DataFrame(out).to_csv(out_csv, index=False)


def run_parallel(rows, out_csv, n_proc=4):
    os.makedirs(RESULTS, exist_ok=True)
    tmpdir = os.path.join(RESULTS, "_sweep_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    chunks = [rows[i::n_proc] for i in range(n_proc)]
    procs = []
    for k, chunk in enumerate(chunks):
        infile = os.path.join(tmpdir, f"in_{k}.json")
        outfile = os.path.join(tmpdir, f"out_{k}.csv")
        with open(infile, "w") as f:
            json.dump(chunk, f, default=str)
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "src.sweep", "worker", infile, outfile],
            cwd=ROOT))
    for pr in procs:
        pr.wait()
    frames = [pd.read_csv(os.path.join(tmpdir, f"out_{k}.csv"))
              for k in range(n_proc)]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(df)} rows)")


def _fix_types(overrides: dict) -> dict:
    out = {}
    for k, v in overrides.items():
        if k == "mom_lookbacks":
            v = tuple(v) if isinstance(v, (list, tuple)) else tuple(
                int(x) for x in str(v).strip("()").split(",") if x.strip())
        out[k] = v
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "oat"
    if mode == "worker":
        infile, outfile = sys.argv[2], sys.argv[3]
        with open(infile) as f:
            rows = [(label, _fix_types(ov)) for label, ov in json.load(f)]
        run_configs(rows, outfile)
        return
    if mode == "oat":
        rows = config_rows_oat()
        print(f"OAT sweep: {len(rows)} configs")
        run_parallel(rows, os.path.join(RESULTS, "sweep_oat.csv"))
    elif mode == "random":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
        rows = config_rows_random(n, seed)
        out_csv = os.path.join(RESULTS, "sweep_random.csv")
        if os.path.exists(out_csv):  # resume: only rows not yet computed
            have = set(pd.read_csv(out_csv)["label"])
            missing = [r for r in rows if r[0] not in have]
            if missing:
                part = os.path.join(RESULTS, "sweep_random_part2.csv")
                print(f"resuming: {len(missing)} missing configs")
                run_parallel(missing, part, n_proc=2)
                merged = pd.concat([pd.read_csv(out_csv), pd.read_csv(part)],
                                   ignore_index=True)
                merged.drop_duplicates("label").to_csv(out_csv, index=False)
                print(f"merged -> {out_csv} ({len(merged)} rows)")
            else:
                print("nothing missing")
            return
        print(f"random sweep: {len(rows)} configs (seed {seed})")
        run_parallel(rows, out_csv)


if __name__ == "__main__":
    main()
