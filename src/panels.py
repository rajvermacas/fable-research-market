"""Consolidate per-symbol parquets into aligned wide panels for the backtest.

Panels (date x symbol, float32):
    adj_close   dividend/split-adjusted close (total-return proxy)
    close       unadjusted close (for turnover, price floors)
    volume      shares traded

Master calendar = trading days on which at least MIN_ACTIVE symbols traded.
Saved to data/panels/.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PANEL_DIR = os.path.join(DATA, "panels")
MIN_ACTIVE = 5

# Yahoo NSE data contains two classes of defect:
#   1. bad prints: a price spikes (or craters) and reverts within a few days
#   2. missing corporate-action adjustments: an unadjusted split/bonus shows
#      up as a permanent one-day cliff at an (almost) exact simple fraction
# Repairs below are deliberately conservative: fraction tolerance is tight so
# genuine crashes (Yes Bank -56%, Satyam-style collapses) are NOT repaired,
# and residual daily GAINS are capped (+40%) while losses are never capped.
SPLIT_FRACTIONS = np.array(
    [1/2, 2/5, 1/3, 3/10, 1/4, 1/5, 1/6, 1/8, 1/10, 1/12, 1/16, 1/20, 1/25, 1/50]
)
REV_FACTORS = np.array([2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 50])
DOWN_TOL = 0.08   # missed-split proximity tolerance
UP_TOL = 0.12     # fake up-moves: repairing more aggressively is conservative
GAIN_CAP = 0.40   # max credible one-day gain after repairs
MAX_REPAIRS = 20  # more than this and the series is untrustworthy -> drop


def clean_series(px: pd.Series) -> tuple[pd.Series, list]:
    """Repair one adjusted-close series. Returns (cleaned, log_entries)."""
    p = px.to_numpy(dtype="float64").copy()
    n = len(p)
    log = []
    # pass 1: spike-reversal bad prints (level reverts within 3 sessions);
    # clustered spikes need several sweeps
    for _ in range(3):
        repaired = False
        q = p[1:] / p[:-1]
        for t in np.where((q >= 2.0) | (q <= 0.5))[0] + 1:
            base = p[t - 1]
            rel0 = p[t] / base
            if not (rel0 >= 2.0 or rel0 <= 0.5):
                continue
            end = min(n, t + 4)
            rel = p[t:end] / base
            back = np.where((rel > 0.80) & (rel < 1.25))[0]
            if len(back) and back[0] > 0:
                p[t:t + back[0]] = base
                repaired = True
                log.append((px.index[t].date(), "spike_revert", round(float(rel0), 3)))
        if not repaired:
            break
    # pass 2: persistent cliffs at exact split fractions -> re-level history
    for _ in range(3):
        q = p[1:] / p[:-1]
        hit = None
        for t in np.where((q <= 0.55) | (q >= 1.8))[0] + 1:
            ratio = p[t] / p[t - 1]
            end = min(n, t + 6)
            persist = np.median(p[t:end]) / p[t - 1]
            if ratio <= 0.55 and persist < 0.70:
                f = SPLIT_FRACTIONS[np.argmin(np.abs(SPLIT_FRACTIONS - ratio))]
                if abs(ratio / f - 1.0) < DOWN_TOL:
                    hit = (t, ratio, "missed_split")
                    break
            elif ratio >= 1.8 and persist > 1.43:
                f = REV_FACTORS[np.argmin(np.abs(REV_FACTORS - ratio))]
                if abs(ratio / f - 1.0) < UP_TOL:
                    hit = (t, ratio, "missed_reverse_adj")
                    break
        if hit is None:
            break
        t, ratio, kind = hit
        p[:t] *= ratio          # re-level so the cliff day's return becomes 0
        log.append((px.index[t].date(), kind, round(float(ratio), 4)))
    # pass 3: cap residual one-day gains (never losses)
    q = p[1:] / p[:-1]
    for t in np.where(q > 1.0 + GAIN_CAP)[0] + 1:
        p[:t] *= (p[t] / p[t - 1]) / (1.0 + GAIN_CAP)
        log.append((px.index[t].date(), "gain_capped", round(float(q[t - 1] - 1), 3)))
    return pd.Series(p, index=px.index), log


def build():
    os.makedirs(PANEL_DIR, exist_ok=True)
    price_dir = os.path.join(DATA, "prices")
    files = sorted(f for f in os.listdir(price_dir) if f.endswith(".parquet"))
    print(f"consolidating {len(files)} symbols")
    acs, cls, vols = {}, {}, {}
    clean_log = []
    for i, fn in enumerate(files):
        sym = fn[:-8]
        try:
            df = pd.read_parquet(os.path.join(price_dir, fn),
                                 columns=["Close", "Adj Close", "Volume"])
        except Exception as e:
            print(f"  skip {sym}: {e}")
            continue
        if len(df) < 30:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        # drop rows with non-positive adjusted close (bad prints)
        df = df[(df["Adj Close"] > 0) & (df["Close"] > 0)]
        if len(df) < 30:
            continue
        cleaned, log = clean_series(df["Adj Close"])
        resid = cleaned.pct_change(fill_method=None)
        n_resid = int(((resid > 0.55) | (resid < -0.75)).sum())
        if len(log) > MAX_REPAIRS or n_resid > 3:
            clean_log.append((sym, "DROPPED", f"{len(log)} repairs/{n_resid} residual", ""))
            continue
        for dt, kind, val in log:
            clean_log.append((sym, dt, kind, val))
        acs[sym] = cleaned
        cls[sym] = df["Close"]
        vols[sym] = df["Volume"]
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(files)}")
    ac = pd.DataFrame(acs)
    c = pd.DataFrame(cls)
    v = pd.DataFrame(vols)
    # master calendar: days with enough active names (kills junk/holiday strays)
    active = ac.notna().sum(axis=1)
    cal = active[active >= MIN_ACTIVE].index
    ac, c, v = ac.loc[cal], c.loc[cal], v.loc[cal]
    ac.astype("float32").to_parquet(os.path.join(PANEL_DIR, "adj_close.parquet"))
    c.astype("float32").to_parquet(os.path.join(PANEL_DIR, "close.parquet"))
    v.astype("float32").to_parquet(os.path.join(PANEL_DIR, "volume.parquet"))
    lg = pd.DataFrame(clean_log, columns=["symbol", "date", "kind", "value"])
    lg.to_csv(os.path.join(PANEL_DIR, "cleaning_log.csv"), index=False)
    dropped = (lg["date"] == "DROPPED").sum()
    print(f"panels: {ac.shape[0]} days x {ac.shape[1]} symbols, "
          f"{cal[0].date()} -> {cal[-1].date()}")
    print(f"repairs: {len(lg)} entries "
          f"({(lg['kind']=='spike_revert').sum() if len(lg) else 0} spike, "
          f"{(lg['kind']=='missed_split').sum() if len(lg) else 0} split, "
          f"{(lg['kind']=='missed_reverse_adj').sum() if len(lg) else 0} rev, "
          f"{(lg['kind']=='gain_capped').sum() if len(lg) else 0} capped), "
          f"{dropped} symbols dropped")


def load():
    ac = pd.read_parquet(os.path.join(PANEL_DIR, "adj_close.parquet"))
    c = pd.read_parquet(os.path.join(PANEL_DIR, "close.parquet"))
    v = pd.read_parquet(os.path.join(PANEL_DIR, "volume.parquet"))
    idx = {}
    for name in ("SENSEX", "NIFTY50", "NIFTY500"):
        p = os.path.join(DATA, "indices", f"{name}.parquet")
        if os.path.exists(p):
            idx[name] = pd.read_parquet(p)
    return ac, c, v, idx


if __name__ == "__main__":
    build()
