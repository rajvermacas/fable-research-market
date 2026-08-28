"""Download the full NSE equity universe daily history from Yahoo Finance.

Universe: every currently listed NSE main-board equity (SERIES == EQ) from the
official NSE equity master (EQUITY_L.csv), plus benchmark indices.

Output layout (all under data/):
    universe/equity_list.csv      raw NSE equity master
    universe/symbols.csv          parsed symbols with NSE listing date
    prices/<SYMBOL>.parquet       per-symbol daily OHLCV + Adj Close
    indices/<NAME>.parquet        benchmark index daily history
    manifest.csv                  download status per symbol

Run:  python3 -m src.download_data
"""
import io
import os
import random
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests as pyrequests

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import yf_compat  # noqa: F401,E402  (patches yfinance transport)
import yfinance as yf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
START = "1995-01-01"

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}

INDICES = {
    "SENSEX": "^BSESN",       # from 1997 — regime + long benchmark
    "NIFTY50": "^NSEI",       # from 2007
    "NIFTY500": "^CRSLDX",    # from 2005
}


def fetch_universe() -> pd.DataFrame:
    os.makedirs(os.path.join(DATA, "universe"), exist_ok=True)
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    r = pyrequests.get(url, headers=NSE_HEADERS, timeout=30)
    r.raise_for_status()
    raw_path = os.path.join(DATA, "universe", "equity_list.csv")
    with open(raw_path, "wb") as f:
        f.write(r.content)
    df = pd.read_csv(io.BytesIO(r.content))
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    df = df[df["SERIES"] == "EQ"].copy()
    df["LISTING_DATE"] = pd.to_datetime(df["DATE OF LISTING"], format="%d-%b-%Y", errors="coerce")
    out = df[["SYMBOL", "NAME OF COMPANY", "LISTING_DATE", "ISIN NUMBER"]].rename(
        columns={"NAME OF COMPANY": "NAME", "ISIN NUMBER": "ISIN"}
    )
    out.to_csv(os.path.join(DATA, "universe", "symbols.csv"), index=False)
    # index membership lists, for reference only
    for name, fn in [
        ("nifty500", "ind_nifty500list.csv"),
        ("niftytotalmarket", "ind_niftytotalmarket_list.csv"),
    ]:
        try:
            r2 = pyrequests.get(
                f"https://nsearchives.nseindia.com/content/indices/{fn}",
                headers=NSE_HEADERS, timeout=30,
            )
            if r2.ok:
                with open(os.path.join(DATA, "universe", f"{name}.csv"), "wb") as f:
                    f.write(r2.content)
        except Exception:
            pass
    return out


def fetch_one(symbol: str, dest: str) -> str:
    """Download one ticker's history. Returns status string."""
    yahoo_sym = f"{symbol}.NS"
    for attempt in range(4):
        try:
            t = yf.Ticker(yahoo_sym)
            df = t.history(start=START, auto_adjust=False, actions=False, timeout=30)
            if df is None or len(df) == 0:
                if attempt == 0:
                    time.sleep(2 + random.random() * 2)
                    continue
                return "empty"
            df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[df.index >= START]
            df = df[df["Close"] > 0]
            if len(df) == 0:
                return "empty"
            df.astype("float64").to_parquet(dest)
            return f"ok:{len(df)}:{df.index[0].date()}:{df.index[-1].date()}"
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Too Many" in msg:
                time.sleep(20 + 20 * attempt + random.random() * 10)
            else:
                time.sleep(2 * (attempt + 1) + random.random() * 2)
            last_err = msg[:120]
    return f"error:{last_err}"


def main():
    os.makedirs(os.path.join(DATA, "prices"), exist_ok=True)
    os.makedirs(os.path.join(DATA, "indices"), exist_ok=True)

    uni = fetch_universe()
    symbols = sorted(uni["SYMBOL"].unique())
    print(f"universe: {len(symbols)} EQ-series symbols", flush=True)

    for name, ysym in INDICES.items():
        dest = os.path.join(DATA, "indices", f"{name}.parquet")
        if os.path.exists(dest):
            continue
        df = yf.download(ysym, start=START, auto_adjust=False, progress=False)
        if df is not None and len(df):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.astype("float64").to_parquet(dest)
            print(f"index {name}: {len(df)} rows {df.index[0].date()} -> {df.index[-1].date()}", flush=True)
        else:
            print(f"index {name}: FAILED", flush=True)

    manifest_path = os.path.join(DATA, "manifest.csv")
    done = {}
    if os.path.exists(manifest_path):
        m = pd.read_csv(manifest_path)
        done = dict(zip(m["symbol"], m["status"]))

    todo = [s for s in symbols if not (
        s in done and (str(done[s]).startswith("ok") or done[s] == "empty")
    )]
    print(f"to download: {len(todo)} (already done: {len(symbols) - len(todo)})", flush=True)

    lock = threading.Lock()
    counter = {"n": 0}

    def work(sym):
        dest = os.path.join(DATA, "prices", f"{sym}.parquet")
        st = fetch_one(sym, dest)
        with lock:
            done[sym] = st
            counter["n"] += 1
            n = counter["n"]
            if n % 25 == 0 or n == len(todo):
                pd.DataFrame(
                    {"symbol": list(done.keys()), "status": list(done.values())}
                ).to_csv(manifest_path, index=False)
                ok = sum(1 for v in done.values() if str(v).startswith("ok"))
                print(f"progress {n}/{len(todo)}  ok={ok}  "
                      f"empty={sum(1 for v in done.values() if v == 'empty')}  "
                      f"err={sum(1 for v in done.values() if str(v).startswith('error'))}",
                      flush=True)
        time.sleep(0.15 + random.random() * 0.2)

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(work, s) for s in todo]
        for f in as_completed(futs):
            f.result()

    pd.DataFrame({"symbol": list(done.keys()), "status": list(done.values())}).to_csv(
        manifest_path, index=False
    )
    ok = sum(1 for v in done.values() if str(v).startswith("ok"))
    print(f"DONE. ok={ok} empty={sum(1 for v in done.values() if v=='empty')} "
          f"err={sum(1 for v in done.values() if str(v).startswith('error'))}", flush=True)


if __name__ == "__main__":
    main()
