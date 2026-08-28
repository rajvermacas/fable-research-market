"""Charts for the report. Writes PNGs to results/."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

C_STRAT = "#1f6feb"
C_BENCH = "#8b949e"
C_BAD = "#d1242f"
C_OK = "#1a7f37"


def main():
    df = pd.read_csv(os.path.join(RESULTS, "base_equity_daily.csv"),
                     index_col=0, parse_dates=True)
    eq = df["equity"]
    sx = pd.read_parquet(os.path.join(ROOT, "data", "indices", "SENSEX.parquet"))["Close"]
    sx = sx[~sx.index.duplicated(keep="last")].sort_index()
    sx = sx.reindex(eq.index).ffill()
    sx = sx / sx.iloc[0]

    # 1. equity curve (log) with forward-test shading
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(eq.index, eq, color=C_STRAT, lw=1.4, label="Strategy")
    ax.plot(sx.index, sx, color=C_BENCH, lw=1.1, label="Sensex")
    ax.set_yscale("log")
    ax.axvline(pd.Timestamp("2022-01-01"), color=C_BAD, ls="--", lw=1)
    ax.text(pd.Timestamp("2022-03-01"), eq.min() * 1.5, "forward test",
            color=C_BAD, fontsize=9)
    ax.set_ylabel("growth of 1 (log)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_title("Regime-gated ensemble momentum vs Sensex, 2000-2026")

    dd = eq / eq.cummax() - 1
    ddb = sx / sx.cummax() - 1
    ax = axes[1]
    ax.fill_between(dd.index, dd * 100, 0, color=C_STRAT, alpha=0.55, lw=0)
    ax.plot(ddb.index, ddb * 100, color=C_BENCH, lw=0.9, label="Sensex DD")
    ax.axhline(-25, color=C_BAD, ls=":", lw=1)
    ax.text(eq.index[100], -27.5, "-25% mandate", color=C_BAD, fontsize=8)
    ax.set_ylabel("drawdown %")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "equity_curve.png"), dpi=150)
    plt.close(fig)

    # 2. yearly returns bar chart
    yr = pd.read_csv(os.path.join(RESULTS, "base_yearly_returns.csv"), index_col=0)
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(yr))
    ax.bar(x - 0.2, yr["strategy"] * 100, width=0.4, color=C_STRAT, label="Strategy")
    ax.bar(x + 0.2, yr["sensex"] * 100, width=0.4, color=C_BENCH, label="Sensex")
    ax.set_xticks(x)
    ax.set_xticklabels(yr.index, rotation=60, fontsize=8)
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(x[yr.index.get_loc(2022)] - 0.5, color=C_BAD, ls="--", lw=1)
    ax.set_ylabel("calendar-year return %")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "yearly_returns.png"), dpi=150)
    plt.close(fig)

    # 3. exposure and gates
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.fill_between(df.index, df["exposure"], 0, color=C_STRAT, alpha=0.45,
                    lw=0, label="actual exposure")
    ax.plot(df.index, df["gate"].rolling(10).mean(), color=C_BAD, lw=0.7,
            alpha=0.8, label="market gate (10d avg)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction invested")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exposure.png"), dpi=150)
    plt.close(fig)

    # 4. rolling 3-year CAGR
    r3 = (eq / eq.shift(756)) ** (1 / 3) - 1
    b3 = (sx / sx.shift(756)) ** (1 / 3) - 1
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(r3.index, r3 * 100, color=C_STRAT, lw=1.2, label="Strategy")
    ax.plot(b3.index, b3 * 100, color=C_BENCH, lw=1.0, label="Sensex")
    ax.axhline(30, color=C_OK, ls=":", lw=1)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("rolling 3y CAGR %")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "rolling_cagr.png"), dpi=150)
    plt.close(fig)
    print("plots written")


if __name__ == "__main__":
    main()
