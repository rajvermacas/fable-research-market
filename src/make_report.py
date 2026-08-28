"""Generate results/report.html — the full research report with embedded charts.

Every number shown is computed from the results CSVs at build time.
"""
import base64
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def img64(name):
    with open(os.path.join(RESULTS, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def pc(x, dp=1, sign=False):
    s = f"{x*100:+.{dp}f}%" if sign else f"{x*100:.{dp}f}%"
    return s


def load_numbers():
    with open(os.path.join(RESULTS, "base_report.json")) as f:
        rep = json.load(f)
    yearly = pd.read_csv(os.path.join(RESULTS, "base_yearly_returns.csv"), index_col=0)
    var = pd.read_csv(os.path.join(RESULTS, "variants.csv"), index_col=0)
    oat = pd.read_csv(os.path.join(RESULTS, "sweep_oat.csv"))
    rnd = pd.read_csv(os.path.join(RESULTS, "sweep_random.csv")).dropna(subset=["is_cagr"])
    return rep, yearly, var, oat, rnd


def yearly_rows(yearly):
    rows = []
    for y, r in yearly.iterrows():
        cls_s = "pos" if r["strategy"] >= 0 else "neg"
        cls_b = "pos" if r["sensex"] >= 0 else "neg"
        fwd = ' class="fwdrow"' if int(y) >= 2022 else ""
        rows.append(
            f'<tr{fwd}><td>{y}</td>'
            f'<td class="num {cls_s}">{pc(r["strategy"], 1, True)}</td>'
            f'<td class="num {cls_b}">{pc(r["sensex"], 1, True)}</td></tr>'
        )
    return "\n".join(rows)


def main():
    rep, yearly, var, oat, rnd = load_numbers()
    is_ = rep["in_sample_2000_2021"]
    fw = rep["forward_2022_today"]
    full = rep["full_2000_today"]
    dg = rep["diagnostics"]
    bench_is = rep["benchmarks"]["SENSEX:in_sample_2000_2021"]
    bench_fw = rep["benchmarks"]["SENSEX:forward_2022_today"]

    oat_near = ((oat["is_cagr"] > 0.27) & (oat["is_maxdd"] > -0.28)).sum()
    oat_pass = ((oat["is_cagr"] > 0.30) & (oat["is_maxdd"] > -0.25)).sum()
    oat_sharpe_min = oat["is_sharpe"].min()
    rnd_sharpe_min = rnd["is_sharpe"].min()
    rnd_fw_dd_ok = (rnd["fw_maxdd"] > -0.25).mean()
    rnd_fw_cagr_med = rnd["fw_cagr"].median()
    n_rnd = len(rnd)
    # configs that beat 30% forward all fail in-sample DD
    big_fw = rnd[rnd["fw_cagr"] > 0.30]
    big_fw_worst_isdd = big_fw["is_maxdd"].max() if len(big_fw) else float("nan")

    v = var
    T = open(os.path.join(ROOT, "src", "report_template.html")).read()
    subs = {
        "%%AS_OF%%": full["end"],
        "%%IS_CAGR%%": pc(is_["CAGR"]), "%%IS_DD%%": pc(is_["MaxDD"]),
        "%%IS_SHARPE%%": f"{is_['Sharpe']:.2f}", "%%IS_CALMAR%%": f"{is_['Calmar']:.2f}",
        "%%IS_VOL%%": pc(is_["Vol"]),
        "%%FW_CAGR%%": pc(fw["CAGR"]), "%%FW_DD%%": pc(fw["MaxDD"]),
        "%%FW_SHARPE%%": f"{fw['Sharpe']:.2f}",
        "%%FULL_CAGR%%": pc(full["CAGR"]), "%%FULL_DD%%": pc(full["MaxDD"]),
        "%%FULL_MULT%%": f"{1+full['total_return']:.0f}",
        "%%BIS_CAGR%%": pc(bench_is["CAGR"]), "%%BIS_DD%%": pc(bench_is["MaxDD"]),
        "%%BFW_CAGR%%": pc(bench_fw["CAGR"]), "%%BFW_DD%%": pc(bench_fw["MaxDD"]),
        "%%EXPOSURE%%": pc(dg["avg_exposure"], 0),
        "%%TURNOVER%%": f"{dg['one_way_turnover_per_year']:.1f}",
        "%%N_STOPS%%": str(dg["n_crash_stops"]),
        "%%RAW_CAGR%%": pc(v.loc["raw_momentum", "is_cagr"]),
        "%%RAW_DD%%": pc(v.loc["raw_momentum", "is_maxdd"]),
        "%%EW_CAGR%%": pc(v.loc["ew_universe", "is_cagr"]),
        "%%EW_DD%%": pc(v.loc["ew_universe", "is_maxdd"]),
        "%%LC_CAGR%%": pc(v.loc["largecap_only", "is_cagr"]),
        "%%LC_DD%%": pc(v.loc["largecap_only", "is_maxdd"]),
        "%%NORF_CAGR%%": pc(v.loc["no_riskfree", "is_cagr"]),
        "%%OFFSET_CAGR%%": pc(v.loc["reb_offset_10", "is_cagr"]),
        "%%OFFSET_DD%%": pc(v.loc["reb_offset_10", "is_maxdd"]),
        "%%OAT_N%%": str(len(oat)), "%%OAT_NEAR%%": str(oat_near),
        "%%OAT_PASS%%": str(oat_pass),
        "%%OAT_SHARPE_MIN%%": f"{oat_sharpe_min:.2f}",
        "%%RND_N%%": str(n_rnd),
        "%%RND_SHARPE_MIN%%": f"{rnd_sharpe_min:.2f}",
        "%%RND_FWDD_OK%%": pc(rnd_fw_dd_ok, 0),
        "%%RND_FW_MED%%": pc(rnd_fw_cagr_med),
        "%%BIGFW_WORST_ISDD%%": pc(big_fw_worst_isdd),
        "%%BIGFW_N%%": str(len(big_fw)),
        "%%YEARLY_ROWS%%": yearly_rows(yearly),
        "%%IMG_EQUITY%%": img64("equity_curve.png"),
        "%%IMG_EXPOSURE%%": img64("exposure.png"),
        "%%IMG_YEARLY%%": img64("yearly_returns.png"),
        "%%IMG_ROLLING%%": img64("rolling_cagr.png"),
    }
    for k, val in subs.items():
        T = T.replace(k, val)
    out = os.path.join(RESULTS, "report.html")
    with open(out, "w") as f:
        f.write(T)
    print(f"wrote {out} ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
