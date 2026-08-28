"""Named robustness variants beyond the parameter sweep.

  raw_momentum     same selection, ALL risk layers off (what risk mgmt adds)
  no_riskfree      cash earns 0% instead of T-bill proxy
  largecap_only    liquidity pool = top 100 (survivorship-bias stress test:
                   missing delisted names are concentrated in small caps)
  smaller_pool     pool = top 200
  ew_universe      passive equal-weight of the whole eligible pool, monthly,
                   no momentum, no risk layers (internal benchmark carrying
                   the SAME survivorship bias as the strategy universe)
  reb_offset_10    rebalance 10 sessions after month-end (date-luck test)

Usage: python3 -m src.variants
"""
import json
import os

import pandas as pd

from . import metrics
from .engine import Market, Params, run
from .sweep import evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

VARIANTS = {
    "base": {},
    "raw_momentum": {"regime_ma": 0, "idx_dd_exit": 0.0, "use_breadth": False,
                     "vol_target": 0.0, "eq_brake_ma": 0, "crash_stop": 0.0,
                     "cash_rate": 0.0},
    "no_riskfree": {"cash_rate": 0.0},
    "largecap_only": {"top_turnover": 100, "n_stocks": 10},
    "smaller_pool": {"top_turnover": 200},
    "ew_universe": {"ew_universe": True, "regime_ma": 0, "idx_dd_exit": 0.0,
                    "use_breadth": False, "vol_target": 0.0, "eq_brake_ma": 0,
                    "crash_stop": 0.0, "cash_rate": 0.0},
    "reb_offset_10": {"reb_offset": 10},
}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    mkt = Market()
    rows = []
    for label, ov in VARIANTS.items():
        p = Params(**ov)
        m = evaluate(mkt, p)
        m["label"] = label
        rows.append(m)
        print(label, {k: round(v, 3) for k, v in m.items()
                      if isinstance(v, float)}, flush=True)
    df = pd.DataFrame(rows).set_index("label")
    df.to_csv(os.path.join(RESULTS, "variants.csv"))
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
