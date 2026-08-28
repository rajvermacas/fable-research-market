"""Backtest engine: regime-gated ensemble-momentum rotation on NSE equities.

Execution model (no look-ahead):
  - Every signal is computed on close of day T and executed at close of day T+1.
  - Long-only cash equities, no leverage. Un-invested capital earns the cash
    rate (liquid fund / T-bill proxy).

Strategy layers (selection):
  1. Point-in-time eligibility: trading now, >=1yr history, liquidity top-K by
     63d median rupee turnover, price floor, stock above its own 200d SMA,
     positive momentum score.
  2. Ensemble momentum score: mean of 3/6/12-month returns (each skipping the
     most recent month) divided by 6-month realized vol.
  3. Monthly rotation into top-N (equal weight) with a rank buffer
     (hold while still in the top buffer_mult*N).

Risk layers (exposure = trend gate x breadth scalar x vol scalar x eq brake):
  4. Index trend gate (binary, hysteresis band): Sensex above its 200d SMA,
     AND not more than idx_dd_exit below its idx_dd_window-day high
     (fast-crash circuit breaker, self-healing as the rolling high decays).
  5. Breadth scalar (continuous): share of the liquid pool above their own
     200d SMA, mapped linearly from breadth_lo -> 0 to breadth_hi -> 1.
  6. Volatility targeting: min(1, vol_target / EWMA vol of the invested
     book). Cuts size within days in vol explosions; no leverage ever.
  7. Equity-curve brake: while own equity sits below its own 200d (log) SMA,
     exposure is multiplied by eq_brake_scalar - catches factor winters that
     market-level gates cannot see (e.g. 2018-19).
  8. Per-stock trailing stop vs post-entry peak close; stale/delisting
     force-exit with haircut. Exposure trades only when the desired level
     drifts >= exp_band from actual (cost control).
  9. Transaction costs per side on all traded notional; idle cash earns the
     T-bill/liquid-fund proxy rate.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import panels

TRADING_DAYS = 252


@dataclass
class Params:
    # selection
    n_stocks: int = 15
    buffer_mult: float = 2.0
    mom_lookbacks: tuple = (63, 126, 252)
    mom_skip: int = 21
    sigma_window: int = 126
    sigma_floor: float = 0.10
    top_turnover: int = 300
    price_floor: float = 10.0
    stock_sma: int = 200
    weighting: str = "equal"          # "equal" | "invvol"
    weight_cap: float = 0.10
    min_history: int = 252
    min_obs_252: int = 200
    # market gate: exposure = sensex_trend(0/1) * breadth_scalar * vol_scalar
    regime_ma: int = 200              # Sensex SMA window (0 = leg off)
    regime_band: float = 0.01
    idx_dd_exit: float = 0.10         # fast-crash breaker: index this far below
    idx_dd_window: int = 63           # ...its rolling high -> gate 0 (0 = off)
    use_breadth: bool = True
    breadth_lo: float = 0.25          # breadth <= lo -> scalar 0
    breadth_hi: float = 0.55          # breadth >= hi -> scalar 1
    breadth_smooth: int = 5           # smooth breadth over N sessions
    # volatility targeting (crash brake: binds only in abnormal stress)
    vol_target: float = 0.25          # annualized; 0 = off
    vol_lambda: float = 0.94          # EWMA decay (RiskMetrics standard)
    exp_band: float = 0.15            # rebalance exposure when drift >= band
    # equity-curve brake: factor winters that market gates cannot see
    eq_brake_ma: int = 200            # SMA of own equity (0 = off)
    eq_brake_scalar: float = 0.5      # exposure multiplier while below
    # stock-level protection: trailing stop vs post-entry peak close
    crash_stop: float = 0.25          # 0 = off
    stale_limit: int = 10
    stale_haircut: float = 0.10
    # frictions and cash
    cost_per_side: float = 0.003
    cash_rate: float = 0.06
    # testing aids
    reb_offset: int = 0               # rebalance N sessions after month-end
    ew_universe: bool = False         # passive equal-weight of eligible pool


class Market:
    """Precomputed signal arrays shared across backtest runs."""

    def __init__(self, start_pad: str = "1996-01-01"):
        ac, c, v, idx = panels.load()
        ac = ac[ac.index >= start_pad]
        c = c.reindex(ac.index)
        v = v.reindex(ac.index)
        self.dates = ac.index
        self.syms = list(ac.columns)
        self.n_sym = len(self.syms)

        self.valid = ac.notna().to_numpy()
        acf = ac.ffill()
        cf = c.ffill()
        self.AC = acf.to_numpy(dtype="float64")
        self.C = cf.to_numpy(dtype="float64")

        r = acf.pct_change(fill_method=None)
        self.R = np.nan_to_num(r.to_numpy(dtype="float64"), nan=0.0)

        turn = (c * v)
        self.med_turn = turn.rolling(63, min_periods=40).median().to_numpy(dtype="float64")

        self.obs252 = (
            pd.DataFrame(self.valid, index=self.dates)
            .rolling(252, min_periods=1).sum().to_numpy(dtype="float64")
        )
        self.obs_total = np.cumsum(self.valid, axis=0).astype("float64")

        idx_arange = np.arange(len(self.dates))[:, None]
        last_real = np.where(self.valid, idx_arange, -1)
        last_real = np.maximum.accumulate(last_real, axis=0)
        self.stale = np.where(last_real >= 0, idx_arange - last_real, 10**6)

        self.sigma_cache = {}
        self.sma_cache = {}
        self.mom_cache = {}
        self._acf_df = acf

        per = self.dates.to_period("M")
        self.is_month_end = np.zeros(len(self.dates), dtype=bool)
        self.is_month_end[:-1] = per[:-1] != per[1:]
        self.is_month_end[-1] = True

        sx = idx["SENSEX"]["Close"]
        sx = sx[~sx.index.duplicated(keep="last")].sort_index()
        self.sensex = sx.reindex(self.dates).ffill()
        self.regime_ma_cache = {}
        self.pool_cache = {}
        self.indices = idx

    def sigma(self, window: int) -> np.ndarray:
        if window not in self.sigma_cache:
            r = pd.DataFrame(self.R, index=self.dates)
            s = r.rolling(window, min_periods=int(window * 0.6)).std() * np.sqrt(TRADING_DAYS)
            self.sigma_cache[window] = s.to_numpy(dtype="float64")
        return self.sigma_cache[window]

    def sma(self, window: int) -> np.ndarray:
        if window not in self.sma_cache:
            self.sma_cache[window] = (
                self._acf_df.rolling(window, min_periods=window).mean().to_numpy(dtype="float64")
            )
        return self.sma_cache[window]

    def mom(self, lookback: int, skip: int) -> np.ndarray:
        key = (lookback, skip)
        if key not in self.mom_cache:
            a = self._acf_df.shift(skip)
            b = self._acf_df.shift(skip + lookback)
            self.mom_cache[key] = (a / b - 1.0).to_numpy(dtype="float64")
        return self.mom_cache[key]

    def regime_sma(self, window: int) -> np.ndarray:
        if window not in self.regime_ma_cache:
            self.regime_ma_cache[window] = (
                self.sensex.rolling(window, min_periods=window).mean().to_numpy(dtype="float64")
            )
        return self.regime_ma_cache[window]

    def sensex_roll_max(self, window: int) -> np.ndarray:
        key = ("rmax", window)
        if key not in self.regime_ma_cache:
            self.regime_ma_cache[key] = (
                self.sensex.rolling(window, min_periods=1).max().to_numpy(dtype="float64")
            )
        return self.regime_ma_cache[key]

    def breadth_and_pool_ret(self, top_k: int, sma_window: int):
        """Daily (breadth, equal-weight return) of the liquid top-K pool.

        breadth = share of pool members above their own SMA; pool return is a
        vol proxy for periods when the book is empty. Both use same-day info
        only (they feed signals evaluated at that close).
        """
        key = (top_k, sma_window)
        if key not in self.pool_cache:
            n_days = len(self.dates)
            sma = self.sma(sma_window)
            breadth = np.full(n_days, np.nan)
            pool_ret = np.zeros(n_days)
            for d in range(n_days):
                turn = self.med_turn[d]
                ok = np.isfinite(turn) & (turn > 0) & (self.stale[d] < 5)
                idx_ok = np.where(ok)[0]
                if len(idx_ok) < 20:
                    continue
                if len(idx_ok) > top_k:
                    part = np.argpartition(turn[idx_ok], -top_k)[-top_k:]
                    idx_ok = idx_ok[part]
                above = self.AC[d][idx_ok] > sma[d][idx_ok]
                known = np.isfinite(sma[d][idx_ok])
                if known.sum() >= 20:
                    breadth[d] = above[known].mean()
                pool_ret[d] = np.nanmean(self.R[d][idx_ok])
            self.pool_cache[key] = (breadth, pool_ret)
        return self.pool_cache[key]


@dataclass
class Result:
    equity: pd.Series = None
    exposure: pd.Series = None
    n_held: pd.Series = None
    gate: pd.Series = None
    vol_scalar: pd.Series = None
    turnover_total: float = 0.0
    cost_total: float = 0.0
    n_rebalances: int = 0
    n_crash_stops: int = 0
    n_forced_exits: int = 0


def _cap_weights(raw: np.ndarray, cap: float) -> np.ndarray:
    w = raw / raw.sum()
    for _ in range(6):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        s = float(w[under].sum())
        if s <= 1e-12:
            break
        w[under] *= (s + excess) / s
    return np.minimum(w, cap)


def run(mkt: Market, p: Params, start: str, end: str) -> Result:
    dates = mkt.dates
    i0 = int(dates.searchsorted(pd.Timestamp(start)))
    i1 = int(dates.searchsorted(pd.Timestamp(end), side="right"))
    n_days = i1 - i0
    n_sym = mkt.n_sym

    sig = mkt.sigma(p.sigma_window)
    sma_stock = mkt.sma(p.stock_sma) if p.stock_sma else None
    moms = [mkt.mom(k, p.mom_skip) for k in p.mom_lookbacks]
    reg_ma = mkt.regime_sma(p.regime_ma) if p.regime_ma else None
    roll_max = mkt.sensex_roll_max(p.idx_dd_window) if p.idx_dd_exit else None
    sensex = mkt.sensex.to_numpy(dtype="float64")
    breadth, pool_ret = (None, None)
    if p.use_breadth:
        breadth, pool_ret = mkt.breadth_and_pool_ret(p.top_turnover,
                                                     p.stock_sma or 200)
        if p.breadth_smooth > 1:
            breadth = (
                pd.Series(breadth).rolling(p.breadth_smooth, min_periods=1)
                .mean().to_numpy()
            )
    cash_daily = (1.0 + p.cash_rate) ** (1.0 / TRADING_DAYS) - 1.0

    def select(d: int, held_idx: np.ndarray):
        """Top-N names and base weights (sum to 1) using close of day d."""
        if p.ew_universe:
            with np.errstate(invalid="ignore", divide="ignore"):
                elig = (
                    mkt.valid[d]
                    & (mkt.obs_total[d] >= p.min_history)
                    & (mkt.obs252[d] >= p.min_obs_252)
                    & (mkt.C[d] > p.price_floor)
                )
                turn = np.where(np.isfinite(mkt.med_turn[d]), mkt.med_turn[d], -1.0)
                elig &= turn > 0
                if elig.sum() > p.top_turnover:
                    thresh = np.partition(turn[elig], -p.top_turnover)[-p.top_turnover]
                    elig &= turn >= thresh
            idx_e = np.where(elig)[0]
            if len(idx_e) == 0:
                return np.array([], dtype=int), np.array([]), np.array([])
            base = np.full(len(idx_e), 1.0 / len(idx_e))
            return idx_e, base, mkt.AC[d][idx_e].copy()
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
            )
            if sma_stock is not None:
                elig &= mkt.AC[d] > sma_stock[d]
            turn = np.where(np.isfinite(mkt.med_turn[d]), mkt.med_turn[d], -1.0)
            elig &= turn > 0
            if elig.sum() > p.top_turnover:
                thresh = np.partition(turn[elig], -p.top_turnover)[-p.top_turnover]
                elig &= turn >= thresh

        idx_e = np.where(elig)[0]
        if len(idx_e) == 0:
            return np.array([], dtype=int), np.array([]), np.array([])
        order = idx_e[np.argsort(-score[idx_e])]
        rank = {s: r for r, s in enumerate(order)}
        buffer_n = int(p.buffer_mult * p.n_stocks)
        keep = [s for s in held_idx if s in rank and rank[s] < buffer_n]
        chosen = list(keep)
        for s in order:
            if len(chosen) >= p.n_stocks:
                break
            if s not in chosen:
                chosen.append(s)
        chosen = np.array(chosen[: p.n_stocks], dtype=int)
        if p.weighting == "equal":
            base = np.full(len(chosen), 1.0 / p.n_stocks)
        else:
            raw = 1.0 / np.maximum(sig[d][chosen], p.sigma_floor)
            base = _cap_weights(raw, p.weight_cap)
        return chosen, base, mkt.AC[d][chosen].copy()

    reb_flags = mkt.is_month_end
    if p.reb_offset:
        src = np.where(mkt.is_month_end)[0] + p.reb_offset
        reb_flags = np.zeros(len(dates), dtype=bool)
        reb_flags[src[src < len(dates)]] = True

    # ---- state ----
    w = np.zeros(n_sym)               # actual weights, sum = current exposure
    base_w = np.zeros(n_sym)          # selection weights, sum <= 1
    peak_px = np.zeros(n_sym)         # trailing high-water close per position
    equity = 1.0
    sensex_leg = True                 # resolved on first evaluated close
    ewma_var = (0.20 ** 2) / TRADING_DAYS
    eq_cumsum = np.zeros(n_days + 1)  # running sum of log-equity for own-SMA
    brake_on = False
    pending = None                    # (target_w, entry_updates, forced_w, is_selection)
    res = Result()
    eq_out = np.empty(n_days)
    exp_out = np.empty(n_days)
    nh_out = np.empty(n_days, dtype=int)
    g_out = np.empty(n_days)
    vs_out = np.empty(n_days)

    for j in range(n_days):
        d = i0 + j
        # 1) accrue returns close(d-1) -> close(d)
        exposure = float(w.sum())
        cash = 1.0 - exposure
        port_ret = float(w @ mkt.R[d]) + cash * cash_daily
        equity *= 1.0 + port_ret
        if 1.0 + port_ret > 1e-9:
            w = w * (1.0 + mkt.R[d]) / (1.0 + port_ret)

        # update realized-vol estimate of the invested book (or pool proxy)
        exposure = float(w.sum())
        if exposure > 0.05:
            r_book = float(w @ mkt.R[d]) / exposure
        elif pool_ret is not None:
            r_book = float(pool_ret[d])
        else:
            r_book = 0.0
        ewma_var = p.vol_lambda * ewma_var + (1 - p.vol_lambda) * r_book * r_book

        # 2) execute pending target at today's close
        if pending is not None:
            target, entry_updates, forced_w, new_base = pending
            traded = float(np.abs(target - w).sum())
            cost = p.cost_per_side * traded
            haircut = forced_w * p.stale_haircut
            equity *= (1.0 - cost) * (1.0 - haircut)
            res.turnover_total += traded
            res.cost_total += cost
            w = target.copy()
            for s, px in entry_updates:
                peak_px[s] = max(peak_px[s], px)
            base_w = new_base
            pending = None

        # 3) signals on close of day d -> execute close of day d+1
        # 3a. market gate: index trend (binary w/ hysteresis) x breadth scalar
        if reg_ma is not None and np.isfinite(reg_ma[d]):
            if sensex_leg and sensex[d] < (1.0 - p.regime_band) * reg_ma[d]:
                sensex_leg = False
            elif not sensex_leg and sensex[d] > (1.0 + p.regime_band) * reg_ma[d]:
                sensex_leg = True
        gate = 1.0 if sensex_leg else 0.0
        if roll_max is not None and sensex[d] < (1.0 - p.idx_dd_exit) * roll_max[d]:
            gate = 0.0
        if breadth is not None and np.isfinite(breadth[d]):
            b_scalar = (breadth[d] - p.breadth_lo) / max(p.breadth_hi - p.breadth_lo, 1e-9)
            gate *= min(1.0, max(0.0, b_scalar))

        # 3b. vol-target scalar
        if p.vol_target > 0:
            realized = np.sqrt(ewma_var * TRADING_DAYS)
            vol_scalar = min(1.0, p.vol_target / max(realized, 1e-6))
        else:
            vol_scalar = 1.0

        # 3c. equity-curve brake (log-SMA of own equity, with hysteresis)
        eq_cumsum[j + 1] = eq_cumsum[j] + np.log(max(equity, 1e-12))
        eq_scalar = 1.0
        if p.eq_brake_ma and j + 1 >= p.eq_brake_ma:
            sma_log = (eq_cumsum[j + 1] - eq_cumsum[j + 1 - p.eq_brake_ma]) / p.eq_brake_ma
            log_eq = np.log(max(equity, 1e-12))
            if not brake_on and log_eq < sma_log - np.log(1.0 + p.regime_band):
                brake_on = True
            elif brake_on and log_eq > sma_log + np.log(1.0 + p.regime_band):
                brake_on = False
            if brake_on:
                eq_scalar = p.eq_brake_scalar

        desired_e = gate * vol_scalar * eq_scalar
        if desired_e < 0.05:          # not worth holding a sliver of a book
            desired_e = 0.0

        exposure = float(w.sum())
        held_idx = np.where(w > 1e-8)[0]
        base_sum = float(base_w.sum())

        if desired_e <= 1e-9 and exposure > 1e-9:
            pending = (np.zeros(n_sym), [], 0.0, np.zeros(n_sym))
        elif desired_e > 1e-9 and (reb_flags[d] or (exposure <= 1e-9)):
            # monthly rotation, or (re-)entry from cash -> fresh selection
            chosen, cw, px = select(d, held_idx)
            new_base = np.zeros(n_sym)
            if len(chosen):
                new_base[chosen] = cw
            target = new_base * desired_e
            fresh = [(s_, px_) for s_, px_ in zip(chosen, px) if w[s_] <= 1e-8]
            for s_, _ in fresh:
                peak_px[s_] = 0.0     # reset stale high-water marks on entry
            pending = (target, fresh, 0.0, new_base)
            res.n_rebalances += 1
        else:
            # stock-level exits first
            drop = np.zeros(n_sym, dtype=bool)
            forced_w = 0.0
            if len(held_idx):
                if p.crash_stop:
                    np.maximum.at(peak_px, held_idx, mkt.AC[d][held_idx])
                    stopped = held_idx[
                        mkt.AC[d][held_idx] <= (1.0 - p.crash_stop) * peak_px[held_idx]
                    ]
                    if len(stopped):
                        drop[stopped] = True
                        res.n_crash_stops += len(stopped)
                stale_hit = held_idx[mkt.stale[d][held_idx] > p.stale_limit]
                if len(stale_hit):
                    new_stale = stale_hit[~drop[stale_hit]]
                    forced_w = float(w[new_stale].sum())
                    drop[stale_hit] = True
                    res.n_forced_exits += len(new_stale)
            new_base = base_w.copy()
            new_base[drop] = 0.0
            nb_sum = float(new_base.sum())
            # a dropped stock's slot stays in cash until the next rotation:
            # target total = desired exposure scaled by surviving slot share
            alive_frac = nb_sum / base_sum if base_sum > 1e-9 else 0.0
            tgt_total = desired_e * alive_frac
            cur_alive = w.copy()
            cur_alive[drop] = 0.0
            cur_total = float(cur_alive.sum())
            need_exposure_trade = abs(tgt_total - cur_total) >= p.exp_band
            if drop.any() or need_exposure_trade:
                if need_exposure_trade and cur_total > 1e-9:
                    target = cur_alive * (tgt_total / cur_total)
                elif need_exposure_trade and nb_sum > 1e-9:
                    target = new_base * (tgt_total / nb_sum)
                else:
                    target = cur_alive
                pending = (target, [], forced_w, new_base)

        eq_out[j] = equity
        exp_out[j] = float(w.sum())
        nh_out[j] = len(held_idx)
        g_out[j] = gate
        vs_out[j] = vol_scalar

    didx = dates[i0:i1]
    res.equity = pd.Series(eq_out, index=didx, name="equity")
    res.exposure = pd.Series(exp_out, index=didx, name="exposure")
    res.n_held = pd.Series(nh_out, index=didx, name="n_held")
    res.gate = pd.Series(g_out, index=didx, name="gate")
    res.vol_scalar = pd.Series(vs_out, index=didx, name="vol_scalar")
    return res
