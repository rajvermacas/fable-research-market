"""Performance metrics on a daily equity curve."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return np.nan
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0

def annual_vol(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    return float(r.std() * np.sqrt(TRADING_DAYS))


def sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    r = equity.pct_change().dropna()
    if r.std() == 0:
        return np.nan
    excess = r - rf / TRADING_DAYS
    return float(excess.mean() / r.std() * np.sqrt(TRADING_DAYS))


def yearly_returns(equity: pd.Series) -> pd.Series:
    y = equity.resample("YE").last()
    first = equity.iloc[0]
    y_prev = y.shift(1)
    y_prev.iloc[0] = first
    out = y / y_prev - 1
    out.index = out.index.year
    return out


def summary(equity: pd.Series, label: str = "") -> dict:
    return {
        "label": label,
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "CAGR": cagr(equity),
        "MaxDD": max_drawdown(equity),
        "Vol": annual_vol(equity),
        "Sharpe": sharpe(equity),
        "Calmar": cagr(equity) / abs(max_drawdown(equity)) if max_drawdown(equity) != 0 else np.nan,
    }
