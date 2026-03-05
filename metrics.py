"""
metrics.py
----------
Computes all performance metrics from a backtest result DataFrame.
All functions are pure — they accept a Series or DataFrame and return a value.
"""

import numpy as np
import pandas as pd


def cagr(equity: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan
    n_years = len(equity) / 252
    total_return = equity.iloc[-1] / equity.iloc[0]
    return total_return ** (1 / n_years) - 1


def sharpe_ratio(net_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe Ratio."""
    excess = net_returns.dropna() - risk_free_rate / 252
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(252)


def sortino_ratio(net_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sortino Ratio (penalises only downside deviation)."""
    excess = net_returns.dropna() - risk_free_rate / 252
    downside = excess[excess < 0].std()
    if downside == 0:
        return np.nan
    return (excess.mean() / downside) * np.sqrt(252)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative decimal."""
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min()


def calmar_ratio(equity: pd.Series) -> float:
    """CAGR / |Max Drawdown|."""
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return np.nan
    return cagr(equity) / mdd


def win_rate(trade_log: pd.DataFrame) -> float:
    """Fraction of trades that were profitable."""
    if trade_log.empty:
        return np.nan
    return (trade_log["Return_pct"] > 0).mean()


def avg_win_loss(trade_log: pd.DataFrame) -> dict:
    """Average win and average loss in percent."""
    wins   = trade_log[trade_log["Return_pct"] > 0]["Return_pct"]
    losses = trade_log[trade_log["Return_pct"] < 0]["Return_pct"]
    return {
        "avg_win":  round(wins.mean(), 2) if not wins.empty else 0.0,
        "avg_loss": round(losses.mean(), 2) if not losses.empty else 0.0,
    }


def turnover(df: pd.DataFrame) -> float:
    """
    Average daily portfolio turnover (fraction of portfolio traded per day).
    High turnover → high transaction costs.
    """
    if "Traded" not in df.columns:
        return np.nan
    return df["Traded"].mean()


def compute_all(df: pd.DataFrame, trade_log: pd.DataFrame | None = None) -> dict:
    """
    Convenience function: compute and return all metrics as a flat dict.

    Parameters
    ----------
    df         : backtest result DataFrame (must have Equity, Net_Return columns)
    trade_log  : optional trade log from build_trade_log()
    """
    metrics = {
        "CAGR":           round(cagr(df["Equity"]) * 100, 2),
        "Sharpe":         round(sharpe_ratio(df["Net_Return"]), 2),
        "Sortino":        round(sortino_ratio(df["Net_Return"]), 2),
        "Max Drawdown":   round(max_drawdown(df["Equity"]) * 100, 2),
        "Calmar":         round(calmar_ratio(df["Equity"]), 2),
        "Turnover (avg)": round(turnover(df) * 100, 2),
        "Total Return":   round((df["Equity"].iloc[-1] / df["Equity"].iloc[0] - 1) * 100, 2),
    }

    if trade_log is not None and not trade_log.empty:
        wl = avg_win_loss(trade_log)
        metrics.update({
            "Win Rate":  round(win_rate(trade_log) * 100, 2),
            "Avg Win":   wl["avg_win"],
            "Avg Loss":  wl["avg_loss"],
            "Num Trades": len(trade_log),
        })

    return metrics
