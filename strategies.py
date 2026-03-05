"""
strategies.py
-------------
Each strategy function accepts a prepared DataFrame (with indicators already
added) and returns it with a 'Signal' column: 1 = long, 0 = flat, -1 = short.

CONTRACT
--------
  • Signal is the DESIRED position on day T, derived from day-T data.
  • backtest.py shifts Signal by 1 to get Position (executed on T+1).
  • Do NOT shift inside this file.
  • Values must be in {0, 1} for long-only, or {-1, 0, 1} if shorts allowed.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Momentum Strategy
# ---------------------------------------------------------------------------

def momentum_signal(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """
    Long (1) when fast SMA > slow SMA, flat (0) otherwise.
    Requires SMA_{fast} and SMA_{slow} columns (add via indicators.add_sma).
    """
    fast_col = f"SMA_{fast}"
    slow_col = f"SMA_{slow}"

    if fast_col not in df.columns or slow_col not in df.columns:
        raise ValueError(
            f"Missing columns: {fast_col} or {slow_col}. "
            "Run indicators.add_sma() before calling this strategy."
        )

    df["Signal"] = np.where(df[fast_col] > df[slow_col], 1, 0)
    # NaNs in SMA columns → signal should be 0 (no position), not NaN
    df["Signal"] = df["Signal"].where(df[fast_col].notna() & df[slow_col].notna(), 0)
    return df


# ---------------------------------------------------------------------------
# Mean Reversion Strategy — State Machine
# ---------------------------------------------------------------------------

def mean_reversion_signal(
    df: pd.DataFrame,
    window: int = 20,
    k: float = 2.0,
    allow_short: bool = False,
) -> pd.DataFrame:
    """
    State-machine mean reversion using Bollinger Bands.

    Rules (long-only, allow_short=False):
      • Enter LONG  when Close crosses BELOW the lower band  (oversold)
      • Exit  LONG  when Close crosses ABOVE the middle band (mean reversion complete)
      • Otherwise   hold current state

    Rules (with allow_short=True, adds short leg):
      • Enter SHORT when Close crosses ABOVE the upper band  (overbought)
      • Exit  SHORT when Close crosses BELOW the middle band

    This produces a persistent POSITION series (1 / 0 / -1), not impulse spikes.
    Requires BB_UPPER_{window}, BB_MID_{window}, BB_LOWER_{window} columns.
    """
    upper  = f"BB_UPPER_{window}"
    mid    = f"BB_MID_{window}"
    lower  = f"BB_LOWER_{window}"

    for col in [upper, mid, lower]:
        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}'. "
                "Run indicators.add_bollinger_bands() before calling this strategy."
            )

    close = df["Close"].values
    up    = df[upper].values
    mid_  = df[mid].values
    lo    = df[lower].values
    n     = len(df)

    signal = np.zeros(n, dtype=int)
    state  = 0  # current desired position: 0=flat, 1=long, -1=short

    for i in range(n):
        # Skip rows where indicator is NaN (warm-up period)
        if np.isnan(lo[i]) or np.isnan(up[i]) or np.isnan(mid_[i]):
            signal[i] = 0
            state = 0
            continue

        if state == 0:
            if close[i] < lo[i]:
                state = 1                              # oversold → go long
            elif allow_short and close[i] > up[i]:
                state = -1                             # overbought → go short

        elif state == 1:
            if close[i] > mid_[i]:                    # reverted → exit long
                state = 0

        elif state == -1:
            if close[i] < mid_[i]:                    # reverted → exit short
                state = 0

        signal[i] = state

    df["Signal"] = signal
    return df


# ---------------------------------------------------------------------------
# Regime-Aware Strategy Switcher
# ---------------------------------------------------------------------------

def regime_aware_signal(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    bb_window: int = 20,
    bb_k: float = 2.0,
    allow_short: bool = False,
) -> pd.DataFrame:
    """
    Selects strategy by regime:
      Trending  → momentum_signal
      Choppy    → mean_reversion_signal
      HighVol   → flat (0)  — capital preservation

    Requires 'Regime' column from regime.classify_regime().
    """
    if "Regime" not in df.columns:
        raise ValueError("Missing 'Regime' column. Run regime.classify_regime() first.")

    df = df.copy()
    df_mom = momentum_signal(df.copy(), fast, slow)
    df_rev = mean_reversion_signal(df.copy(), bb_window, bb_k, allow_short=allow_short)

    df["Signal"] = 0
    trending_mask = df["Regime"] == "Trending"
    choppy_mask   = df["Regime"] == "Choppy"

    df.loc[trending_mask, "Signal"] = df_mom.loc[trending_mask, "Signal"]
    df.loc[choppy_mask,   "Signal"] = df_rev.loc[choppy_mask,   "Signal"]

    return df


# ---------------------------------------------------------------------------
# Rotation Strategy  (used by Portfolio tab)
# ---------------------------------------------------------------------------

def rotation_signal(
    prices: pd.DataFrame,
    lookback: int = 20,
    vol_filter: bool = True,
    safe_haven: str = "TLT",
) -> pd.Series:
    """
    Given a wide DataFrame of prices (columns = tickers), return a Series
    mapping date → selected ticker (highest momentum score).
    Momentum = % return over lookback days.
    Falls back to safe_haven when best momentum is negative and vol_filter=True.
    """
    momentum = prices.pct_change(lookback)
    selected = pd.Series(index=prices.index, dtype=object)

    for date in prices.index:
        row = momentum.loc[date].dropna()
        if row.empty:
            selected[date] = None
            continue
        best = row.idxmax()
        if vol_filter and row[best] < 0 and safe_haven in prices.columns:
            selected[date] = safe_haven
        else:
            selected[date] = best

    return selected
