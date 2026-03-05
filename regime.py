"""
regime.py
---------
Classifies the current market regime into one of three states:
  - "Trending"  : strong directional trend, momentum strategies preferred
  - "Choppy"    : low trend, mean reversion preferred
  - "HighVol"   : elevated volatility, reduce exposure or use safe-haven fallback

Classification is based on:
  1. SMA_200 slope (trend direction + strength)
  2. Volatility percentile (annualised vol vs its own rolling history)
"""

import numpy as np
import pandas as pd


def _sma_slope(df: pd.DataFrame, window: int = 200, slope_window: int = 20) -> pd.Series:
    """
    Normalised slope of the SMA_200 over the last slope_window bars.
    Positive = uptrend, Negative = downtrend.
    """
    sma_col = f"SMA_{window}"
    if sma_col not in df.columns:
        raise ValueError(f"Missing {sma_col}. Run add_sma({window}) first.")

    slope = df[sma_col].diff(slope_window) / df[sma_col].shift(slope_window)
    return slope


def _vol_percentile(df: pd.DataFrame, vol_col: str = "VOL_20", lookback: int = 252) -> pd.Series:
    """
    Rolling percentile rank of current volatility vs the past `lookback` days.
    Returns a 0–100 value. ≥ 75 is considered "high vol".
    """
    if vol_col not in df.columns:
        raise ValueError(f"Missing {vol_col}. Run add_volatility() first.")

    pct = df[vol_col].rolling(lookback).rank(pct=True) * 100
    return pct


def classify_regime(
    df: pd.DataFrame,
    trend_threshold: float = 0.001,
    high_vol_pct: float = 75.0,
    vol_lookback: int = 252,
) -> pd.DataFrame:
    """
    Add a 'Regime' column to df with values: "Trending", "Choppy", "HighVol".

    Logic (evaluated in priority order):
      1. If vol percentile >= high_vol_pct → "HighVol"
      2. If |SMA slope| >= trend_threshold  → "Trending"
      3. Otherwise                          → "Choppy"

    Parameters
    ----------
    trend_threshold : minimum absolute SMA slope to call a trend
    high_vol_pct    : vol percentile cutoff for "HighVol" regime
    vol_lookback    : rolling window for vol percentile calculation
    """
    slope   = _sma_slope(df)
    vol_pct = _vol_percentile(df, lookback=vol_lookback)

    conditions = [
        vol_pct >= high_vol_pct,          # HighVol takes priority
        slope.abs() >= trend_threshold,   # Then Trending
    ]
    choices = ["HighVol", "Trending"]

    df["Regime"]     = np.select(conditions, choices, default="Choppy")
    df["SMA_Slope"]  = slope
    df["Vol_Pct"]    = vol_pct

    return df


def regime_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary dict of regime distribution for display in the UI.
    """
    if "Regime" not in df.columns:
        raise ValueError("Run classify_regime() first.")

    counts = df["Regime"].value_counts(normalize=True).mul(100).round(1)
    current = df["Regime"].iloc[-1] if not df.empty else "Unknown"

    return {
        "current_regime": current,
        "distribution": counts.to_dict(),
        "current_vol_pct": round(df["Vol_Pct"].iloc[-1], 1) if "Vol_Pct" in df.columns else None,
        "current_slope": round(df["SMA_Slope"].iloc[-1] * 100, 3) if "SMA_Slope" in df.columns else None,
    }
