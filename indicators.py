"""
indicators.py
-------------
Pure functions that take a DataFrame and return it with new indicator columns.
No side effects — all functions return a copy or mutate in-place (consistent).
"""

import numpy as np
import pandas as pd


def add_sma(df: pd.DataFrame, window: int, col: str = "Close") -> pd.DataFrame:
    df[f"SMA_{window}"] = df[col].rolling(window).mean()
    return df


def add_ema(df: pd.DataFrame, window: int, col: str = "Close") -> pd.DataFrame:
    df[f"EMA_{window}"] = df[col].ewm(span=window, adjust=False).mean()
    return df


def add_volatility(df: pd.DataFrame, window: int = 20, col: str = "Close") -> pd.DataFrame:
    """Rolling standard deviation of daily log returns."""
    log_ret = np.log(df[col] / df[col].shift(1))
    df[f"VOL_{window}"] = log_ret.rolling(window).std() * np.sqrt(252)  # annualised
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average True Range — useful for position sizing."""
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"ATR_{window}"] = tr.rolling(window).mean()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14, col: str = "Close") -> pd.DataFrame:
    """Relative Strength Index."""
    delta = df[col].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, k: float = 2.0, col: str = "Close") -> pd.DataFrame:
    """Upper and lower Bollinger Bands."""
    sma = df[col].rolling(window).mean()
    std = df[col].rolling(window).std()
    df[f"BB_MID_{window}"]   = sma
    df[f"BB_UPPER_{window}"] = sma + k * std
    df[f"BB_LOWER_{window}"] = sma - k * std
    df[f"BB_STD_{window}"]   = std
    return df


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper — add a standard set of indicators used across strategies."""
    df = add_sma(df, 20)
    df = add_sma(df, 50)
    df = add_sma(df, 200)
    df = add_ema(df, 20)
    df = add_volatility(df, 20)
    df = add_atr(df, 14)
    df = add_rsi(df, 14)
    df = add_bollinger_bands(df, 20, 2.0)
    return df
