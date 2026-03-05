"""
data.py
-------
Responsible for fetching, cleaning, and caching market data via yfinance.
All other modules consume DataFrames produced here.
"""

import os
import pandas as pd
import yfinance as yf

CACHE_DIR = "data"


def _cache_path(ticker: str, start: str, end: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.csv")


def fetch_data(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker from yfinance.
    Returns a clean DataFrame indexed by Date with columns:
    [Open, High, Low, Close, Volume]

    Parameters
    ----------
    ticker    : e.g. "QQQ"
    start     : "YYYY-MM-DD"
    end       : "YYYY-MM-DD"
    use_cache : skip download if CSV already exists
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = _cache_path(ticker, start, end)

    if use_cache and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
        return df

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Flatten MultiIndex columns if present (yfinance >= 0.2.x quirk)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.dropna(inplace=True)

    df.to_csv(cache_file)
    return df


def fetch_basket(tickers: list[str], start: str, end: str, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """
    Fetch data for a list of tickers. Returns {ticker: DataFrame}.
    Used by the portfolio rotation module.
    """
    return {t: fetch_data(t, start, end, use_cache) for t in tickers}


def align_basket(basket: dict[str, pd.DataFrame], price_col: str = "Close") -> pd.DataFrame:
    """
    Align multiple ticker price series on a common date index.
    Returns a wide DataFrame: columns = tickers, rows = dates.
    """
    frames = {ticker: df[price_col].rename(ticker) for ticker, df in basket.items()}
    aligned = pd.concat(frames, axis=1).dropna()
    return aligned
