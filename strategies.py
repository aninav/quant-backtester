import pandas as pd

def add_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    return df

def momentum_strategy(df):
    df['Signal'] = 0
    df['Signal'][20:] = (df['SMA_20'][20:] > df['SMA_50'][20:]).astype(int)
    return df

def mean_reversion_strategy(df, k=1):
    df['Buy'] = df['Close'] < df['SMA_20'] - k*df['STD_20']
    df['Sell'] = df['Close'] > df['SMA_20'] + k*df['STD_20']
    return df
    