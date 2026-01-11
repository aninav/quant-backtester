import yfinance as yf
import os
import pandas as pd
#import strategies from strategies.py
from strategies import add_indicators, momentum_strategy, mean_reversion_strategy

if not os.path.exists("data"):
    os.makedirs("data")
data = yf.download("QQQ", start="2020-01-01", end="2025-01-01")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns] 

data.reset_index(inplace=True)

data.to_csv("data/qqq.csv", index=False)
print("Data downloaded and saved! Columns are:", data.columns)

#take the data by date
df = pd.read_csv("data/qqq.csv", index_col='Date', parse_dates=True)
print(df.head())
#run strategies
df = add_indicators(df)
df = momentum_strategy(df)
df= mean_reversion_strategy(df)

df['Return'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']
df['Equity'] = (1+ df['Strategy_Return']).cumprod()

print(df[['Close', 'Signal', 'Equity']].tail())

