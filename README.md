# Quant Research Lab + Trading Terminal

A modular systematic trading desk built with Streamlit + Plotly.

## Project Structure

```
quant_terminal/
├── app.py            ← Streamlit UI (tabs, controls, charts)
├── data.py           ← yfinance fetch + caching
├── indicators.py     ← SMA, EMA, Bollinger, RSI, ATR, vol
├── strategies.py     ← Momentum, mean reversion, regime-aware, rotation
├── regime.py         ← Trend + volatility regime classification
├── events.py         ← Event flagging + forward return impact
├── backtest.py       ← Portfolio simulation engine (with costs)
├── metrics.py        ← CAGR, Sharpe, Sortino, drawdown, win rate
├── walkforward.py    ← Rolling optimisation framework
├── requirements.txt
└── data/             ← Auto-created CSV cache
```

## Features

| Tab | What it does |
|-----|-------------|
| Home Terminal | Regime badge, equity curve, drawdown, KPIs |
| Strategy Lab | Parameter heatmap, signal chart, walk-forward |
| Event Impact | FOMC event vs non-event return comparison |
| Portfolio | Rotation basket backtester |
| Trade Log | Entry/exit table with P&L |

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture Notes

- **No look-ahead bias**: signals are shifted 1 day before position sizing
- **Transaction costs**: commission + slippage deducted on every trade
- **Regime detection**: 200-SMA slope + volatility percentile (Trending / Choppy / HighVol)
- **Walk-forward**: optimises fast/slow SMA on rolling train window, applies to next test window
- **Rotation**: picks highest-momentum asset in basket; falls back to TLT in drawdown

## Extending

- Add new strategies in `strategies.py` and wire them up in `app.py`'s strategy selector
- Add new indicators in `indicators.py` (pure functions, no side effects)
- Swap placeholder FOMC dates in `events.py` for a real economic calendar API
- Add multi-parameter optimisation in `walkforward.py`
