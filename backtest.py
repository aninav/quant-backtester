"""
backtest.py
-----------
Core portfolio simulation engine.

COST MODEL
----------
Transaction cost is charged once per unit of position change:

    cost[T] = abs(Position[T] - Position[T-1]) * (commission_bps + slippage_bps) / 10_000

Interpretation: each basis-point figure represents one-way cost (entry OR exit).
A complete round-trip (e.g., 0 → 1 on day A, then 1 → 0 on day B) is charged
on both days — naturally accumulating to 2 × one-way cost in total.
This is intentional and mirrors real execution: you pay to get in, and pay again
to get out.

Do NOT multiply by 2 here — that double-charges single-leg transitions.
"""

import numpy as np
import pandas as pd

from utils import validate_position


def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "Signal",
    price_col: str = "Close",
    commission_bps: float = 5.0,
    slippage_bps: float = 2.0,
    allow_short: bool = False,
    initial_capital: float = 100_000.0,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Simulate a strategy from signals.

    Parameters
    ----------
    df              : DataFrame with price and signal columns
    signal_col      : column name for raw signal (1, 0, -1)
    price_col       : column to compute returns from
    commission_bps  : one-way commission in basis points
    slippage_bps    : one-way slippage in basis points
    allow_short     : if False, clips Signal to [0, 1]
    initial_capital : starting portfolio value
    validate        : run contract checks before simulation (disable for speed in WF loops)

    Returns
    -------
    df with added columns:
      Position, Return, Gross_Return, Transaction_Cost, Net_Return, Equity, Drawdown, Traded
    """
    df = df.copy()

    # --- 1. Clip / prepare signal ------------------------------------------
    signal = df[signal_col].fillna(0)
    if not allow_short:
        signal = signal.clip(0, 1)

    # --- 2. Contract validation --------------------------------------------
    if validate:
        validate_position(signal, allow_short=allow_short)

    # --- 3. Shift signal → position (T+1 execution) -----------------------
    df["Position"] = signal.shift(1).fillna(0)

    # --- 4. Daily market returns -------------------------------------------
    df["Return"] = df[price_col].pct_change()

    # --- 5. Position change (absolute units traded) -------------------------
    pos_change       = df["Position"].diff().abs()
    df["Traded"]     = pos_change > 0

    # --- 6. Transaction cost — one-way per leg, charged on each transition --
    #   cost_rate = total one-way cost per unit of notional traded
    cost_rate               = (commission_bps + slippage_bps) / 10_000
    df["Transaction_Cost"]  = pos_change * cost_rate

    # --- 7. Net return ------------------------------------------------------
    df["Gross_Return"] = df["Position"] * df["Return"]
    df["Net_Return"]   = df["Gross_Return"] - df["Transaction_Cost"]

    # --- 8. Equity curve ----------------------------------------------------
    df["Equity"] = initial_capital * (1 + df["Net_Return"]).cumprod()

    # --- 9. Drawdown --------------------------------------------------------
    rolling_peak    = df["Equity"].cummax()
    df["Drawdown"]  = (df["Equity"] - rolling_peak) / rolling_peak

    return df


def build_trade_log(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Extract entry/exit pairs from the Position column.

    Returns a DataFrame with columns:
      Entry_Date, Exit_Date, Entry_Price, Exit_Price, Return_pct, Side, Bars_Held
    """
    trades   = []
    position = df["Position"]
    prices   = df[price_col]
    changes  = position.diff().fillna(0)

    in_trade   = False
    entry_date = None
    entry_px   = None
    side       = None

    for date, chg in changes.items():
        if not in_trade and chg != 0:
            in_trade   = True
            entry_date = date
            entry_px   = prices[date]
            side       = "Long" if chg > 0 else "Short"

        elif in_trade and (chg != 0 or position[date] == 0):
            exit_px   = prices[date]
            ret       = (exit_px - entry_px) / entry_px if side == "Long" \
                        else (entry_px - exit_px) / entry_px
            bars_held = (pd.Timestamp(date) - pd.Timestamp(entry_date)).days

            trades.append({
                "Entry_Date":  entry_date,
                "Exit_Date":   date,
                "Entry_Price": round(entry_px, 2),
                "Exit_Price":  round(exit_px, 2),
                "Return_pct":  round(ret * 100, 2),
                "Side":        side,
                "Bars_Held":   bars_held,
            })
            in_trade = False

            # Immediately re-enter on a flip
            if chg != 0 and position[date] != 0:
                in_trade   = True
                entry_date = date
                entry_px   = prices[date]
                side       = "Long" if position[date] > 0 else "Short"

    return pd.DataFrame(trades)
