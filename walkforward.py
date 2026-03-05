"""
walkforward.py
--------------
Rolling walk-forward optimisation framework.

For each fold:
  1. Train window  → grid-search for best momentum parameters (max Sharpe)
  2. Test window   → apply best params to out-of-sample slice
  3. Aggregate all test slices → single OOS equity curve

Guarantees:
  • Indicators are re-built from scratch per fold (no data leakage)
  • Positions are forced to 0 at fold boundaries (flatten=True by default)
  • Validation is disabled inside the loop for speed; it already ran at strategy level
"""

import itertools

import numpy as np
import pandas as pd

from backtest import run_backtest
from indicators import add_sma, add_bollinger_bands
from metrics import sharpe_ratio
from strategies import momentum_signal


# ---------------------------------------------------------------------------
# Parameter Grids
# ---------------------------------------------------------------------------

MOMENTUM_GRID: dict[str, list] = {
    "fast": [10, 15, 20, 25, 30],
    "slow": [40, 50, 60, 75, 100],
}


# ---------------------------------------------------------------------------
# Per-fold optimiser
# ---------------------------------------------------------------------------

def optimise_momentum(
    df_train: pd.DataFrame,
    param_grid: dict = MOMENTUM_GRID,
    commission_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> dict:
    """
    Grid search over (fast, slow) SMA pairs on the training slice.
    Returns the dict of params that maximises Sharpe ratio.
    Falls back to (20, 50) if no combo produces a valid Sharpe.
    """
    best_sharpe = -np.inf
    best_params = {"fast": 20, "slow": 50}

    for fast, slow in itertools.product(param_grid["fast"], param_grid["slow"]):
        if fast >= slow:
            continue
        try:
            tmp = df_train.copy()
            tmp = add_sma(tmp, fast)
            tmp = add_sma(tmp, slow)
            tmp = momentum_signal(tmp, fast, slow)
            tmp = run_backtest(
                tmp,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                validate=False,   # skip validation inside the hot loop
            )
            sr = sharpe_ratio(tmp["Net_Return"])
        except Exception:
            continue

        if np.isfinite(sr) and sr > best_sharpe:
            best_sharpe = sr
            best_params = {"fast": fast, "slow": slow}

    return best_params


# ---------------------------------------------------------------------------
# Walk-Forward Engine
# ---------------------------------------------------------------------------

def walk_forward(
    df: pd.DataFrame,
    train_days: int = 252,
    test_days: int = 63,
    commission_bps: float = 5.0,
    slippage_bps: float = 2.0,
    strategy: str = "momentum",
    flatten_at_boundary: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Roll train/test windows across the full DataFrame.

    Parameters
    ----------
    df                   : full prepared OHLCV DataFrame (indicators NOT required — added per fold)
    train_days           : number of bars in each training window
    test_days            : number of bars in each test window
    commission_bps       : one-way commission per leg
    slippage_bps         : one-way slippage per leg
    strategy             : "momentum" (mean_reversion optimiser is a TODO)
    flatten_at_boundary  : if True, force Position=0 on the first bar of each fold
                           so we never carry a position across fold boundaries

    Returns
    -------
    oos_df   : concatenated out-of-sample result DataFrame with continuous equity
    fold_log : list of per-fold dicts: window dates, best params, OOS Sharpe
    """
    results  = []
    fold_log = []
    n        = len(df)
    start_idx = train_days
    fold_num  = 0

    while start_idx + test_days <= n:
        fold_num   += 1
        train_slice = df.iloc[start_idx - train_days : start_idx].copy()
        test_slice  = df.iloc[start_idx : start_idx + test_days].copy()

        # --- Optimise on train ---------------------------------------------
        if strategy == "momentum":
            best_params = optimise_momentum(
                train_slice,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
            )
        else:
            # TODO: wire mean_reversion optimiser
            best_params = {"fast": 20, "slow": 50}

        # --- Build indicators on test slice (clean, no leakage) -----------
        # We need a context window so SMAs have enough history.
        # Use the tail of the train slice as context, then trim back.
        context_size = best_params["slow"] + 10
        context_start = max(0, start_idx - context_size)
        with_context = df.iloc[context_start : start_idx + test_days].copy()

        try:
            with_context = add_sma(with_context, best_params["fast"])
            with_context = add_sma(with_context, best_params["slow"])
            with_context = momentum_signal(with_context, best_params["fast"], best_params["slow"])
            with_context = run_backtest(
                with_context,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                validate=False,
            )
            # Trim back to just the test window
            test_result = with_context.iloc[-test_days:].copy()
        except Exception as e:
            start_idx += test_days
            continue

        # --- Flatten at fold boundary (optional) ---------------------------
        if flatten_at_boundary:
            # Force the very first position bar to 0; it will self-correct on bar 2
            test_result.iloc[0, test_result.columns.get_loc("Position")] = 0

        fold_log.append({
            "Fold":        fold_num,
            "Train_Start": df.index[start_idx - train_days],
            "Train_End":   df.index[start_idx - 1],
            "Test_Start":  df.index[start_idx],
            "Test_End":    df.index[min(start_idx + test_days - 1, n - 1)],
            "Best_Fast":   best_params["fast"],
            "Best_Slow":   best_params["slow"],
            "OOS_Sharpe":  round(sharpe_ratio(test_result["Net_Return"]), 2),
        })

        results.append(test_result)
        start_idx += test_days

    if not results:
        return pd.DataFrame(), fold_log

    oos = pd.concat(results)

    # Rebuild continuous equity from OOS net returns (chain the folds)
    oos["Equity"] = (1 + oos["Net_Return"]).cumprod() * 100_000

    return oos, fold_log
