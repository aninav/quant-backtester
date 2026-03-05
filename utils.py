"""
utils.py
--------
Small shared helpers.

The key export is `validate_position()`, a contract checker called by
run_backtest() before the simulation loop. Raises immediately with a clear
message so bugs surface at the source rather than silently corrupting results.
"""

import numpy as np
import pandas as pd


def validate_position(
    signal: pd.Series,
    allow_short: bool = False,
    warmup_bars: int = 200,
) -> None:
    """
    Assert that a Signal series satisfies the strategy contract:

    1. Values are in the allowed set ({0,1} or {-1,0,1}).
    2. No NaN values after the warmup period (first `warmup_bars` rows may be NaN
       due to rolling indicators; that is expected and allowed).
    3. Does NOT check for shifting — shifting is backtest.py's responsibility.

    Raises ValueError with a descriptive message on any violation.

    Parameters
    ----------
    signal      : raw Signal series from a strategy function (NOT yet shifted)
    allow_short : if True, -1 is a valid value
    warmup_bars : number of leading rows allowed to be NaN (indicator warm-up)
    """
    allowed = {-1, 0, 1} if allow_short else {0, 1}

    # --- Check 1: NaNs after warmup ----------------------------------------
    post_warmup = signal.iloc[warmup_bars:]
    nan_count = post_warmup.isna().sum()
    if nan_count > 0:
        first_nan = post_warmup[post_warmup.isna()].index[0]
        raise ValueError(
            f"Signal has {nan_count} NaN value(s) after warmup (first at {first_nan}). "
            "Fill NaNs before passing to backtest."
        )

    # --- Check 2: Disallowed values -----------------------------------------
    filled = signal.dropna()
    unique_vals = set(filled.unique())
    bad_vals = unique_vals - allowed
    if bad_vals:
        raise ValueError(
            f"Signal contains disallowed values: {bad_vals}. "
            f"Allowed set is {allowed} (allow_short={allow_short})."
        )
