"""
events.py
---------
Identifies notable event days and computes their forward return impact.

For now, events are defined via a placeholder calendar (e.g. FOMC dates, earnings).
The module outputs:
  - Flagged DataFrame with an 'IsEvent' boolean column
  - Forward return comparison: event days vs non-event days
  - Volatility comparison

Replace PLACEHOLDER_EVENTS with a real data source (e.g. scraped FOMC calendar,
earnings API, economic calendar CSV) when needed.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Placeholder event calendar
# Extend this dict with real dates or load from a CSV / API
# ---------------------------------------------------------------------------
PLACEHOLDER_EVENTS: dict[str, list[str]] = {
    "FOMC": [
        "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
        "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
        "2020-12-16", "2021-01-27", "2021-03-17", "2021-04-28",
        "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03",
        "2021-12-15", "2022-01-26", "2022-03-16", "2022-05-04",
        "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02",
        "2022-12-14", "2023-02-01", "2023-03-22", "2023-05-03",
        "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01",
        "2023-12-13", "2024-01-31", "2024-03-20", "2024-05-01",
        "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07",
        "2024-12-18",
    ]
}


def flag_events(df: pd.DataFrame, event_type: str = "FOMC") -> pd.DataFrame:
    """
    Add an 'IsEvent' boolean column marking known event dates.

    Parameters
    ----------
    event_type : key in PLACEHOLDER_EVENTS (default "FOMC")
    """
    dates = pd.to_datetime(PLACEHOLDER_EVENTS.get(event_type, []))
    df["IsEvent"]   = df.index.isin(dates)
    df["EventType"] = event_type
    return df


def compute_event_impact(df: pd.DataFrame, forward_windows: list[int] = [1, 3, 5]) -> pd.DataFrame:
    """
    Compute mean forward returns and volatility on event vs non-event days.

    Returns a summary DataFrame with rows = [Event, Non-Event]
    and columns = forward return windows + volatility.

    Requires: 'Close' and 'IsEvent' columns.
    """
    if "IsEvent" not in df.columns:
        raise ValueError("Run flag_events() first.")

    df = df.copy()
    for w in forward_windows:
        df[f"Fwd_{w}d"] = df["Close"].pct_change(w).shift(-w)

    fwd_cols = [f"Fwd_{w}d" for w in forward_windows]

    event_rows     = df[df["IsEvent"]]
    non_event_rows = df[~df["IsEvent"]]

    summary = pd.DataFrame({
        "Group": ["Event", "Non-Event"],
        **{col: [
            event_rows[col].mean(),
            non_event_rows[col].mean()
        ] for col in fwd_cols},
        "Avg_DayVol": [
            event_rows["Close"].pct_change().abs().mean(),
            non_event_rows["Close"].pct_change().abs().mean(),
        ]
    }).set_index("Group")

    return summary.round(4)


def event_return_distribution(df: pd.DataFrame, window: int = 1) -> tuple[pd.Series, pd.Series]:
    """
    Return two Series of forward returns:
      - event_returns : returns on event days
      - non_event_returns : returns on non-event days

    Used for the distribution chart in the UI.
    """
    if "IsEvent" not in df.columns:
        raise ValueError("Run flag_events() first.")

    fwd = df["Close"].pct_change(window).shift(-window)
    event_returns     = fwd[df["IsEvent"]].dropna()
    non_event_returns = fwd[~df["IsEvent"]].dropna()

    return event_returns, non_event_returns
