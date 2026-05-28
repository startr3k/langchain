"""Historical earnings data from YFinance.

Provides time-aligned earnings surprise and days-to-earnings features.
Each training row gets the most recent earnings surprise that was
actually known at that date, avoiding look-ahead bias.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

EARNINGS_FEATURES = [
    "earnings_surprise_pct",
    "earnings_eps_actual",
    "days_since_last_earnings",
    "days_to_next_earnings",
]


def get_earnings_history(ticker: str) -> pd.DataFrame:
    """Fetch historical earnings dates with surprise data.

    Returns:
        DataFrame with earnings date index and surprise columns,
        sorted oldest first.
    """
    try:
        stock = yf.Ticker(ticker)
        ed = stock.earnings_dates
        if ed is None or ed.empty:
            return pd.DataFrame()

        # Strip timezone from index to avoid alignment issues
        idx = ed.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        idx = pd.to_datetime(idx)

        result = pd.DataFrame({
            "earnings_surprise_pct": pd.to_numeric(ed["Surprise(%)"].values, errors="coerce"),
            "earnings_eps_actual": pd.to_numeric(ed["Reported EPS"].values, errors="coerce"),
            "earnings_eps_estimate": pd.to_numeric(ed["EPS Estimate"].values, errors="coerce"),
        }, index=idx)

        # Only keep rows where earnings have actually been reported
        result = result.dropna(subset=["earnings_eps_actual"])
        result = result.sort_index()
        return result

    except Exception:
        logger.debug("Could not fetch earnings history for %s", ticker)
        return pd.DataFrame()


def align_earnings_to_dates(
    earnings_df: pd.DataFrame,
    dates: pd.DatetimeIndex | pd.Index,
) -> pd.DataFrame:
    """Align earnings data to training row dates.

    For each date, finds the most recent past earnings (for surprise)
    and the next future earnings (for days_to_next_earnings).
    """
    if earnings_df.empty:
        return pd.DataFrame(
            {col: np.nan for col in EARNINGS_FEATURES},
            index=range(len(dates)),
        )

    earnings_dates = earnings_df.index.values
    aligned_rows: list[dict] = []

    for d in dates:
        ts = pd.Timestamp(d)
        rec: dict = {}

        # Most recent past earnings
        past_mask = earnings_dates <= ts
        if past_mask.any():
            past_idx = np.where(past_mask)[0][-1]
            past_row = earnings_df.iloc[past_idx]
            past_date = pd.Timestamp(earnings_dates[past_idx])
            rec["earnings_surprise_pct"] = past_row.get("earnings_surprise_pct", np.nan)
            rec["earnings_eps_actual"] = past_row.get("earnings_eps_actual", np.nan)
            rec["earnings_eps_estimate"] = past_row.get("earnings_eps_estimate", np.nan)
            rec["days_since_last_earnings"] = float((ts - past_date).days)
        else:
            rec["earnings_surprise_pct"] = np.nan
            rec["earnings_eps_actual"] = np.nan
            rec["earnings_eps_estimate"] = np.nan
            rec["days_since_last_earnings"] = np.nan

        # Next future earnings
        future_mask = earnings_dates > ts
        if future_mask.any():
            future_idx = np.where(future_mask)[0][0]
            future_date = pd.Timestamp(earnings_dates[future_idx])
            rec["days_to_next_earnings"] = float((future_date - ts).days)
        else:
            rec["days_to_next_earnings"] = np.nan

        aligned_rows.append(rec)

    return pd.DataFrame(aligned_rows)
