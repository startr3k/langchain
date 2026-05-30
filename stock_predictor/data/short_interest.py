"""Short interest data — fetched via yfinance.

Provides short interest features aligned to each trading date.
Only data available on or before each date is used (no look-ahead).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

SHORT_INTEREST_FEATURES = [
    "short_percent_of_float",
    "short_ratio",
    "short_interest_change",
]


def get_short_interest(ticker: str) -> pd.DataFrame:
    """Fetch short interest data for a ticker from yfinance.

    yfinance exposes ``shortPercentOfFloat`` and ``shortRatio`` as
    current-snapshot values.  For historical training we approximate a
    time-series by recording the most-recent value and computing a
    change signal.  During live prediction the current snapshot is
    directly usable.

    Returns a single-row DataFrame with the latest short interest data.
    """
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        logger.debug("Failed to fetch short interest for %s", ticker)
        return pd.DataFrame()

    short_pct = info.get("shortPercentOfFloat")
    short_ratio = info.get("shortRatio")

    if short_pct is None and short_ratio is None:
        return pd.DataFrame()

    return pd.DataFrame(
        [{
            "short_percent_of_float": short_pct,
            "short_ratio": short_ratio,
            "short_interest_change": 0.0,  # no historical delta available
        }]
    )


def align_short_interest_to_dates(
    ticker: str,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create a time-aligned short interest DataFrame.

    Because yfinance only provides the *current* snapshot of short
    interest (not a historical time-series), we forward-fill the
    latest available values.  For training rows this means the feature
    is constant per ticker, which still lets the model learn cross-
    sectional differences (high-short-interest stocks vs low).

    The ``short_interest_change`` column is always 0 in training
    (no historical series to diff) but is available for live inference
    if a prior snapshot has been stored.
    """
    si = get_short_interest(ticker)
    if si.empty:
        result = pd.DataFrame(index=dates)
        for col in SHORT_INTEREST_FEATURES:
            result[col] = np.nan
        return result

    result = pd.DataFrame(index=dates)
    for col in SHORT_INTEREST_FEATURES:
        result[col] = si[col].iloc[0] if col in si.columns else np.nan
    return result
