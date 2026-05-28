"""Google Trends historical search interest as a sentiment proxy.

Uses the pytrends library to fetch time-series search interest for a
stock ticker, providing a historically accurate measure of public
attention/interest at each point in time.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRENDS_FEATURES = [
    "gtrends_interest",
    "gtrends_interest_sma_4w",
    "gtrends_interest_change_4w",
]


def get_google_trends(ticker: str, timeframe: str = "today 5-y") -> pd.DataFrame:
    """Fetch Google Trends interest over time for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        timeframe: Pytrends timeframe string.

    Returns:
        Weekly date-indexed DataFrame with interest columns, or empty DataFrame.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            from pytrends.request import TrendReq

            # Add a stock-related keyword to disambiguate
            kw = f"{ticker} stock"

            pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
            pytrends.build_payload([kw], cat=0, timeframe=timeframe, geo="US")
            df = pytrends.interest_over_time()

            if df is None or df.empty:
                return pd.DataFrame()

            result = pd.DataFrame(index=df.index)
            result["gtrends_interest"] = df[kw].values
            result["gtrends_interest_sma_4w"] = result["gtrends_interest"].rolling(4).mean()
            result["gtrends_interest_change_4w"] = result["gtrends_interest"].pct_change(4)

            return result

        except Exception:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.debug("Google Trends attempt %d failed for %s, retrying in %ds", attempt + 1, ticker, wait)
                time.sleep(wait)
            else:
                logger.debug("Could not fetch Google Trends for %s after %d attempts", ticker, max_retries)
                return pd.DataFrame()


def align_trends_to_dates(
    trends_df: pd.DataFrame,
    dates: pd.DatetimeIndex | pd.Index,
) -> pd.DataFrame:
    """Align weekly Google Trends data to daily training dates.

    Uses the most recent available weekly data point for each date.
    """
    if trends_df.empty:
        return pd.DataFrame(
            {col: np.nan for col in TRENDS_FEATURES},
            index=range(len(dates)),
        )

    feature_cols = [c for c in TRENDS_FEATURES if c in trends_df.columns]
    trends_dates = trends_df.index.values

    aligned_rows: list[dict] = []
    for d in dates:
        ts = pd.Timestamp(d)
        mask = trends_dates <= ts
        if mask.any():
            idx = np.where(mask)[0][-1]
            row = trends_df.iloc[idx]
            aligned_rows.append({col: row.get(col, np.nan) for col in feature_cols})
        else:
            aligned_rows.append({col: np.nan for col in feature_cols})

    return pd.DataFrame(aligned_rows)
