"""GDELT news sentiment features.

Uses the GDELT DOC 2.0 API to fetch historical news tone/volume for
stock tickers. All features are time-aligned — only articles published
on or before each row date are used.

Features:
- news_tone_avg_7d: average article tone in last 7 days
- news_volume_7d: number of articles mentioning ticker in last 7 days
- news_tone_change_7d: tone change vs prior 7-day window
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

GDELT_FEATURES = [
    "news_tone_avg_7d",
    "news_volume_7d",
    "news_tone_change_7d",
]

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def _fetch_gdelt_tone(
    query: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Fetch tone timeline from GDELT DOC API.

    Args:
        query: search query (e.g. company name or ticker)
        start_date: YYYYMMDDHHMMSS format
        end_date: YYYYMMDDHHMMSS format

    Returns:
        dict with 'tone' (average) and 'volume' (article count)
    """
    try:
        params = {
            "query": query,
            "mode": "timelinetone",
            "startdatetime": start_date,
            "enddatetime": end_date,
            "format": "json",
            "maxrecords": 250,
        }
        resp = requests.get(GDELT_DOC_API, params=params, timeout=15)
        time.sleep(5.5)  # GDELT rate limit: 1 request per 5 seconds

        if resp.status_code != 200:
            return {"tone": np.nan, "volume": 0}

        data = resp.json()
        timeline = data.get("timeline", [])
        if not timeline:
            return {"tone": np.nan, "volume": 0}

        # Timeline is a list of series; each has 'data' with date/value pairs
        tones = []
        volumes = []
        for series in timeline:
            for point in series.get("data", []):
                val = point.get("value", 0)
                tones.append(val)
                volumes.append(1)

        if tones:
            return {"tone": float(np.mean(tones)), "volume": len(tones)}
        return {"tone": np.nan, "volume": 0}

    except Exception:
        return {"tone": np.nan, "volume": 0}


def get_gdelt_daily_sentiment(
    ticker: str,
    company_name: str | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch daily GDELT news sentiment for a ticker.

    Queries GDELT in weekly chunks to build a daily tone timeline.
    Uses company name for better matching (e.g. "Apple" instead of "AAPL").

    Returns DataFrame with columns: date, tone, volume
    """
    query = f'"{ticker}" OR "{company_name}"' if company_name else f'"{ticker}"'

    if start_date is None:
        start_date = pd.Timestamp.now() - pd.Timedelta(days=365 * 5)
    if end_date is None:
        end_date = pd.Timestamp.now()

    # GDELT DOC API supports timelinetone mode which returns daily tone
    # Query the full range — API handles it efficiently
    sd = start_date.strftime("%Y%m%d%H%M%S")
    ed = end_date.strftime("%Y%m%d%H%M%S")

    try:
        params = {
            "query": query,
            "mode": "timelinetone",
            "startdatetime": sd,
            "enddatetime": ed,
            "format": "json",
            "maxrecords": 250,
        }
        resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
        time.sleep(0.3)

        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        timeline = data.get("timeline", [])
        if not timeline:
            return pd.DataFrame()

        records = []
        for series in timeline:
            for point in series.get("data", []):
                date_str = point.get("date", "")
                val = point.get("value", 0)
                if date_str:
                    try:
                        dt = pd.Timestamp(date_str)
                        records.append({"date": dt, "tone": val, "volume": 1})
                    except Exception:
                        pass

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # Aggregate by date (multiple series may overlap)
        df = df.groupby("date").agg(
            tone=("tone", "mean"),
            volume=("volume", "sum"),
        ).reset_index()
        df = df.sort_values("date").reset_index(drop=True)
        return df

    except Exception:
        logger.debug("Error fetching GDELT data for %s", ticker)
        return pd.DataFrame()


def _compute_gdelt_features_at_date(
    daily_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback_days: int = 7,
) -> dict:
    """Compute GDELT features using only data on or before as_of_date."""
    if daily_df.empty:
        return {
            "news_tone_avg_7d": np.nan,
            "news_volume_7d": np.nan,
            "news_tone_change_7d": np.nan,
        }

    cutoff = as_of_date - pd.Timedelta(days=lookback_days)
    prior_cutoff = cutoff - pd.Timedelta(days=lookback_days)

    # Current window
    current = daily_df[
        (daily_df["date"] >= cutoff) & (daily_df["date"] <= as_of_date)
    ]
    # Prior window (for change computation)
    prior = daily_df[
        (daily_df["date"] >= prior_cutoff) & (daily_df["date"] < cutoff)
    ]

    if current.empty:
        return {
            "news_tone_avg_7d": 0.0,
            "news_volume_7d": 0.0,
            "news_tone_change_7d": 0.0,
        }

    tone_avg = float(current["tone"].mean())
    volume = float(current["volume"].sum())
    prior_tone = float(prior["tone"].mean()) if not prior.empty else tone_avg
    tone_change = tone_avg - prior_tone

    return {
        "news_tone_avg_7d": tone_avg,
        "news_volume_7d": volume,
        "news_tone_change_7d": tone_change,
    }


def align_gdelt_to_dates(
    ticker: str,
    dates: pd.DatetimeIndex | pd.Index,
    company_name: str | None = None,
) -> pd.DataFrame:
    """Create time-aligned GDELT sentiment DataFrame.

    For each date, computes 7-day rolling tone and volume.
    Only articles published on or before the row date are used.
    """
    if len(dates) == 0:
        return pd.DataFrame(
            {col: np.nan for col in GDELT_FEATURES},
            index=range(0),
        )

    min_date = pd.Timestamp(min(dates)) - pd.Timedelta(days=14)
    max_date = pd.Timestamp(max(dates))

    daily_df = get_gdelt_daily_sentiment(
        ticker, company_name=company_name,
        start_date=min_date, end_date=max_date,
    )

    rows = []
    for d in dates:
        features = _compute_gdelt_features_at_date(daily_df, pd.Timestamp(d))
        rows.append(features)

    return pd.DataFrame(rows, index=dates)
