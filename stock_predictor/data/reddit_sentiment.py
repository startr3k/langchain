"""Historical Reddit sentiment — time-aligned via Arctic Shift API.

Fetches historical Reddit posts and comments mentioning a ticker from
finance subreddits (wallstreetbets, stocks, investing, stockmarket)
and computes rolling sentiment features.  Each feature is computed
using only data available on or before the row date (no look-ahead).
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
from textblob import TextBlob

logger = logging.getLogger(__name__)

REDDIT_SENTIMENT_FEATURES = [
    "reddit_mention_count_7d",
    "reddit_mean_sentiment_7d",
    "reddit_bullish_ratio_7d",
]

ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"
FINANCE_SUBREDDITS = "wallstreetbets,stocks,investing,stockmarket"
_RATE_LIMIT_DELAY = 1.0  # seconds between API calls


def _fetch_posts(
    ticker: str, after: str, before: str, limit: int = 100,
) -> list[dict]:
    """Fetch Reddit posts mentioning ticker from Arctic Shift."""
    try:
        resp = requests.get(
            f"{ARCTIC_SHIFT_BASE}/posts/search",
            params={
                "query": ticker,
                "subreddit": FINANCE_SUBREDDITS,
                "after": after,
                "before": before,
                "limit": limit,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json().get("data") or []
            return data
    except Exception:
        logger.debug("Arctic Shift posts request failed for %s", ticker)
    return []


def _fetch_comments(
    ticker: str, after: str, before: str, limit: int = 100,
) -> list[dict]:
    """Fetch Reddit comments mentioning ticker from Arctic Shift."""
    try:
        resp = requests.get(
            f"{ARCTIC_SHIFT_BASE}/comments/search",
            params={
                "body": ticker,
                "subreddit": FINANCE_SUBREDDITS,
                "after": after,
                "before": before,
                "limit": limit,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json().get("data") or []
            return data
    except Exception:
        logger.debug("Arctic Shift comments request failed for %s", ticker)
    return []


def _sentiment_polarity(text: str) -> float:
    """Compute sentiment polarity [-1, 1] using TextBlob."""
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0


def get_reddit_sentiment_history(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch Reddit mentions and sentiment for a ticker over a date range.

    Returns a DataFrame indexed by date with columns:
    - created_utc: timestamp of the post/comment
    - text: post title or comment body
    - polarity: sentiment polarity [-1, 1]
    """
    posts = _fetch_posts(ticker, after=start_date, before=end_date, limit=500)
    time.sleep(_RATE_LIMIT_DELAY)
    comments = _fetch_comments(ticker, after=start_date, before=end_date, limit=500)
    time.sleep(_RATE_LIMIT_DELAY)

    records = []
    for p in posts:
        ts = p.get("created_utc", 0)
        title = p.get("title", "")
        selftext = p.get("selftext", "")
        text = f"{title} {selftext}".strip()
        if text:
            records.append({
                "date": pd.Timestamp.utcfromtimestamp(ts).normalize(),
                "text": text,
                "polarity": _sentiment_polarity(text),
            })

    for c in comments:
        ts = c.get("created_utc", 0)
        body = c.get("body", "")
        if body and body != "[deleted]" and body != "[removed]":
            records.append({
                "date": pd.Timestamp.utcfromtimestamp(ts).normalize(),
                "text": body,
                "polarity": _sentiment_polarity(body),
            })

    if not records:
        return pd.DataFrame(columns=["date", "text", "polarity"])

    return pd.DataFrame(records)


def align_reddit_sentiment_to_dates(
    ticker: str,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create time-aligned Reddit sentiment features.

    For each date, computes rolling 7-day sentiment metrics using
    only posts/comments with timestamps on or before that date.
    This ensures zero look-ahead bias.
    """
    if len(dates) == 0:
        result = pd.DataFrame(index=dates)
        for col in REDDIT_SENTIMENT_FEATURES:
            result[col] = np.nan
        return result

    # Fetch the full history with a 7-day buffer before the first date
    start = (dates.min() - timedelta(days=10)).strftime("%Y-%m-%d")
    end = (dates.max() + timedelta(days=1)).strftime("%Y-%m-%d")

    history = get_reddit_sentiment_history(ticker, start, end)

    if history.empty:
        result = pd.DataFrame(index=dates)
        for col in REDDIT_SENTIMENT_FEATURES:
            result[col] = np.nan
        return result

    # Ensure date column is timezone-naive for comparison
    history["date"] = pd.to_datetime(history["date"]).dt.tz_localize(None)

    rows = []
    for d in dates:
        d_ts = pd.Timestamp(d)
        window_start = d_ts - timedelta(days=7)
        window = history[
            (history["date"] >= window_start) & (history["date"] <= d_ts)
        ]

        n_mentions = len(window)
        if n_mentions == 0:
            rows.append({
                "reddit_mention_count_7d": 0.0,
                "reddit_mean_sentiment_7d": 0.0,
                "reddit_bullish_ratio_7d": 0.0,
            })
        else:
            mean_pol = window["polarity"].mean()
            bullish_ratio = (window["polarity"] > 0).sum() / n_mentions
            rows.append({
                "reddit_mention_count_7d": float(n_mentions),
                "reddit_mean_sentiment_7d": round(mean_pol, 4),
                "reddit_bullish_ratio_7d": round(bullish_ratio, 4),
            })

    return pd.DataFrame(rows, index=dates)
