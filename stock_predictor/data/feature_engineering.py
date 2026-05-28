"""Feature engineering pipeline that combines YFinance and sentiment data."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from stock_predictor.data.sentiment import get_sentiment_features
from stock_predictor.data.yfinance_client import (
    compute_technical_features,
    get_fundamentals_features,
    get_stock_data,
)

logger = logging.getLogger(__name__)

# Technical feature columns produced by compute_technical_features
TECHNICAL_FEATURES = [
    "Return_1d", "Return_5d", "Return_20d", "Return_60d",
    "SMA_5", "SMA_10", "SMA_20", "SMA_50", "SMA_200",
    "EMA_5", "EMA_10", "EMA_20", "EMA_50", "EMA_200",
    "Price_to_SMA_20", "Price_to_SMA_50", "Price_to_SMA_200",
    "Volatility_20d", "Volatility_60d",
    "Volume_Ratio", "Volume_Spike", "Volume_Spike_Magnitude",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "BB_Position",
    "ATR_14",
    "OBV", "OBV_SMA_20",
]

FUNDAMENTAL_FEATURES = [
    "marketCap", "trailingPE", "forwardPE", "priceToBook",
    "dividendYield", "beta", "revenueGrowth", "earningsGrowth",
    "profitMargins", "returnOnEquity", "debtToEquity",
    "currentRatio", "numberOfAnalystOpinions",
]

SENTIMENT_FEATURES = [
    "sentiment_mean_polarity", "sentiment_std_polarity",
    "sentiment_max_polarity", "sentiment_min_polarity",
    "sentiment_mean_subjectivity", "sentiment_total_mentions",
    "reddit_mention_count", "reddit_mean_polarity",
    "reddit_mean_score", "reddit_total_comments",
    "finviz_mention_count", "finviz_mean_polarity",
    "stocktwits_mention_count", "stocktwits_mean_polarity",
    "stocktwits_bullish_count", "stocktwits_bearish_count",
    "stocktwits_bull_bear_ratio",
]

ALL_FEATURE_NAMES = TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES + SENTIMENT_FEATURES

TARGET_COLUMN = "Forward_Return_3M"


def build_training_row(
    ticker: str,
    include_sentiment: bool = True,
) -> dict | None:
    """Build a single feature row (latest data point) for a ticker.

    Args:
        ticker: Stock ticker symbol.
        include_sentiment: Whether to add sentiment features.

    Returns:
        Dictionary of feature values or None on failure.
    """
    try:
        df = get_stock_data(ticker, period="2y")
        if df.empty or len(df) < 200:
            logger.warning("Insufficient data for %s (%d rows)", ticker, len(df))
            return None

        df = compute_technical_features(df)
        latest = df.iloc[-1]
        row: dict = {"Ticker": ticker}
        for col in TECHNICAL_FEATURES:
            row[col] = latest.get(col, np.nan)

        # Fundamentals
        fundamentals = get_fundamentals_features(ticker)
        for col in FUNDAMENTAL_FEATURES:
            row[col] = fundamentals.get(col, np.nan)

        # Sentiment
        if include_sentiment:
            sentiment = get_sentiment_features(ticker)
            for col in SENTIMENT_FEATURES:
                row[col] = sentiment.get(col, 0.0)

        return row
    except Exception:
        logger.exception("Error building feature row for %s", ticker)
        return None


def build_training_dataset(
    tickers: list[str],
    include_sentiment: bool = True,
) -> pd.DataFrame:
    """Build a training dataset across multiple tickers.

    For training, we use historical data points. The target is the actual
    3-month forward return computed from historical prices.

    Args:
        tickers: List of ticker symbols.
        include_sentiment: Whether to add sentiment features.

    Returns:
        DataFrame ready for model training.
    """
    all_rows: list[dict] = []

    for ticker in tickers:
        try:
            df = get_stock_data(ticker, period="5y")
            if df.empty or len(df) < 300:
                logger.warning("Skipping %s — insufficient history", ticker)
                continue

            df = compute_technical_features(df)

            # Compute 3-month forward return (~63 trading days)
            df[TARGET_COLUMN] = df["Close"].shift(-63) / df["Close"] - 1

            # Fundamentals (static, applied to all rows)
            fundamentals = get_fundamentals_features(ticker)

            # Sentiment (current snapshot, applied to all rows as proxy)
            sentiment = {}
            if include_sentiment:
                sentiment = get_sentiment_features(ticker)

            # Sample rows that have valid target and enough history
            valid_mask = df[TARGET_COLUMN].notna() & df["SMA_200"].notna()
            valid_df = df[valid_mask]

            if valid_df.empty:
                continue

            # Sample up to 200 data points per ticker to leverage 5yr of data
            sample_indices = np.linspace(
                0, len(valid_df) - 1, min(200, len(valid_df)), dtype=int
            )
            sampled = valid_df.iloc[sample_indices]

            for _, row in sampled.iterrows():
                data_point: dict = {"Ticker": ticker}
                for col in TECHNICAL_FEATURES:
                    data_point[col] = row.get(col, np.nan)
                for col in FUNDAMENTAL_FEATURES:
                    data_point[col] = fundamentals.get(col, np.nan)
                for col in SENTIMENT_FEATURES:
                    data_point[col] = sentiment.get(col, 0.0)
                data_point[TARGET_COLUMN] = row[TARGET_COLUMN]
                all_rows.append(data_point)

        except Exception:
            logger.exception("Error processing %s for training", ticker)

    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)
    logger.info(
        "Built training dataset: %d rows, %d features",
        len(result), len(result.columns) - 2,  # minus Ticker and target
    )
    return result
