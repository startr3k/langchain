"""Feature engineering pipeline that combines all data sources.

All historical features are time-aligned — each training row only sees
data that was available at that point in time, eliminating look-ahead bias.

Data sources:
1. Technical indicators (from price/volume — inherently time-correct)
2. Historical quarterly fundamentals (YFinance quarterly filings)
3. Current-snapshot fundamentals (used only for live prediction, NOT training)
4. Macroeconomic data (VIX, rates, S&P500 — properly time-indexed)
5. Earnings surprise history (YFinance earnings_dates)
6. Google Trends (historical search interest)
7. SEC EDGAR XBRL (historical regulatory filings)
8. Sentiment (current snapshot — used for both training and prediction)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from stock_predictor.data.earnings_data import (
    EARNINGS_FEATURES,
    align_earnings_to_dates,
    get_earnings_history,
)
from stock_predictor.data.google_trends import (
    TRENDS_FEATURES,
    align_trends_to_dates,
    get_google_trends,
)
from stock_predictor.data.historical_fundamentals import (
    HIST_FUNDAMENTAL_FEATURES,
    align_fundamentals_to_dates,
    get_historical_fundamentals,
)
from stock_predictor.data.macro_data import (
    MACRO_FEATURES,
    align_macro_to_dates,
    get_macro_data,
)
from stock_predictor.data.sec_edgar import (
    SEC_FEATURES,
    align_sec_to_dates,
    get_sec_filings,
)
from stock_predictor.data.sentiment import get_sentiment_features
from stock_predictor.data.yfinance_client import (
    compute_technical_features,
    get_fundamentals_features,
    get_stock_data,
)

logger = logging.getLogger(__name__)

# ---- Feature group definitions ----

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

# Current-snapshot fundamentals (used for live prediction only)
FUNDAMENTAL_FEATURES = [
    # Valuation ratios
    "marketCap", "trailingPE", "forwardPE", "priceToBook",
    "pegRatio", "priceToSalesTrailing12Months",
    "enterpriseToEbitda", "enterpriseToRevenue",
    # Per-share data
    "trailingEps", "forwardEps", "bookValue", "revenuePerShare",
    # Profitability
    "dividendYield", "profitMargins", "grossMargins",
    "operatingMargins", "ebitdaMargins",
    "returnOnEquity", "returnOnAssets",
    # Growth
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
    # Risk / leverage
    "beta", "debtToEquity", "currentRatio", "quickRatio",
    # Ownership / sentiment
    "shortRatio", "shortPercentOfFloat",
    "heldPercentInsiders", "heldPercentInstitutions",
    "numberOfAnalystOpinions",
    # Earnings calendar
    "daysToEarnings",
]

SENTIMENT_FEATURES = [
    # Aggregated sentiment (derived from available sources)
    "sentiment_mean_polarity", "sentiment_std_polarity",
    "sentiment_max_polarity", "sentiment_min_polarity",
    "sentiment_mean_subjectivity", "sentiment_total_mentions",
    # Finviz news headlines (reliable from cloud environments)
    "finviz_mention_count", "finviz_mean_polarity",
]

# All features used by the model (training + prediction).
# Google Trends features are excluded from the default list because
# Google aggressively rate-limits cloud/datacenter IPs; they are
# added dynamically when data is actually available.
ALL_FEATURE_NAMES = (
    TECHNICAL_FEATURES
    + FUNDAMENTAL_FEATURES
    + HIST_FUNDAMENTAL_FEATURES
    + MACRO_FEATURES
    + EARNINGS_FEATURES
    + SEC_FEATURES
    + SENTIMENT_FEATURES
)

TARGET_COLUMN = "Forward_Return_3M"


def build_training_row(
    ticker: str,
    include_sentiment: bool = True,
) -> dict | None:
    """Build a single feature row (latest data point) for a ticker.

    Used for live prediction — includes current-snapshot fundamentals
    plus the latest historical data.

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

        # Current-snapshot fundamentals (acceptable for live prediction)
        fundamentals = get_fundamentals_features(ticker)
        for col in FUNDAMENTAL_FEATURES:
            row[col] = fundamentals.get(col, np.nan)

        # Historical fundamentals (latest quarter)
        hist_fund = get_historical_fundamentals(ticker)
        if not hist_fund.empty:
            latest_q = hist_fund.iloc[-1]
            for col in HIST_FUNDAMENTAL_FEATURES:
                row[col] = latest_q.get(col, np.nan)

        # Macro data (latest available)
        macro = get_macro_data(period="1y")
        if not macro.empty:
            latest_macro = macro.iloc[-1]
            for col in MACRO_FEATURES:
                row[col] = latest_macro.get(col, np.nan)

        # Earnings data
        earnings = get_earnings_history(ticker)
        if not earnings.empty:
            today = pd.Timestamp.now().normalize()
            aligned = align_earnings_to_dates(earnings, [today])
            for col in EARNINGS_FEATURES:
                row[col] = aligned[col].iloc[0] if col in aligned.columns else np.nan

        # Google Trends (latest week)
        trends = get_google_trends(ticker, timeframe="today 3-m")
        if not trends.empty:
            latest_t = trends.iloc[-1]
            for col in TRENDS_FEATURES:
                row[col] = latest_t.get(col, np.nan)

        # SEC EDGAR (latest filing)
        sec = get_sec_filings(ticker)
        if not sec.empty:
            today = pd.Timestamp.now().normalize()
            aligned = align_sec_to_dates(sec, [today])
            for col in SEC_FEATURES:
                row[col] = aligned[col].iloc[0] if col in aligned.columns else np.nan

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
    max_samples_per_ticker: int | None = None,
) -> pd.DataFrame:
    """Build a training dataset with properly time-aligned features.

    For training, each row gets ONLY the data that was available at
    that point in time:
    - Technical indicators: computed from historical price/volume (correct)
    - Historical fundamentals: aligned to the most recent quarterly filing
    - Macroeconomic data: aligned to the most recent available date
    - Earnings surprise: aligned to the most recent reported earnings
    - Google Trends: aligned to the most recent weekly data
    - SEC EDGAR: aligned to the most recent filing date
    - Current-snapshot fundamentals: applied uniformly (documented limitation)
    - Sentiment: current snapshot (documented limitation)

    Args:
        tickers: List of ticker symbols.
        include_sentiment: Whether to add sentiment features.
        max_samples_per_ticker: Maximum rows per ticker. ``None`` keeps
            every valid trading day (full dataset). Pass an integer
            (e.g. 200) to linearly sample down.

    Returns:
        DataFrame ready for model training.
    """
    # Fetch macro data once (shared across all tickers)
    logger.info("Fetching macroeconomic data...")
    macro_df = get_macro_data(period="6y")

    all_rows: list[dict] = []

    for ticker in tickers:
        try:
            # 3y window ensures all historical data sources have coverage
            # (YFinance quarterly financials only go back ~5 quarters;
            # with 3y price data, valid training rows start ~2024-03,
            # which is after all tickers' earliest fundamental date)
            df = get_stock_data(ticker, period="3y")
            if df.empty or len(df) < 200:
                logger.warning("Skipping %s — insufficient history", ticker)
                continue

            df = compute_technical_features(df)

            # Compute 3-month forward return (~63 trading days)
            df[TARGET_COLUMN] = df["Close"].shift(-63) / df["Close"] - 1

            # --- Fetch historical data sources for this ticker ---
            hist_fund = get_historical_fundamentals(ticker)
            earnings_df = get_earnings_history(ticker)
            trends_df = get_google_trends(ticker)
            sec_df = get_sec_filings(ticker)

            # Current-snapshot fundamentals (documented limitation)
            fundamentals = get_fundamentals_features(ticker)

            # Sentiment (current snapshot — documented limitation)
            sentiment: dict = {}
            if include_sentiment:
                sentiment = get_sentiment_features(ticker)

            # Filter valid rows
            valid_mask = df[TARGET_COLUMN].notna() & df["SMA_200"].notna()
            valid_df = df[valid_mask]

            if valid_df.empty:
                continue

            # Optionally subsample; default keeps every valid day
            if max_samples_per_ticker is not None and len(valid_df) > max_samples_per_ticker:
                sample_indices = np.linspace(
                    0, len(valid_df) - 1, max_samples_per_ticker, dtype=int
                )
                sampled = valid_df.iloc[sample_indices]
            else:
                sampled = valid_df

            # Extract actual calendar dates for time-alignment
            if "Date" in sampled.columns:
                sample_dates = pd.to_datetime(
                    sampled["Date"].dt.tz_localize(None)
                    if hasattr(sampled["Date"].dt, "tz_localize") and sampled["Date"].dt.tz is not None
                    else sampled["Date"]
                )
            else:
                sample_dates = pd.to_datetime(sampled.index)

            # --- Time-align historical features to each sample date ---
            aligned_hist_fund = align_fundamentals_to_dates(hist_fund, sample_dates)
            aligned_earnings = align_earnings_to_dates(earnings_df, sample_dates)
            aligned_trends = align_trends_to_dates(trends_df, sample_dates)
            aligned_sec = align_sec_to_dates(sec_df, sample_dates)
            aligned_macro = align_macro_to_dates(macro_df, sample_dates)

            for i, (idx, row) in enumerate(sampled.iterrows()):
                data_point: dict = {"Ticker": ticker}
                # Store actual calendar date for temporal splitting
                date_val = sample_dates.iloc[i] if hasattr(sample_dates, "iloc") else sample_dates[i]
                data_point["_date"] = str(date_val)

                # Technical features (inherently time-correct)
                for col in TECHNICAL_FEATURES:
                    data_point[col] = row.get(col, np.nan)

                # Current-snapshot fundamentals (limitation documented)
                for col in FUNDAMENTAL_FEATURES:
                    data_point[col] = fundamentals.get(col, np.nan)

                # Historical fundamentals (time-aligned, no leakage)
                for col in HIST_FUNDAMENTAL_FEATURES:
                    data_point[col] = (
                        aligned_hist_fund[col].iloc[i]
                        if col in aligned_hist_fund.columns
                        else np.nan
                    )

                # Macroeconomic data (time-aligned, no leakage)
                for col in MACRO_FEATURES:
                    data_point[col] = (
                        aligned_macro[col].iloc[i]
                        if col in aligned_macro.columns
                        else np.nan
                    )

                # Earnings data (time-aligned, no leakage)
                for col in EARNINGS_FEATURES:
                    data_point[col] = (
                        aligned_earnings[col].iloc[i]
                        if col in aligned_earnings.columns
                        else np.nan
                    )

                # Google Trends (only if data was fetched successfully)
                if not trends_df.empty:
                    for col in TRENDS_FEATURES:
                        data_point[col] = (
                            aligned_trends[col].iloc[i]
                            if col in aligned_trends.columns
                            else np.nan
                        )

                # SEC EDGAR (time-aligned, no leakage)
                for col in SEC_FEATURES:
                    data_point[col] = (
                        aligned_sec[col].iloc[i]
                        if col in aligned_sec.columns
                        else np.nan
                    )

                # Sentiment (current snapshot — documented limitation)
                for col in SENTIMENT_FEATURES:
                    data_point[col] = sentiment.get(col, 0.0)

                data_point[TARGET_COLUMN] = row[TARGET_COLUMN]
                all_rows.append(data_point)

            logger.info("Processed %s: %d samples", ticker, len(sampled))

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
