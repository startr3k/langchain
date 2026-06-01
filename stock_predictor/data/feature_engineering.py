"""Feature engineering pipeline that combines all data sources.

All historical features are time-aligned — each training row only sees
data that was available at that point in time, eliminating look-ahead bias.

Data sources:
1. Technical indicators (from price/volume — inherently time-correct)
2. Historical quarterly fundamentals (YFinance quarterly filings)
3. Macroeconomic data (VIX, rates, S&P500 — properly time-indexed)
4. Earnings surprise history (YFinance earnings_dates)
5. Google Trends (historical search interest)
6. SEC EDGAR XBRL (historical regulatory filings)
7. Short interest (yfinance — cross-sectional snapshot)
8. Options flow (yfinance option chains — cross-sectional snapshot)
9. Insider transactions (SEC Form 4 — time-aligned filing dates)
10. Reddit historical sentiment (Arctic Shift — time-aligned posts)

Excluded from model (data leakage risk):
- Current-snapshot fundamentals: today's values applied to all historical
  rows causes look-ahead bias.  Kept in agent/UI output only.
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
from stock_predictor.data.short_interest import (
    SHORT_INTEREST_FEATURES,
    align_short_interest_to_dates,
)
from stock_predictor.data.options_flow import (
    OPTIONS_FLOW_FEATURES,
    align_options_flow_to_dates,
)
from stock_predictor.data.insider_transactions import (
    INSIDER_FEATURES,
    align_insider_to_dates,
)
from stock_predictor.data.reddit_sentiment import (
    REDDIT_SENTIMENT_FEATURES,
    align_reddit_sentiment_to_dates,
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
    # Normalized price-relative ratios (not raw dollar values)
    "Price_to_SMA_20", "Price_to_SMA_50", "Price_to_SMA_200",
    # SMA 200 cross signal (duration only — daily cross event dropped)
    "Days_Since_SMA200_Cross",
    "Volatility_20d", "Volatility_60d",
    # Breakout-engineered features
    "Volatility_Contraction", "Momentum_Accel",
    "Volume_Price_Confirm", "Dist_52w_High", "Dist_52w_Low",
    "BB_Squeeze_Duration",
    "Volume_Ratio", "Volume_Surge_3d", "Volume_Spike_Magnitude",
    "RSI_14", "MACD", "MACD_Hist",
    "BB_Width", "BB_Position",
]

# Interaction / derived features computed during training preprocessing.
# These combine existing features to capture multi-factor signals.
DERIVED_FEATURES: list[str] = [
    # Fundamental_Surprise was removed — its input (earnings_surprise_pct)
    # is not present in the training CSV.
]

# Current-snapshot fundamentals — EXCLUDED from model training/prediction
# due to data leakage (today's values applied to all historical rows).
# Kept here for reference and used in agent/UI output only.
FUNDAMENTAL_FEATURES = [
    # Valuation ratios
    "marketCap", "trailingPE", "forwardPE", "priceToBook",
    "pegRatio", "priceToSalesTrailing12Months",
    "enterpriseToEbitda", "enterpriseToRevenue",
    # Per-share data
    "trailingEps", "forwardEps", "bookValue", "revenuePerShare",
    # Profitability
    "profitMargins", "grossMargins",
    "operatingMargins", "ebitdaMargins",
    "returnOnEquity", "returnOnAssets",
    # Growth
    "revenueGrowth",
    # Risk / leverage
    "beta", "debtToEquity", "currentRatio",
    # Ownership / sentiment
    "shortRatio", "shortPercentOfFloat",
    "heldPercentInsiders", "heldPercentInstitutions",
    "numberOfAnalystOpinions",
    # Earnings calendar
    "daysToEarnings",
]

# Sentiment features — EXCLUDED from model training/prediction
# due to data leakage (today's headlines applied to all historical rows).
# Kept here for reference and used in agent/UI output only.
SENTIMENT_FEATURES = [
    # Aggregated sentiment (derived from available sources)
    "sentiment_mean_polarity", "sentiment_std_polarity",
    "sentiment_max_polarity", "sentiment_min_polarity",
    "sentiment_mean_subjectivity", "sentiment_total_mentions",
    # Finviz news headlines (reliable from cloud environments)
    "finviz_mention_count", "finviz_mean_polarity",
]

# Features removed based on multicollinearity / grouped permutation
# importance analysis.  These either hurt generalisation (negative
# permutation importance) or are perfectly redundant (r = 1.0).
DROPPED_FEATURES = {
    "treasury_10y",                   # negative perm importance (overfits to rate regime)
    "hist_current_ratio",             # negative perm importance
    "hist_revenue_growth_qoq",        # negative perm importance
    "gold_return_20d",                # negative perm importance
    "insider_total_transactions_90d", # r=1.0 with insider_net_buys_90d
}

# All features used by the model (training + prediction).
# Only time-aligned features are included — no current-snapshot
# fundamentals or sentiment (data leakage).
# Google Trends features are excluded from the default list because
# Google aggressively rate-limits cloud/datacenter IPs; they are
# added dynamically when data is actually available.
ALL_FEATURE_NAMES = [
    f for f in (
        TECHNICAL_FEATURES
        + HIST_FUNDAMENTAL_FEATURES
        + MACRO_FEATURES
        + EARNINGS_FEATURES
        + SEC_FEATURES
        + SHORT_INTEREST_FEATURES
        + OPTIONS_FLOW_FEATURES
        + INSIDER_FEATURES
        + REDDIT_SENTIMENT_FEATURES
        + DERIVED_FEATURES
    )
    if f not in DROPPED_FEATURES
]

TARGET_COLUMN = "Forward_Max_Return_3M"


def build_training_row(
    ticker: str,
    include_sentiment: bool = True,
) -> dict | None:
    """Build a single feature row (latest data point) for a ticker.

    Used for live prediction — uses only time-aligned features that
    match the model's training feature set (no current-snapshot
    fundamentals or sentiment).

    Args:
        ticker: Stock ticker symbol.
        include_sentiment: Ignored (kept for API compatibility).
            Sentiment features are excluded from the model.

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

        # Short interest (current snapshot — cross-sectional)
        si = align_short_interest_to_dates(ticker, pd.DatetimeIndex([pd.Timestamp.now().normalize()]))
        for col in SHORT_INTEREST_FEATURES:
            row[col] = si[col].iloc[0] if col in si.columns else np.nan

        # Options flow (current snapshot — cross-sectional)
        of = align_options_flow_to_dates(ticker, pd.DatetimeIndex([pd.Timestamp.now().normalize()]))
        for col in OPTIONS_FLOW_FEATURES:
            row[col] = of[col].iloc[0] if col in of.columns else np.nan

        # Insider transactions (time-aligned from SEC Form 4)
        today = pd.Timestamp.now().normalize()
        ins = align_insider_to_dates(ticker, pd.DatetimeIndex([today]))
        for col in INSIDER_FEATURES:
            row[col] = ins[col].iloc[0] if col in ins.columns else np.nan

        # Reddit historical sentiment (time-aligned from Arctic Shift)
        reddit = align_reddit_sentiment_to_dates(ticker, pd.DatetimeIndex([today]))
        for col in REDDIT_SENTIMENT_FEATURES:
            row[col] = reddit[col].iloc[0] if col in reddit.columns else np.nan

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

    Current-snapshot fundamentals and sentiment are EXCLUDED to
    prevent data leakage.

    Args:
        tickers: List of ticker symbols.
        include_sentiment: Ignored (kept for API compatibility).
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
            df = get_stock_data(ticker, period="10y")
            if df.empty or len(df) < 200:
                logger.warning("Skipping %s — insufficient history", ticker)
                continue

            df = compute_technical_features(df)

            # Compute max forward return within 3-month window (~63 trading days).
            # For each day, find the highest Close price in the next 63 days
            # and compute the return from today's Close to that peak.
            rolling_max = (
                df["Close"]
                .iloc[::-1]
                .rolling(window=63, min_periods=1)
                .max()
                .iloc[::-1]
            )
            df[TARGET_COLUMN] = rolling_max / df["Close"] - 1
            # NaN out the last 63 rows (incomplete forward window)
            df.loc[df.index[-63:], TARGET_COLUMN] = np.nan

            # --- Fetch historical data sources for this ticker ---
            hist_fund = get_historical_fundamentals(ticker)
            earnings_df = get_earnings_history(ticker)
            trends_df = get_google_trends(ticker)
            sec_df = get_sec_filings(ticker)
            # New data sources — may return empty if API is unavailable
            # Short interest & options flow are cross-sectional snapshots
            # Insider transactions & Reddit sentiment are time-aligned

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
            aligned_si = align_short_interest_to_dates(ticker, sample_dates)
            aligned_of = align_options_flow_to_dates(ticker, sample_dates)
            aligned_ins = align_insider_to_dates(ticker, sample_dates)
            aligned_reddit = align_reddit_sentiment_to_dates(ticker, sample_dates)

            for i, (idx, row) in enumerate(sampled.iterrows()):
                data_point: dict = {"Ticker": ticker}
                # Store actual calendar date for temporal splitting
                date_val = sample_dates.iloc[i] if hasattr(sample_dates, "iloc") else sample_dates[i]
                data_point["_date"] = str(date_val)

                # Technical features (inherently time-correct)
                for col in TECHNICAL_FEATURES:
                    data_point[col] = row.get(col, np.nan)

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

                # Short interest (cross-sectional snapshot)
                for col in SHORT_INTEREST_FEATURES:
                    data_point[col] = (
                        aligned_si[col].iloc[i]
                        if col in aligned_si.columns
                        else np.nan
                    )

                # Options flow (cross-sectional snapshot)
                for col in OPTIONS_FLOW_FEATURES:
                    data_point[col] = (
                        aligned_of[col].iloc[i]
                        if col in aligned_of.columns
                        else np.nan
                    )

                # Insider transactions (time-aligned, no leakage)
                for col in INSIDER_FEATURES:
                    data_point[col] = (
                        aligned_ins[col].iloc[i]
                        if col in aligned_ins.columns
                        else np.nan
                    )

                # Reddit historical sentiment (time-aligned, no leakage)
                for col in REDDIT_SENTIMENT_FEATURES:
                    data_point[col] = (
                        aligned_reddit[col].iloc[i]
                        if col in aligned_reddit.columns
                        else np.nan
                    )

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


# ---------------------------------------------------------------------------
# Incremental dataset builder
# ---------------------------------------------------------------------------

# Number of calendar days of lookback needed for technical indicator warmup.
# SMA_200 requires ~200 trading days ≈ 290 calendar days.  We add margin.
_LOOKBACK_CALENDAR_DAYS = 350


def build_incremental_dataset(
    tickers: list[str],
    existing_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch only new rows that are not already in *existing_df*.

    Instead of re-downloading the full history for every ticker, this
    function:
    1. Finds the latest ``_date`` **per ticker** in *existing_df*.
    2. Fetches price data starting from ``max_date - lookback`` (to warm
       up rolling technical indicators like SMA_200).
    3. Generates training rows only for dates **after** the ticker's
       existing max date.

    Tickers that are entirely absent from *existing_df* are fetched in
    full (``period="10y"``).

    Args:
        tickers: List of ticker symbols.
        existing_df: The current training CSV loaded as a DataFrame.  Must
            contain at least ``Ticker`` and ``_date`` columns.

    Returns:
        DataFrame of **new rows only** (same schema as
        :func:`build_training_dataset`).  The caller should concatenate
        this with *existing_df* and de-duplicate.
    """
    existing_df = existing_df.copy()
    existing_df["_date"] = pd.to_datetime(existing_df["_date"])

    # Per-ticker max date for cutoff
    ticker_max_dates: dict[str, pd.Timestamp] = (
        existing_df.groupby("Ticker")["_date"].max().to_dict()
    )
    global_max = existing_df["_date"].max()

    logger.info(
        "Incremental build: %d tickers in existing data, global max date %s",
        len(ticker_max_dates), global_max.date(),
    )

    # Fetch macro data once
    macro_df = get_macro_data(period="6y")

    all_rows: list[dict] = []
    skipped = 0

    today = pd.Timestamp.now().normalize()

    for ticker in tickers:
        try:
            cutoff = ticker_max_dates.get(ticker)

            if cutoff is not None:
                # Fast path: skip tickers whose data is already up to date.
                # The 63-day target window makes the latest *usable* row
                # ~63 trading days before the last price date, so any
                # ticker whose max date is within the last 2 calendar days
                # cannot possibly produce new training rows.
                days_stale = (today - cutoff).days
                if days_stale <= 2:
                    skipped += 1
                    continue

            if cutoff is None:
                # New ticker — fetch full 10y history to match existing dataset
                df = get_stock_data(ticker, period="10y")
                row_cutoff = None
            else:
                # Existing ticker — fetch from cutoff minus lookback
                start_date = (cutoff - pd.Timedelta(days=_LOOKBACK_CALENDAR_DAYS)).strftime("%Y-%m-%d")
                df = get_stock_data(ticker, start=start_date)
                row_cutoff = cutoff

            if df.empty or len(df) < 200:
                if cutoff is None:
                    logger.warning("Skipping new ticker %s — insufficient history", ticker)
                else:
                    skipped += 1
                continue

            df = compute_technical_features(df)

            # Forward max return target
            rolling_max = (
                df["Close"]
                .iloc[::-1]
                .rolling(window=63, min_periods=1)
                .max()
                .iloc[::-1]
            )
            df[TARGET_COLUMN] = rolling_max / df["Close"] - 1
            df.loc[df.index[-63:], TARGET_COLUMN] = np.nan

            # Fetch auxiliary data sources
            hist_fund = get_historical_fundamentals(ticker)
            earnings_df = get_earnings_history(ticker)
            trends_df = get_google_trends(ticker)
            sec_df = get_sec_filings(ticker)

            valid_mask = df[TARGET_COLUMN].notna() & df["SMA_200"].notna()
            valid_df = df[valid_mask]
            if valid_df.empty:
                continue

            # Extract calendar dates
            if "Date" in valid_df.columns:
                sample_dates = pd.to_datetime(
                    valid_df["Date"].dt.tz_localize(None)
                    if hasattr(valid_df["Date"].dt, "tz_localize") and valid_df["Date"].dt.tz is not None
                    else valid_df["Date"]
                )
            else:
                sample_dates = pd.to_datetime(valid_df.index)

            # Keep only rows after the existing cutoff
            if row_cutoff is not None:
                new_mask = sample_dates > cutoff
                valid_df = valid_df.loc[new_mask.values]
                sample_dates = sample_dates[new_mask.values]

            if valid_df.empty:
                skipped += 1
                continue

            # Time-align auxiliary features
            aligned_hist_fund = align_fundamentals_to_dates(hist_fund, sample_dates)
            aligned_earnings = align_earnings_to_dates(earnings_df, sample_dates)
            aligned_trends = align_trends_to_dates(trends_df, sample_dates)
            aligned_sec = align_sec_to_dates(sec_df, sample_dates)
            aligned_macro = align_macro_to_dates(macro_df, sample_dates)
            aligned_si = align_short_interest_to_dates(ticker, sample_dates)
            aligned_of = align_options_flow_to_dates(ticker, sample_dates)
            aligned_ins = align_insider_to_dates(ticker, sample_dates)
            aligned_reddit = align_reddit_sentiment_to_dates(ticker, sample_dates)

            for i, (idx, row) in enumerate(valid_df.iterrows()):
                data_point: dict = {"Ticker": ticker}
                date_val = sample_dates.iloc[i] if hasattr(sample_dates, "iloc") else sample_dates[i]
                data_point["_date"] = str(date_val)

                for col in TECHNICAL_FEATURES:
                    data_point[col] = row.get(col, np.nan)

                for col in HIST_FUNDAMENTAL_FEATURES:
                    data_point[col] = (
                        aligned_hist_fund[col].iloc[i]
                        if col in aligned_hist_fund.columns
                        else np.nan
                    )

                for col in MACRO_FEATURES:
                    data_point[col] = (
                        aligned_macro[col].iloc[i]
                        if col in aligned_macro.columns
                        else np.nan
                    )

                for col in EARNINGS_FEATURES:
                    data_point[col] = (
                        aligned_earnings[col].iloc[i]
                        if col in aligned_earnings.columns
                        else np.nan
                    )

                if not trends_df.empty:
                    for col in TRENDS_FEATURES:
                        data_point[col] = (
                            aligned_trends[col].iloc[i]
                            if col in aligned_trends.columns
                            else np.nan
                        )

                for col in SEC_FEATURES:
                    data_point[col] = (
                        aligned_sec[col].iloc[i]
                        if col in aligned_sec.columns
                        else np.nan
                    )

                for col in SHORT_INTEREST_FEATURES:
                    data_point[col] = (
                        aligned_si[col].iloc[i]
                        if col in aligned_si.columns
                        else np.nan
                    )

                for col in OPTIONS_FLOW_FEATURES:
                    data_point[col] = (
                        aligned_of[col].iloc[i]
                        if col in aligned_of.columns
                        else np.nan
                    )

                for col in INSIDER_FEATURES:
                    data_point[col] = (
                        aligned_ins[col].iloc[i]
                        if col in aligned_ins.columns
                        else np.nan
                    )

                for col in REDDIT_SENTIMENT_FEATURES:
                    data_point[col] = (
                        aligned_reddit[col].iloc[i]
                        if col in aligned_reddit.columns
                        else np.nan
                    )

                data_point[TARGET_COLUMN] = row[TARGET_COLUMN]
                all_rows.append(data_point)

            logger.info(
                "Incremental %s: %d new samples (cutoff=%s)",
                ticker, len(valid_df),
                cutoff.date() if cutoff is not None else "none",
            )

        except Exception:
            logger.exception("Error processing %s incrementally", ticker)

    logger.info(
        "Incremental build done: %d new rows from %d tickers (%d skipped — already up to date)",
        len(all_rows), len(tickers), skipped,
    )

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)
