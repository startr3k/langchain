"""Generate full NASDAQ training dataset with all valid trading days."""

import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_predictor.data.feature_engineering import (
    ALL_FEATURE_NAMES,
    DERIVED_FEATURES,
    EARNINGS_FEATURES,
    HIST_FUNDAMENTAL_FEATURES,
    MACRO_FEATURES,
    SEC_FEATURES,
    TARGET_COLUMN,
    TECHNICAL_FEATURES,
)
from stock_predictor.models.automl_model import _compute_derived_features
from stock_predictor.data.historical_fundamentals import (
    align_fundamentals_to_dates,
    get_historical_fundamentals,
)
from stock_predictor.data.earnings_data import align_earnings_to_dates, get_earnings_history
from stock_predictor.data.macro_data import align_macro_to_dates, get_macro_data
from stock_predictor.data.sec_edgar import align_sec_to_dates, get_sec_filings
from stock_predictor.data.yfinance_client import (
    compute_technical_features,
    get_stock_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "/home/ubuntu/repos/langchain"
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "full_training_data_checkpoint.csv")
FINAL_FILE = os.path.join(OUTPUT_DIR, "full_training_data.csv")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "generation_progress.txt")
SAVE_EVERY = 25  # Save checkpoint every N tickers


def load_tickers():
    with open(os.path.join(OUTPUT_DIR, "nasdaq_all_tickers.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def load_checkpoint():
    """Load previously processed data and completed tickers."""
    if os.path.exists(CHECKPOINT_FILE):
        df = pd.read_csv(CHECKPOINT_FILE)
        done = set(df["Ticker"].unique())
        logger.info("Loaded checkpoint: %d rows, %d tickers done", len(df), len(done))
        return df, done
    return pd.DataFrame(), set()


def save_checkpoint(df, done_tickers, total_tickers, start_time):
    df.to_csv(CHECKPOINT_FILE, index=False)
    elapsed = time.time() - start_time
    rate = len(done_tickers) / max(elapsed, 1) * 3600
    remaining = total_tickers - len(done_tickers)
    eta_hours = remaining / max(rate, 1)
    with open(PROGRESS_FILE, "w") as f:
        f.write(
            f"Processed: {len(done_tickers)}/{total_tickers} tickers\n"
            f"Rows: {len(df)}\n"
            f"Elapsed: {elapsed/60:.1f} min\n"
            f"Rate: {rate:.0f} tickers/hr\n"
            f"ETA: {eta_hours:.1f} hr\n"
        )


def process_ticker(ticker, macro_df):
    """Process a single ticker and return list of row dicts."""
    rows = []
    try:
        df = get_stock_data(ticker, period="3y")
        if df.empty or len(df) < 200:
            return rows

        df = compute_technical_features(df)

        # Compute max forward return within 3-month window (~63 trading days)
        rolling_max = (
            df["Close"]
            .iloc[::-1]
            .rolling(window=63, min_periods=1)
            .max()
            .iloc[::-1]
        )
        df[TARGET_COLUMN] = rolling_max / df["Close"] - 1
        df.loc[df.index[-63:], TARGET_COLUMN] = np.nan

        # Fetch historical data
        hist_fund = get_historical_fundamentals(ticker)
        earnings_df = get_earnings_history(ticker)
        sec_df = get_sec_filings(ticker)
        # Valid rows
        valid_mask = df[TARGET_COLUMN].notna() & df["SMA_200"].notna()
        valid_df = df[valid_mask]
        if valid_df.empty:
            return rows

        # Extract dates
        if "Date" in valid_df.columns:
            sample_dates = pd.to_datetime(
                valid_df["Date"].dt.tz_localize(None)
                if hasattr(valid_df["Date"].dt, "tz_localize")
                and valid_df["Date"].dt.tz is not None
                else valid_df["Date"]
            )
        else:
            sample_dates = pd.to_datetime(valid_df.index)

        # Time-align
        aligned_hist_fund = align_fundamentals_to_dates(hist_fund, sample_dates)
        aligned_earnings = align_earnings_to_dates(earnings_df, sample_dates)
        aligned_sec = align_sec_to_dates(sec_df, sample_dates)
        aligned_macro = align_macro_to_dates(macro_df, sample_dates)

        for i, (idx, row) in enumerate(valid_df.iterrows()):
            data_point = {"Ticker": ticker}
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
            for col in SEC_FEATURES:
                data_point[col] = (
                    aligned_sec[col].iloc[i]
                    if col in aligned_sec.columns
                    else np.nan
                )
            data_point[TARGET_COLUMN] = row[TARGET_COLUMN]
            rows.append(data_point)

    except Exception:
        logger.exception("Error processing %s", ticker)

    return rows


def main():
    tickers = load_tickers()
    logger.info("Total NASDAQ tickers: %d", len(tickers))

    existing_df, done_tickers = load_checkpoint()
    all_rows = existing_df.to_dict("records") if not existing_df.empty else []

    remaining = [t for t in tickers if t not in done_tickers]
    logger.info("Remaining: %d tickers", len(remaining))

    # Fetch macro data once
    logger.info("Fetching macroeconomic data...")
    macro_df = get_macro_data(period="6y")

    start_time = time.time()
    batch_count = 0
    skipped = 0
    succeeded = 0

    for i, ticker in enumerate(remaining):
        ticker_rows = process_ticker(ticker, macro_df)
        if ticker_rows:
            all_rows.extend(ticker_rows)
            succeeded += 1
            logger.info(
                "[%d/%d] %s: %d rows (total: %d rows, %d tickers)",
                len(done_tickers) + i + 1,
                len(tickers),
                ticker,
                len(ticker_rows),
                len(all_rows),
                succeeded + len(done_tickers),
            )
        else:
            skipped += 1

        done_tickers.add(ticker)
        batch_count += 1

        if batch_count >= SAVE_EVERY:
            df = pd.DataFrame(all_rows)
            save_checkpoint(df, done_tickers, len(tickers), start_time)
            batch_count = 0

    # Final save
    df = pd.DataFrame(all_rows)

    # Compute derived interaction features
    logger.info("Computing derived features...")
    df = _compute_derived_features(df)

    df.to_csv(FINAL_FILE, index=False)
    save_checkpoint(df, done_tickers, len(tickers), start_time)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("DONE in %.1f minutes", elapsed / 60)
    logger.info("Total rows: %d", len(df))
    logger.info("Tickers with data: %d", df["Ticker"].nunique())
    logger.info("Tickers skipped: %d", skipped)
    logger.info("Saved to %s", FINAL_FILE)

    # NaN stats
    feature_cols = [c for c in df.columns if c not in ["Ticker", "_date", TARGET_COLUMN]]
    total = len(df) * len(feature_cols)
    nulls = df[feature_cols].isna().sum().sum()
    logger.info("NaN: %d/%d (%.1f%%)", nulls, total, nulls / total * 100)


if __name__ == "__main__":
    main()
