"""Process remaining NASDAQ tickers and append to checkpoint CSV."""

import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_predictor.data.feature_engineering import (
    ALL_FEATURE_NAMES,
    EARNINGS_FEATURES,
    FUNDAMENTAL_FEATURES,
    HIST_FUNDAMENTAL_FEATURES,
    MACRO_FEATURES,
    SEC_FEATURES,
    SENTIMENT_FEATURES,
    TARGET_COLUMN,
    TECHNICAL_FEATURES,
)
from stock_predictor.data.historical_fundamentals import (
    align_fundamentals_to_dates,
    get_historical_fundamentals,
)
from stock_predictor.data.earnings_data import align_earnings_to_dates, get_earnings_history
from stock_predictor.data.macro_data import align_macro_to_dates, get_macro_data
from stock_predictor.data.sec_edgar import align_sec_to_dates, get_sec_filings
from stock_predictor.data.yfinance_client import (
    compute_technical_features,
    get_fundamentals_features,
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


def process_ticker(ticker, macro_df):
    """Process a single ticker and return list of row dicts."""
    rows = []
    try:
        df = get_stock_data(ticker, period="3y")
        if df.empty or len(df) < 200:
            return rows

        df = compute_technical_features(df)
        df[TARGET_COLUMN] = df["Close"].shift(-63) / df["Close"] - 1

        hist_fund = get_historical_fundamentals(ticker)
        earnings_df = get_earnings_history(ticker)
        sec_df = get_sec_filings(ticker)
        fundamentals = get_fundamentals_features(ticker)

        valid_mask = df[TARGET_COLUMN].notna() & df["SMA_200"].notna()
        valid_df = df[valid_mask]
        if valid_df.empty:
            return rows

        if "Date" in valid_df.columns:
            sample_dates = pd.to_datetime(
                valid_df["Date"].dt.tz_localize(None)
                if hasattr(valid_df["Date"].dt, "tz_localize")
                and valid_df["Date"].dt.tz is not None
                else valid_df["Date"]
            )
        else:
            sample_dates = pd.to_datetime(valid_df.index)

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
            for col in FUNDAMENTAL_FEATURES:
                data_point[col] = fundamentals.get(col, np.nan)
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
            for col in SENTIMENT_FEATURES:
                data_point[col] = 0.0

            data_point[TARGET_COLUMN] = row[TARGET_COLUMN]
            rows.append(data_point)

    except Exception:
        logger.exception("Error processing %s", ticker)

    return rows


def main():
    # Get already-done tickers from checkpoint (memory-efficient)
    done_tickers = set()
    if os.path.exists(CHECKPOINT_FILE):
        for chunk in pd.read_csv(CHECKPOINT_FILE, usecols=["Ticker"], chunksize=100000):
            done_tickers.update(chunk["Ticker"].unique())
    logger.info("Already done: %d tickers", len(done_tickers))

    with open(os.path.join(OUTPUT_DIR, "nasdaq_all_tickers.txt")) as f:
        all_tickers = [l.strip() for l in f if l.strip()]

    remaining = [t for t in all_tickers if t not in done_tickers]
    logger.info("Remaining: %d tickers to process", len(remaining))

    if not remaining:
        logger.info("All tickers already processed!")
        # Just copy checkpoint to final
        os.rename(CHECKPOINT_FILE, FINAL_FILE)
        return

    # Fetch macro data once
    logger.info("Fetching macroeconomic data...")
    macro_df = get_macro_data(period="6y")

    # Process remaining and append to checkpoint
    new_rows = 0
    succeeded = 0
    start = time.time()

    # Batch: collect rows, write every 25 tickers
    batch_rows = []
    batch_count = 0

    for i, ticker in enumerate(remaining):
        ticker_rows = process_ticker(ticker, macro_df)
        if ticker_rows:
            batch_rows.extend(ticker_rows)
            succeeded += 1
            new_rows += len(ticker_rows)
            logger.info(
                "[%d/%d] %s: %d rows (new total: %d)",
                i + 1, len(remaining), ticker, len(ticker_rows), new_rows,
            )

        batch_count += 1
        if batch_count >= 25 and batch_rows:
            # Append to checkpoint CSV
            batch_df = pd.DataFrame(batch_rows)
            batch_df.to_csv(
                CHECKPOINT_FILE, mode="a", header=False, index=False
            )
            batch_rows = []
            batch_count = 0

    # Write any remaining batch
    if batch_rows:
        batch_df = pd.DataFrame(batch_rows)
        batch_df.to_csv(CHECKPOINT_FILE, mode="a", header=False, index=False)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("DONE in %.1f minutes", elapsed / 60)
    logger.info("New rows added: %d", new_rows)
    logger.info("New tickers with data: %d", succeeded)

    # Copy checkpoint to final
    import shutil
    shutil.copy2(CHECKPOINT_FILE, FINAL_FILE)
    logger.info("Saved final dataset to %s", FINAL_FILE)


if __name__ == "__main__":
    main()
