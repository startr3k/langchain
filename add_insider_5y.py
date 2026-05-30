"""Add insider transaction features to the 5-year tech+macro dataset.

For each ticker, fetches Form 4 filings from SEC EDGAR and computes
time-aligned features per date row. Checkpoints every 50 tickers.
"""

import gc
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from stock_predictor.data.insider_transactions import (
    get_insider_transactions,
    _compute_insider_features_at_date,
    INSIDER_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_CSV = "training_data_techonly_5y.csv"
OUTPUT_CSV = "training_data_5y_with_insider.csv"
CHECKPOINT_CSV = "training_data_5y_insider_checkpoint.csv"
PROGRESS_FILE = "insider_5y_progress.txt"
CHECKPOINT_INTERVAL = 50


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_progress(done):
    with open(PROGRESS_FILE, "w") as f:
        for t in sorted(done):
            f.write(t + "\n")


def main():
    logger.info("Loading dataset...")
    df = pd.read_csv(INPUT_CSV)
    df["_date"] = pd.to_datetime(df["_date"])
    logger.info("Loaded: %d rows, %d columns", len(df), len(df.columns))

    # Initialize insider columns if not present
    for col in INSIDER_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    tickers = sorted(df["Ticker"].dropna().unique())
    done = load_progress()
    logger.info("Total tickers: %d, already done: %d", len(tickers), len(done))

    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_CSV) and len(done) > 0:
        logger.info("Loading checkpoint...")
        df = pd.read_csv(CHECKPOINT_CSV)
        df["_date"] = pd.to_datetime(df["_date"])
        logger.info("Loaded checkpoint: %d rows", len(df))

    processed_since_checkpoint = 0

    for ticker in tickers:
        if ticker in done:
            continue

        try:
            # Fetch all Form 4 filings for this ticker
            transactions = get_insider_transactions(ticker)

            # Get the rows for this ticker
            mask = df["Ticker"] == ticker
            ticker_dates = df.loc[mask, "_date"]

            if transactions.empty:
                # No filings found — leave as NaN
                logger.debug("No insider filings for %s", ticker)
            else:
                # Compute features for each date
                for idx, date_val in ticker_dates.items():
                    features = _compute_insider_features_at_date(
                        transactions, pd.Timestamp(date_val)
                    )
                    for col, val in features.items():
                        df.at[idx, col] = val

            # Rate limit: SEC asks for max 10 req/sec
            time.sleep(0.15)

        except Exception:
            logger.exception("Error processing %s", ticker)

        done.add(ticker)
        processed_since_checkpoint += 1

        if processed_since_checkpoint % 10 == 0:
            logger.info(
                "Progress: %d / %d tickers (%.1f%%)",
                len(done), len(tickers), len(done) / len(tickers) * 100,
            )

        if processed_since_checkpoint >= CHECKPOINT_INTERVAL:
            logger.info("Checkpointing at %d tickers...", len(done))
            save_progress(done)
            df.to_csv(CHECKPOINT_CSV, index=False)
            processed_since_checkpoint = 0
            gc.collect()

    # Final save
    logger.info("Saving final dataset...")
    save_progress(done)
    df.to_csv(OUTPUT_CSV, index=False)

    # Stats
    for col in INSIDER_FEATURES:
        nan_pct = df[col].isna().mean() * 100
        logger.info("  %s: %.1f%% NaN", col, nan_pct)

    logger.info(
        "Done. %d rows, %d columns saved to %s",
        len(df), len(df.columns), OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
