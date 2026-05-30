"""Add SEC filing sentiment and GDELT news sentiment to the 5-year dataset.

For each ticker:
1. Fetch 10-K/10-Q filing text from EDGAR, compute LM sentiment
2. Fetch daily GDELT news tone, compute 7-day rolling features
3. Time-align to each training row date

Checkpoints every 25 tickers.
"""

import gc
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from stock_predictor.data.sec_sentiment import (
    get_filing_sentiments,
    align_sec_sentiment_to_dates,
    SEC_SENTIMENT_FEATURES,
)
from stock_predictor.data.gdelt_sentiment import (
    align_gdelt_to_dates,
    GDELT_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_CSV = "training_data_techonly_5y.csv"
OUTPUT_CSV = "training_data_5y_with_sentiment.csv"
CHECKPOINT_CSV = "training_data_5y_sentiment_checkpoint.csv"
PROGRESS_FILE = "sec_gdelt_progress.txt"
CHECKPOINT_INTERVAL = 25

ALL_NEW_FEATURES = SEC_SENTIMENT_FEATURES + GDELT_FEATURES


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
    # Check if checkpoint exists and use it
    done = load_progress()

    if os.path.exists(CHECKPOINT_CSV) and len(done) > 0:
        df = pd.read_csv(CHECKPOINT_CSV)
        logger.info("Loaded checkpoint: %d rows", len(df))
    else:
        df = pd.read_csv(INPUT_CSV)
        logger.info("Loaded fresh: %d rows", len(df))

    df["_date"] = pd.to_datetime(df["_date"])

    # Initialize new columns if not present
    for col in ALL_NEW_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    tickers = sorted(df["Ticker"].dropna().unique())
    logger.info(
        "Total tickers: %d, already done: %d, remaining: %d",
        len(tickers), len(done), len(tickers) - len(done),
    )

    processed_since_checkpoint = 0

    for ticker in tickers:
        if ticker in done:
            continue

        mask = df["Ticker"] == ticker
        ticker_dates = df.loc[mask, "_date"]

        if ticker_dates.empty:
            done.add(ticker)
            continue

        # --- SEC Filing Sentiment ---
        try:
            sec_df = get_filing_sentiments(ticker)
            if not sec_df.empty:
                aligned_sec = align_sec_sentiment_to_dates(
                    sec_df, ticker_dates.values
                )
                for col in SEC_SENTIMENT_FEATURES:
                    if col in aligned_sec.columns:
                        df.loc[mask, col] = aligned_sec[col].values
        except Exception:
            logger.debug("SEC sentiment failed for %s", ticker)

        # --- GDELT News Sentiment ---
        try:
            aligned_gdelt = align_gdelt_to_dates(
                ticker, ticker_dates.values
            )
            for col in GDELT_FEATURES:
                if col in aligned_gdelt.columns:
                    df.loc[mask, col] = aligned_gdelt[col].values
        except Exception:
            logger.debug("GDELT failed for %s", ticker)

        # Small delay to respect aggregate API rate limits
        time.sleep(0.5)

        done.add(ticker)
        processed_since_checkpoint += 1

        if processed_since_checkpoint % 5 == 0:
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
    logger.info("\nFeature NaN rates:")
    for col in ALL_NEW_FEATURES:
        nan_pct = df[col].isna().mean() * 100
        logger.info("  %s: %.1f%% NaN", col, nan_pct)

    logger.info(
        "Done. %d rows, %d columns saved to %s",
        len(df), len(df.columns), OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
