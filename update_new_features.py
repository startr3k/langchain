"""Incremental dataset update — adds only the 12 new features to the existing CSV.

Loads full_training_data_v2.csv and computes:
- Short interest (3 features) — one yfinance call per ticker
- Options flow (3 features) — one yfinance option chain call per ticker
- Insider transactions (3 features) — one SEC EDGAR call per ticker, time-aligned
- Reddit sentiment (3 features) — Arctic Shift API, time-aligned per ticker

All existing 65 features are preserved unchanged.  Checkpoints every 25 tickers.
"""

import gc
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from stock_predictor.data.short_interest import (
    SHORT_INTEREST_FEATURES,
    get_short_interest,
)
from stock_predictor.data.options_flow import (
    OPTIONS_FLOW_FEATURES,
    get_options_flow,
)
from stock_predictor.data.insider_transactions import (
    INSIDER_FEATURES,
    get_insider_transactions,
    _compute_insider_features_at_date,
)
from stock_predictor.data.reddit_sentiment import (
    REDDIT_SENTIMENT_FEATURES,
    get_reddit_sentiment_history,
)
from textblob import TextBlob

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_CSV = "full_training_data_v2.csv"
OUTPUT_CSV = "full_training_data_v3.csv"
CHECKPOINT_CSV = "full_training_data_v3_checkpoint.csv"
PROGRESS_FILE = "update_progress.txt"
CHECKPOINT_INTERVAL = 25


def load_progress():
    """Load the set of already-processed tickers."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_progress(done_tickers):
    with open(PROGRESS_FILE, "w") as f:
        for t in sorted(done_tickers):
            f.write(t + "\n")


def main():
    logger.info("Loading existing dataset...")
    df = pd.read_csv(INPUT_CSV)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Initialize new columns if not already present
    all_new_cols = SHORT_INTEREST_FEATURES + OPTIONS_FLOW_FEATURES + INSIDER_FEATURES + REDDIT_SENTIMENT_FEATURES
    for col in all_new_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["_date"] = pd.to_datetime(df["_date"])

    tickers = df["Ticker"].unique()
    done = load_progress()
    logger.info("Total tickers: %d, already done: %d", len(tickers), len(done))

    processed_since_checkpoint = 0

    for i, ticker in enumerate(tickers):
        if ticker in done:
            continue

        mask = df["Ticker"] == ticker
        ticker_dates = df.loc[mask, "_date"]
        n_rows = mask.sum()

        try:
            # --- Short interest (cross-sectional snapshot) ---
            si = get_short_interest(ticker)
            if not si.empty:
                for col in SHORT_INTEREST_FEATURES:
                    if col in si.columns:
                        df.loc[mask, col] = si[col].iloc[0]

            # --- Options flow (cross-sectional snapshot) ---
            of = get_options_flow(ticker)
            if not of.empty:
                for col in OPTIONS_FLOW_FEATURES:
                    if col in of.columns:
                        df.loc[mask, col] = of[col].iloc[0]

            # --- Insider transactions (time-aligned) ---
            transactions = get_insider_transactions(ticker)
            if not transactions.empty:
                # Compute per-date insider features
                for idx in df.loc[mask].index:
                    d = df.at[idx, "_date"]
                    feats = _compute_insider_features_at_date(transactions, d)
                    for col, val in feats.items():
                        df.at[idx, col] = val
            else:
                for col in INSIDER_FEATURES:
                    df.loc[mask, col] = 0.0

            # --- Reddit sentiment (time-aligned) ---
            # Fetch once for the full date range, then window per row
            min_date = ticker_dates.min() - pd.Timedelta(days=10)
            max_date = ticker_dates.max() + pd.Timedelta(days=1)
            history = get_reddit_sentiment_history(
                ticker,
                min_date.strftime("%Y-%m-%d"),
                max_date.strftime("%Y-%m-%d"),
            )
            if not history.empty:
                history["date"] = pd.to_datetime(history["date"]).dt.tz_localize(None)
                for idx in df.loc[mask].index:
                    d = df.at[idx, "_date"]
                    window_start = d - pd.Timedelta(days=7)
                    window = history[
                        (history["date"] >= window_start) & (history["date"] <= d)
                    ]
                    n_mentions = len(window)
                    if n_mentions > 0:
                        df.at[idx, "reddit_mention_count_7d"] = float(n_mentions)
                        df.at[idx, "reddit_mean_sentiment_7d"] = round(window["polarity"].mean(), 4)
                        df.at[idx, "reddit_bullish_ratio_7d"] = round((window["polarity"] > 0).sum() / n_mentions, 4)
                    else:
                        df.at[idx, "reddit_mention_count_7d"] = 0.0
                        df.at[idx, "reddit_mean_sentiment_7d"] = 0.0
                        df.at[idx, "reddit_bullish_ratio_7d"] = 0.0
            else:
                df.loc[mask, "reddit_mention_count_7d"] = 0.0
                df.loc[mask, "reddit_mean_sentiment_7d"] = 0.0
                df.loc[mask, "reddit_bullish_ratio_7d"] = 0.0

        except Exception:
            logger.exception("Error processing %s", ticker)

        done.add(ticker)
        processed_since_checkpoint += 1

        if processed_since_checkpoint % 10 == 0:
            logger.info(
                "Progress: %d/%d tickers (%.1f%%) — %s done",
                len(done), len(tickers), len(done) / len(tickers) * 100, ticker,
            )

        # Checkpoint
        if processed_since_checkpoint >= CHECKPOINT_INTERVAL:
            logger.info("Checkpointing at %d/%d...", len(done), len(tickers))
            save_progress(done)
            df.to_csv(CHECKPOINT_CSV, index=False)
            processed_since_checkpoint = 0
            gc.collect()

        # Rate limit — be gentle with APIs
        time.sleep(0.2)

    # Final save
    logger.info("Saving final dataset...")
    save_progress(done)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(
        "Done. %d rows, %d columns saved to %s",
        len(df), len(df.columns), OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
